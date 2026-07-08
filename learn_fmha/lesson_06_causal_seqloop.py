# SPDX-License-Identifier: Apache-2.0
"""Lesson 06 — Streaming attention: runtime kv-loop, ONLINE softmax, causal mask.

Goal: real attention processes a long key sequence in TILES, folding each tile into a
running result with the flash-attention "online softmax." We also add the causal mask.
This turns Lesson 05's single-tile kernel into one that handles arbitrary seqlen_k with a
RUNTIME loop. One query tile (BQ=16) vs all kv tiles (BKV=16 each), one wavefront.

### Online softmax (the flash trick)
We can't see all kv before normalizing. So we keep, per query, a running max `m` and
running denominator `l`, and the unnormalized output accumulator `o`. For each kv-tile:
  m_new = max(m_old, max_kv tile)
  corr  = exp(m_old - m_new)            # rescale factor for what we already accumulated
  p     = exp(s - m_new)                # this tile's weights (unnormalized)
  l     = l*corr + sum(p)
  o     = o*corr + p @ V_tile           # rescale old o, add this tile
At the end: O = o / l. This is mathematically exact, never materializes the full S.

### FlyDSL runtime loop
Compile-time loops use range_constexpr (unrolled). A RUNTIME-bounded loop (n_kv depends
on seqlen) uses scf.for via:
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=[m, l, o0..o3]):
        ... use st[0]=m, st[1]=l, st[2+d]=o_d ...
        st = yield [m_new, l_new, o0_new, ...]
The loop-carried state (m, l, o-tiles) MUST be threaded through init= and yield — there
are no mutable captured variables across iterations.

### Causal mask
Query at absolute row `qrow` may attend to key `kv` only if `kv <= qrow + (sk - sq)`.
The `(sk - sq)` offset aligns the diagonal when the key sequence is longer than the query
sequence (prefix KV). Forgetting it silently breaks any case with sk != sq. Masked entries
get score -inf -> p=0. The all-masked guard (Lesson 04) prevents NaN for fully-masked rows.

### Idiomatic style (flydsl-layout-algebra skill)
Both GEMMs use the layout algebra: one typed MFMA atom `fx.make_mma_atom(fx.rocdl.MFMA(16,16,16,
bf16))` feeds a `make_tiled_mma`, whose derived A/B/C fragment layouts drive `fx.gemm`. The global
LOADS stay direct (`create_buffer_resource` + `buffer_load`): the per-lane causal-masked, strided
kv gathers are the skill's "jagged / bespoke register packing" stay-direct case, so tiled-copy
partitioning is deliberately not applied — the masked values are dropped into the MMA fragments by
hand, then `fx.gemm` runs the matmul. The online softmax stays direct too (a reduction, skill §7).

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_06_causal_seqloop.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BQ = 16
BKV = 16
HD = 64
HDV = 64
KSTEPS = HD // 16
DT = HDV // 16
LOG2E = 1.4426950408889634


@flyc.kernel(known_block_size=[64, 1, 1])
def attn_kernel(Q, K, V, O, sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    neg_inf = fx.Float32(-3.0e38)

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # IDIOMATIC (flydsl-layout-algebra skill, "tiled-MMA" recipe): declare the 16x16x16 bf16 MFMA
    # ONCE as a typed layout-API atom; a `make_tiled_mma` then derives the A/B/C fragment layouts
    # that drive BOTH GEMMs through `fx.gemm`. The GLOBAL LOADS stay direct (create_buffer_resource
    # + buffer_load): per-lane causal-masked, strided kv gathers are the skill's "jagged / bespoke
    # register packing" stay-direct case, so tiled-copy partitioning is not applied — the masked
    # values are dropped into the MMA fragments by hand, then `fx.gemm` runs the matmul.
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

    tid = fx.thread_idx.x
    tiled_mma = fx.make_tiled_mma(_mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)
    Qb = fx.rocdl.make_buffer_tensor(Q)
    Ob = fx.rocdl.make_buffer_tensor(O)
    g_bf16 = fx.slice(fx.flat_divide(Qb, (16, 16)), (None, None, 0, 0))   # (16,16) bf16 shape donor
    g_f32 = fx.slice(fx.flat_divide(Ob, (16, 16)), (None, None, 0, 0))    # (16,16) f32  shape donor

    # this lane's query column q = mn (single q-tile at row 0..15). Preload Q fragment (reused).
    q_packs = []
    for ks in fx.range_constexpr(KSTEPS):
        k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
        q_packs.append(fx.buffer_ops.buffer_load(rQ, mn * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16))

    # per-lane causal bound for query qrow = mn: valid kv <= eff_bound
    qrow = mn
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_i - fx.Int32(1)).select(cb, sk_i - fx.Int32(1))
    else:
        eff_bound = sk_i - fx.Int32(1)

    n_kv = (sk_i + fx.Int32(BKV - 1)) // fx.Int32(BKV)
    m0 = fx.Float32(-3.0e38)
    l0 = fx.Float32(0.0)
    o0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(DT)]
    init = [m0, l0] + o0
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BKV)

        # GEMM1 for this kv-tile (fx.gemm): sv[e] = S[kv = kv0 + k_outer*4 + e, q = mn].
        # A=K, B=Q. The K rows are OOB-masked (stay-direct load), then dropped into frag_K;
        # `fx.gemm` accumulates over the KSTEPS hd sub-tiles into the C fragment.
        frag_S = thr_mma.make_fragment_C(g_f32)
        frag_S.fill(0)
        frag_K = thr_mma.make_fragment_A(g_bf16)
        frag_Q = thr_mma.make_fragment_B(g_bf16)
        for ks in fx.range_constexpr(KSTEPS):
            k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
            # In GEMM1 A=K, lane(k_outer,mn) LOADS K row = kv0 + mn (output C row = k_outer*4+e).
            k_row = kv0 + mn
            k_row_safe = (k_row < sk_i).select(k_row, fx.Int32(0))
            k_vec = fx.buffer_ops.buffer_load(rK, k_row_safe * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16)
            frag_K.store(fx.Vector(k_vec))
            frag_Q.store(fx.Vector(q_packs[ks]))
            fx.gemm(_mma_atom, frag_S, frag_K, frag_Q, frag_S)
        # apply scale + causal mask
        s_reg = frag_S.load()
        sv = []
        for e in fx.range_constexpr(4):
            kv = kv0 + k_outer * fx.Int32(4) + fx.Int32(e)
            s = fx.Float32(s_reg[e]) * fx.Float32(sm_scale)
            sv.append((kv <= eff_bound).select(s, neg_inf))

        # online softmax update
        m_loc = sv[0]
        for e in fx.range_constexpr(3):
            m_loc = m_loc.maximumf(sv[e + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(16), fx.Int32(64)))
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(32), fx.Int32(64)))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)

        p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[e] - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
        l_loc = p[0]
        for e in fx.range_constexpr(3):
            l_loc = l_loc + p[e + 1]
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(16), fx.Int32(64))
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(32), fx.Int32(64))
        l_run = l_run * corr + l_loc

        # GEMM2: o += V^T @ P, output O[d=row, q=col=mn]. We put d on the row and q on the COLUMN
        # so the running normalizer l (indexed by this lane's query q=mn) lines up with the output
        # column at epilogue time. (If you instead make q the output ROW, each lane's inv_l would
        # belong to the wrong query — a classic flash-attention orientation bug.)
        # MFMA result[m=d,n=q]=sum_k A[m,k]B[n,k]: A[d,kv]=V[kv,d] (=V^T), B[q,kv]=P[q,kv].
        # B = P[q=mn, kv=k_outer*4+e] is register-resident; drop it straight into frag_B.
        corr4 = fx.Vector.filled(4, fx.Float32(corr), fx.Float32)
        frag_B2 = thr_mma.make_fragment_B(g_bf16)
        frag_B2.store(fx.Vector.from_elements([p[e].to(fx.BFloat16) for e in range(4)], fx.BFloat16))
        frag_A2 = thr_mma.make_fragment_A(g_bf16)
        frag_O = thr_mma.make_fragment_C(g_f32)
        new_o = []
        for dt in fx.range_constexpr(DT):
            # A = Vᵀ: lane(k_outer,mn) holds A[d=mn, kv=k_outer*4+e] = V[kv=k_outer*4+e, d=dt*16+mn].
            # kv is V's strided/OOB-masked axis, so it stays a direct scalar gather into frag_A.
            d_row = fx.Int32(dt * 16) + mn
            a_elems = []
            for e in fx.range_constexpr(4):
                kv = kv0 + k_outer * fx.Int32(4) + fx.Int32(e)
                kv_ok = kv < sk_i
                kv_safe = kv_ok.select(kv, fx.Int32(0))
                vval = fx.buffer_ops.buffer_load(rV, kv_safe * fx.Int32(HDV) + d_row, vec_width=1, dtype=fx.BFloat16)
                a_elems.append(kv_ok.select(vval, fx.BFloat16(0.0)))
            frag_A2.store(fx.Vector.from_elements(a_elems, fx.BFloat16))
            frag_O.store(fx.Vector(o_acc[dt]) * corr4)                 # C = rescaled running accumulator
            fx.gemm(_mma_atom, frag_O, frag_A2, frag_B2, frag_O)       # o_new = Vᵀ @ P + o_resc
            new_o.append(frag_O.load())
        st = yield [m_new, l_run] + new_o

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    # l_run is this lane's normalizer for query q = mn. With O[d=row, q=col], every output element
    # this lane writes has q = mn, so one inv_l applies to all of them.
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    for dt in fx.range_constexpr(DT):
        ov = fx.Vector(o_acc[dt])
        for e in fx.range_constexpr(4):
            d = fx.Int32(dt * 16) + k_outer * fx.Int32(4) + fx.Int32(e)  # C row = d within this dt-block
            q = mn  # C result col = q = mn
            fx.buffer_ops.buffer_store((fx.Float32(ov[e]) * inv_l).ir_value(), rO, (q * fx.Int32(HDV) + d).ir_value())


@flyc.jit
def run_attn(Q, K, V, O, sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    attn_kernel(Q, K, V, O, sq, sk, sm_scale, causal).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    for (sq, sk, causal) in [(16, 64, 1), (16, 64, 0), (16, 48, 1)]:
        Q = torch.randn(sq, HD, dtype=torch.bfloat16).cuda()
        K = torch.randn(sk, HD, dtype=torch.bfloat16).cuda()
        V = torch.randn(sk, HDV, dtype=torch.bfloat16).cuda()
        O = torch.zeros(sq, HDV, dtype=torch.float32).cuda()
        sm = 1.0 / HD**0.5
        run_attn(Q, K, V, O, sq, sk, sm, causal, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        S = (Q.float() @ K.float().T) * sm
        if causal:
            qi = torch.arange(sq).view(-1, 1).cuda()
            ki = torch.arange(sk).view(1, -1).cuda()
            S = S.masked_fill(ki > qi + (sk - sq), float("-inf"))
        ref = torch.softmax(S, dim=1) @ V.float()
        err = (O - ref).abs().max().item()
        print(f"sq={sq} sk={sk} causal={causal}  err={err:.4f}  {'PASS' if err < 5e-2 else 'FAIL'}")
