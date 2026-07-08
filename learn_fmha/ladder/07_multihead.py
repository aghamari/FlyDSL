# SPDX-License-Identifier: Apache-2.0
"""Ladder 07 — MULTIHEAD (Lessons 22b/22c, ENDPOINT): a head grid that fills the CUs.

FINAL rung. Everything in 06 (multiwave + column-V + register-P + cooperative-LDS, fx.gemm
GEMMs) plus a HEAD dimension. Occupancy is a GRID-MAPPING property: a single head's
ceil(sq/BM) workgroups can't fill 80 CUs, but nq heads give grid = nq * ceil(sq/BM), which
does. We launch a 2-D grid (block_idx.x = q-tile, block_idx.y = head) and offset Q/K/V/O by
the head. This lifts the starved single-head throughput to a realistic wall-clock number.

Inputs are per-head: Q[nq,sq,HD], K[nq,sk,HD], V[nq,HDV,sk] (column-major), O[nq,sq,HDV].
Per-tensor fp8 scales are shared across heads (kept as constexpr for simplicity).

This is the full layout-algebra endpoint of the ladder — the sibling of lesson_22c.

Run one shape:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/07_multihead.py 1024 1024 1 8
All shapes:     HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/07_multihead.py
"""
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench  # noqa: E402

HD = 128
HDV = 128
KSTEPS = HD // 16    # GEMM1 hd contraction (32x32x16 -> K=16): 8 steps
DT = HDV // 32       # GEMM2 d-tiles: 4
BN = 32              # kv per MFMA tile
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
WAVE_ROWS = 32       # each wave owns 32 q-rows
BM = NWAVES * WAVE_ROWS
NTHREADS = NWAVES * 64
LOG2E = 1.4426950408889634
_LGKMCNT0 = 0xC07F   # s_waitcnt mask: drain LDS/bpermute before consuming

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="ladder07_smem")
_K_BYTES = BN * HD      # K tile [kv, hd] fp8
_V_BYTES = HDV * BN     # V tile [d, kv] fp8 (column-major)
_K_OFF = 0
_V_OFF = _K_OFF + _K_BYTES
_alloc.ptr = _V_OFF + _V_BYTES


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    tid = fx.Int32(fx.thread_idx.x)
    wave = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)         # wave-local lane
    q_local = lane % fx.Int32(32)     # C column / q index (32x32x16)
    half = lane // fx.Int32(32)       # 0/1: which kv half-group
    blk = fx.Int32(fx.block_idx.x)
    head = fx.Int32(fx.block_idx.y)   # 2-D grid: y = head
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    f32t = fx.typing.T.f32
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)

    # per-head base offsets (Q/K/O row-major [nq,s,hd]; V column-major [nq,hd,sk]).
    q_head_off = head * sq_i * fx.Int32(HD)
    k_head_off = head * sk_i * fx.Int32(HD)
    v_head_off = head * fx.Int32(HDV) * sk_i
    o_head_off = head * sq_i * fx.Int32(HDV)

    # workgroup blk owns q-rows [blk*BM, blk*BM+BM); each wave owns its 32-row slice.
    wave_q0 = blk * fx.Int32(BM) + wave * fx.Int32(WAVE_ROWS)
    qrow = wave_q0 + q_local
    qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # ---- layout-algebra MMA: one typed atom -> tiled_mma -> fx.gemm for both GEMMs ----
    # Shape-only fragment donors (K/O are 3-D [nq,...] now, so take flat (32,16)/(32,32) views).
    g_f8 = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(K)), fx.make_layout((32, 16), (16, 1)))
    g_c = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(O)), fx.make_layout((32, 32), (32, 1)))
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.typing.T.f8))
    tiled_mma = fx.make_tiled_mma(_mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(lane)   # MMA is per-wave: use the wave-local lane

    # preload this lane's Q fragment: Q[qrow, hd = ks*16 + half*8 + e] (8 fp8) per k-step.
    q_frag = []
    for ks in fx.range_constexpr(KSTEPS):
        off = q_head_off + qrow_safe * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
        w = fx.buffer_ops.buffer_load(rQ, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
        q_frag.append(fx.Vector(w).bitcast(fx.Float8E4M3FNUZ))
    qk_descale = fx.Float32(qd_s * kd_s * sm_scale)

    sk_m1 = sk_i - fx.Int32(1)
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1

    kv_local = lane % fx.Int32(32)
    # causal kv-tile cap (Lesson 14): skip tiles that are fully masked for this q-tile.
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)
    if fx.const_expr(causal == 0):
        n_kv = n_kv_full
    else:
        q_max = blk * fx.Int32(BM) + fx.Int32(BM - 1)
        kv_max = q_max + (sk_i - sq_i)
        n_kv_c = (kv_max + fx.Int32(BN)) // fx.Int32(BN)
        n_kv = (n_kv_c < n_kv_full).select(n_kv_c, n_kv_full)

    m0 = fx.Float32(-3.0e38)
    l0 = fx.Float32(0.0)
    o0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]
    init = [m0, l0] + o0
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BN)

        # ---- cooperative load: workgroup stages K[kv,hd] and V[d,kv] into LDS ONCE ----
        k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(_K_BYTES,)).get()
        vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(_V_BYTES,)).get()
        NSLOT_K = BN * (HD // 16)      # 32 kv * 8 hd-groups of 16 = 256
        for p in fx.range_constexpr((NSLOT_K + NTHREADS - 1) // NTHREADS):
            slot = fx.Int32(p * NTHREADS) + tid
            ok = slot < fx.Int32(NSLOT_K)
            slot_s = ok.select(slot, fx.Int32(0))
            kv_row = slot_s // fx.Int32(HD // 16)
            cg = slot_s % fx.Int32(HD // 16)
            kvg = kv0 + kv_row
            kvg_s = (kvg < sk_i).select(kvg, fx.Int32(0))
            k_off = k_head_off + kvg_s * fx.Int32(HD) + cg * fx.Int32(16)
            kw = fx.buffer_ops.buffer_load(rK, k_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)
            fx.Vector(kw).bitcast(fx.Int8).store(k_lds, [fx.Index(kv_row * fx.Int32(HD) + cg * fx.Int32(16))])
        NSLOT_V = HDV * (BN // 16)     # 128 d * 2 kv-groups of 16 = 256
        for p in fx.range_constexpr((NSLOT_V + NTHREADS - 1) // NTHREADS):
            s2 = fx.Int32(p * NTHREADS) + tid
            ok2 = s2 < fx.Int32(NSLOT_V)
            s2s = ok2.select(s2, fx.Int32(0))
            d_row = s2s // fx.Int32(BN // 16)
            kvg16 = (s2s % fx.Int32(BN // 16)) * fx.Int32(16)
            gkv = kv0 + kvg16
            gkv_s = (gkv < sk_i).select(gkv, fx.Int32(0))
            voff = v_head_off + d_row * sk_i + gkv_s   # column-major V[d, kv] for this head
            vw = fx.buffer_ops.buffer_load(rV, voff // fx.Int32(4), vec_width=4, dtype=fx.Int32)
            fx.Vector(vw).bitcast(fx.Int8).store(vt_lds, [fx.Index(d_row * fx.Int32(BN) + kvg16)])
        fx.gpu.barrier()

        # ---- GEMM1: S[kv,q] = K @ Q^T (K read from shared LDS tile) ----
        frag_S = thr_mma.make_fragment_C(g_c)
        frag_S.fill(0)
        frag_K = thr_mma.make_fragment_A(g_f8)
        frag_Q = thr_mma.make_fragment_B(g_f8)
        for ks in fx.range_constexpr(KSTEPS):
            k_elem = kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
            kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_elem)])
            frag_K.store(fx.Vector(kv8).bitcast(fx.Float8E4M3FNUZ))
            frag_Q.store(q_frag[ks])
            fx.gemm(_mma_atom, frag_S, frag_K, frag_Q, frag_S)
        sv_raw = frag_S.load()
        # 32x32x16 C-layout: lane holds S[kv=(i//4)*8+half*4+i%4, q=q_local], i=0..15.
        sv = []
        for i in fx.range_constexpr(16):
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            s = fx.Float32(sv_raw[i]) * qk_descale
            sv.append((kv <= eff_bound).select(s, neg_inf))

        # ---- online softmax (baseline: generic Float32.exp2) ----
        m_loc = sv[0]
        for i in fx.range_constexpr(15):
            m_loc = m_loc.maximumf(sv[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        p_vals = []
        l_loc = fx.Float32(0.0)
        for i in fx.range_constexpr(16):
            pv = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap((sv[i] - safe_m) * fx.Float32(LOG2E))))
            p_vals.append(pv)
            l_loc = l_loc + pv
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # ---- register-resident P-transpose (Lesson 12): ds_bpermute across lanes, NO LDS ----
        def _cvt4(v0, v1, v2, v3):   # 4 f32 -> 4 fp8 packed in one i32
            lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, v0.ir_value(), v1.ir_value(), fx.Int32(0).ir_value(), False)
            return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, v2.ir_value(), v3.ir_value(), lo, True)
        q_byte = q_local * fx.Int32(4)
        q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)
        is_h0 = half == fx.Int32(0)
        frag_B2 = thr_mma.make_fragment_B(g_f8)
        frag_A2 = thr_mma.make_fragment_A(g_f8)
        frag_O = thr_mma.make_fragment_C(g_c)
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        p_i64_s = []
        for s in fx.range_constexpr(2):
            pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])
            pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])
            h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
            h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
            h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
            h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
            fx.rocdl.s_waitcnt(_LGKMCNT0)
            w0 = is_h0.select(h0_b0, h0_b1)
            w1 = is_h0.select(h1_b0, h1_b1)
            p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32))
        for dt in fx.range_constexpr(DT):
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            frag_O.store(fx.Vector(o_acc[dt]) * corr_vec)   # rescale running accumulator
            for s in fx.range_constexpr(2):
                # COLUMN-V in LDS: vt_lds is [d, kv] -> 8 contiguous kv for fixed d = one wide ds_read.
                v_elem = d_col * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_elem)])
                frag_A2.store(fx.Vector(vv8).bitcast(fx.Float8E4M3FNUZ))
                frag_B2.store(fx.Vector(p_i64_s[s]).bitcast(fx.Float8E4M3FNUZ))
                fx.gemm(_mma_atom, frag_O, frag_A2, frag_B2, frag_O)
            o_acc[dt] = frag_O.load()
        fx.gpu.barrier()   # done reading the shared K/V tile before the next iter overwrites it
        st = yield [m_new, l_run] + o_acc

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    out_scale = fx.Float32(vd_s) * inv_l
    in_b = qrow < sq_i
    if in_b:
        for dt in fx.range_constexpr(DT):
            ov = fx.Vector(o_acc[dt]) * fx.Vector.filled(16, out_scale, fx.Float32)
            ov_bf = fx.Vector(ov).to(fx.BFloat16)
            for j in fx.range_constexpr(4):
                d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                v4 = fx.Vector.from_elements([fx.Vector(ov_bf)[j * 4 + e] for e in range(4)], fx.BFloat16)
                o_idx = fx.Int32(o_head_off + qrow_safe * fx.Int32(HDV) + d)
                fx.buffer_ops.buffer_store(v4.ir_value(), rO, o_idx.ir_value())


@flyc.jit
def run_attn(Q, K, V, O, qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
             sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
             n_qtiles: fx.Int32, nq: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, sm_scale, causal).launch(
        grid=(n_qtiles, nq), block=(NWAVES * 64,), stream=stream)


def _run_one(sq, sk, causal, nq=8):
    torch.manual_seed(0)
    Qf = torch.randn(nq, sq, HD)
    Kf = torch.randn(nq, sk, HD)
    Vf = torch.randn(nq, sk, HDV)
    Qq, qd = _bench.quant(Qf)
    Kq, kd = _bench.quant(Kf)
    Vq, vd = _bench.quant(Vf)
    O = torch.zeros(nq, sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD ** 0.5
    n_qtiles = (sq + BM - 1) // BM
    Qc = Qq.cuda()
    Kc = Kq.cuda()
    Vc = Vq.transpose(1, 2).contiguous().cuda()   # per-head COLUMN-major V: [nq, d, kv]

    def enqueue(strm):
        run_attn(Qc, Kc, Vc, O, qd, kd, vd, sq, sk, sm, causal, n_qtiles, nq, stream=strm)

    enqueue(torch.cuda.current_stream())
    torch.cuda.synchronize()
    # per-head reference
    ref = torch.stack([
        _bench.reference(Qq[h], qd, Kq[h], kd, Vq[h], vd, sm, causal, sq, sk) for h in range(nq)
    ])
    ok = _bench.check(f"07_multihead sq={sq} sk={sk} nq={nq} causal={causal}", O, ref)
    us = _bench.graph_us(enqueue)
    print(f"    graph {us:.2f} us   {_bench.tflops(sq, sk, causal, us, nheads=nq):.2f} TFLOPS")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        nq = int(sys.argv[4]) if len(sys.argv) >= 5 else 8
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), nq)
    else:
        import subprocess
        # correctness (small) + endpoint perf (nq heads * q-tiles fills the 80 CUs)
        for shp in [(256, 256, 1, 8), (256, 256, 0, 8), (1024, 1024, 1, 8), (2048, 2048, 1, 8)]:
            subprocess.run([sys.executable, __file__, *map(str, shp)], check=False)
