# SPDX-License-Identifier: Apache-2.0
"""Lesson 05 — GEMM2 (O = P @ V) and the first FULLY FUSED attention tile (bf16).

Goal: finish attention. We have S=[kv,q] (Lesson 03) and softmax P (Lesson 04) in
registers; now do the second GEMM  O[q, d] = sum_kv P[q, kv] * V[kv, d]  and write the
output. Single tile: BQ=16 queries, BKV=16 keys, HD=64 (QK contraction), hd_v=64 output
(=> DT=4 output d-tiles). One wavefront, NO LDS. End-to-end fused attention.

### GEMM2 orientation
MFMA computes result[m,n] = sum_k A[m,k] B[n,k]. We want O[q,d] = sum_kv P[q,kv] V[kv,d].
Set m=q, n=d, k=kv:  A = P[q, kv],  B[d, kv] = V[kv, d]  (so B = V^T).
  - A operand: lane(k_outer,mn) holds A[q=mn, kv=k_outer*4+e].
  - B operand: lane(k_outer,mn) holds B[d=mn, kv=k_outer*4+e] = V[kv=k_outer*4+e, d=mn]
        (V stored [kv,d] row-major => 4 loads at stride hd_v).
  - C result: lane(k_outer,mn) holds O[q=k_outer*4+e, d=mn].
P doesn't depend on d, so we reuse the same A-fragment for all DT=4 output d-tiles.

### The "P-transpose problem" — and why it DOESN'T bite us here (important!)
GEMM2 needs the A-operand P[q=mn, kv=k_outer*4+e]. Softmax (Lesson 04) left this lane
holding P[kv=k_outer*4+e, q=mn] — which is THE SAME 4 SCALARS. For the 16x16x16 MFMA the
output C-fragment layout (rows grouped in 4s by k_outer) exactly matches the input
A-fragment layout (K grouped in 4s by k_outer), so **P is already in the right registers —
no transpose needed.**

That is a LUCKY property of this MFMA shape. The production fp8 kernel uses mfma_32x32x16,
whose C-fragment scatters kv differently than its A-fragment wants -> there the transpose
is REAL and must be done (through LDS, or via ds_bpermute). We meet that head-on in
Lesson 07 (fp8) and fix it in Lessons 11-12. Teaching point: **whether you pay a transpose
depends on the MFMA shape you chose** — it is not fundamental to attention.

### Idiomatic note (layout algebra)
Both GEMMs use the layout algebra: ONE typed 16x16x16 MMA atom (`make_mma_atom`) feeds a
`make_tiled_mma`, whose derived A/B/C fragment layouts drive `fx.gemm`, and
`make_tiled_copy_{A,B,C}` moves global↔register. The softmax stays DIRECT (Lesson 04 /
layout skill §7: reductions use register + `shuffle_xor`, not tiled copies).

The headline "no-transpose" insight survives the port — just restated in fragment terms:
softmax leaves P in the GEMM1 C-fragment, and for the 16x16x16 MFMA the C-fragment layout
equals the GEMM2 A-fragment layout, so `p_norm[e]` stores straight into `frag_A[e]` with no
transpose (see the `frag_A` block). V is row-major [kv,d], so the kv contraction is a
STRIDED gather: we view V as [d,kv] (`make_view`) and use a scalar copy atom
(`BufferCopy16b`) — exactly the 4 strided loads the hand-mapped code issued. Whether you
pay a real transpose still depends on the MFMA shape (Lessons 07/11/12/17).

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_05_pv_fused.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BQ = 16
BKV = 16
HD = 64       # QK head dim (contraction of GEMM1)
HDV = 64      # V head dim (output width)
KSTEPS = HD // 16
DT = HDV // 16
LOG2E = 1.4426950408889634


@flyc.kernel(known_block_size=[64, 1, 1])
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, sm_scale: fx.Constexpr[float]):
    tid = fx.thread_idx.x
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap

    K = fx.rocdl.make_buffer_tensor(K)
    Q = fx.rocdl.make_buffer_tensor(Q)

    # IDIOMATIC: one typed 16x16x16 bf16 MFMA atom serves BOTH attention GEMMs.
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

    # --- GEMM1 (layout algebra): S[kv, q] = K @ Qᵀ  (A=K m=kv, B=Q n=q, k=hd) ---
    # `make_tiled_mma` derives the A/B/C fragment layouts; tiled copies move global→reg.
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)
    gK = fx.slice(fx.flat_divide(K, (BKV, 16)), (None, None, 0, None))   # (kv, k, nK)
    gQ = fx.slice(fx.flat_divide(Q, (BQ, 16)), (None, None, 0, None))    # (q,  k, nK)
    # S is register-only; borrow a (BKV, BQ) tile purely as a C-fragment shape donor.
    gS = fx.slice(fx.flat_divide(K, (BKV, BQ)), (None, None, 0, 0))      # (kv, q) shape
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((BKV * 16 // 64) * 16), fx.BFloat16)
    tcK = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcQ = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)
    thr_gK = tcK.partition_S(gK)
    thr_gQ = tcQ.partition_S(gQ)
    frag_K = thr_mma.make_fragment_A(fx.slice(gK, (None, None, 0)))
    frag_Q = thr_mma.make_fragment_B(fx.slice(gQ, (None, None, 0)))
    frag_S = thr_mma.make_fragment_C(gS)
    frag_S.fill(0)
    for kt in fx.range_constexpr(KSTEPS):
        fx.copy(cp_ab, fx.slice(thr_gK, (None, None, None, kt)), tcK.retile(frag_K))
        fx.copy(cp_ab, fx.slice(thr_gQ, (None, None, None, kt)), tcQ.retile(frag_Q))
        fx.gemm(mma_atom, frag_S, frag_K, frag_Q, frag_S)
    # frag_S: lane holds S[kv=k_outer*4+e, q=mn], e=0..3 (same layout the manual code had).
    sv = [fx.Float32(frag_S.load()[e]) * fx.Float32(sm_scale) for e in range(4)]

    # --- softmax over kv (Lesson 04) -> p[e] = P[kv=k_outer*4+e, q=mn], normalized ---
    m = sv[0]
    for e in fx.range_constexpr(3):
        m = m.maximumf(sv[e + 1])
    m = m.maximumf(m.shuffle_xor(fx.Int32(16), fx.Int32(64)))
    m = m.maximumf(m.shuffle_xor(fx.Int32(32), fx.Int32(64)))
    safe_m = (m < fx.Float32(-1.0e38)).select(fx.Float32(0.0), m)
    p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[e] - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
    l = p[0]
    for e in fx.range_constexpr(3):
        l = l + p[e + 1]
    l = l + l.shuffle_xor(fx.Int32(16), fx.Int32(64))
    l = l + l.shuffle_xor(fx.Int32(32), fx.Int32(64))
    inv_l = fx.Float32(1.0) / l
    p_norm = [p[e] * inv_l for e in range(4)]

    # --- GEMM2 (layout algebra): O[q, d] = P @ V  (A=P m=q, B=Vᵀ n=d, k=kv) ---
    # A = P is ALREADY register-resident from softmax. For the 16x16x16 MFMA the GEMM1
    # C-fragment layout equals the GEMM2 A-fragment layout, so p_norm[e] drops straight
    # into frag_A[e] — no transpose (the headline lesson, now stated in fragment terms).
    gA = fx.slice(fx.flat_divide(K, (BQ, BKV)), (None, None, 0, 0))       # (q, kv) shape donor
    frag_A = thr_mma.make_fragment_A(gA)
    frag_A.store(fx.Vector.from_elements([p_norm[e].to(fx.BFloat16) for e in range(4)], fx.BFloat16))

    # B = Vᵀ: V is stored [kv, d]; view it as [d, kv] so a tiled copy expresses the
    # kv-contraction gather. kv is strided by HDV there (the "V transpose"), so the copy
    # atom is scalar (BufferCopy16b) — the same 4 strided loads the hand code issued.
    V = fx.rocdl.make_buffer_tensor(V)
    O = fx.rocdl.make_buffer_tensor(O)
    Vt = fx.make_view(fx.get_iter(V), fx.make_layout((HDV, BKV), (1, HDV)))   # [d, kv]
    cp_v = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcB = fx.make_tiled_copy_B(cp_v, tiled_mma).get_slice(tid)
    tcC = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
    for dt in fx.range_constexpr(DT):
        gV = fx.slice(fx.flat_divide(Vt, (16, 16)), (None, None, dt, 0))      # (d, kv) tile dt
        gO = fx.slice(fx.flat_divide(O, (BQ, 16)), (None, None, 0, dt))       # (q, d) tile dt
        frag_B = thr_mma.make_fragment_B(gV)
        fx.copy(cp_v, tcB.partition_S(gV), tcB.retile(frag_B))
        frag_O = thr_mma.make_fragment_C(gO)
        frag_O.fill(0)
        fx.gemm(mma_atom, frag_O, frag_A, frag_B, frag_O)
        fx.copy(cp_c, tcC.retile(frag_O), tcC.partition_S(gO))


@flyc.jit
def run_attn(Q, K, V, O, sm_scale: fx.Constexpr[float], stream: fx.Stream = fx.Stream(None)):
    attn_kernel(Q, K, V, O, sm_scale).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(BQ, HD, dtype=torch.bfloat16).cuda()
    K = torch.randn(BKV, HD, dtype=torch.bfloat16).cuda()
    V = torch.randn(BKV, HDV, dtype=torch.bfloat16).cuda()
    O = torch.zeros(BQ, HDV, dtype=torch.float32).cuda()
    sm = 1.0 / HD**0.5

    run_attn(Q, K, V, O, sm, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    S = (Q.float() @ K.float().T) * sm     # [q, kv]
    P = torch.softmax(S, dim=1)            # over kv
    ref = P @ V.float()                    # [q, d]
    err = (O - ref).abs().max().item()
    print(f"fused attention  max abs err = {err:.4f}  ->  {'PASS' if err < 5e-2 else 'FAIL'}")
