# SPDX-License-Identifier: Apache-2.0
"""Lesson 04 — Softmax over the scores (the cross-lane reduction).

Goal: given S = [kv, q] from Lesson 03, compute per-query softmax weights
P[kv, q] = exp(S[kv,q] - max_kv S) / sum_kv exp(...). The reduction is over kv FOR EACH
query q. This lesson is about HOW a reduction works when the values you reduce live in
different LANES — the single trickiest mechanic in attention.

This lesson does softmax over ONE 16-kv tile (the whole S fits in registers). The
"online"/streaming version that folds many kv-tiles with a running max/sum is Lesson 06
(it needs the runtime kv-loop). The math here is the per-tile core of that.

### Two halves, two idioms (this is the whole point of the lesson)
Attention is a GEMM followed by a reduction, and the layout-algebra skill uses a DIFFERENT
tool for each:

  * **GEMM1 (S = K @ Qᵀ)** is a matmul, so it is written in the full layout-algebra style
    (skill §2/§6, `research_mfma/minimal_tiled_gemm.py`): `flat_divide` carves the global
    tiles, `make_tiled_mma` derives the A/B/C fragment layouts, `make_tiled_copy_{A,B,C}`
    moves global↔register, and `fx.gemm` issues the MFMA. There is NO hand-written
    `mn*HD + k0` lane math and no raw opcode/bitcast — the partitions compute every address.
    (Per the skill, bf16 on gfx942 uses the 16×16×16 atom + `fx.gemm`; the raw `..._1k`
    i16-bitcast intrinsic is only needed for the 32×32×8 atom.)

  * **The softmax itself is a REDUCTION**, and the skill's §7 rule is explicit: reductions
    stay `buffer_load`/register + warp `shuffle_xor` — tiled-copy / `zipped_divide` is for
    GEMM and transpose, NOT reductions (see `row_reduction_idiomatic.py`). So the cross-lane
    max/sum stays DIRECT: the intra-lane stage uses `fx.Vector.reduce`, the cross-lane stage
    is a `shuffle_xor` butterfly. The lane-shuffle mechanic is exactly what this lesson
    teaches, and layout algebra would only hide it.

### Where the kv values live (from Lesson 01's C-fragment layout)
`make_tiled_mma` lays S out exactly as Lesson 01 derived by hand: lane
(k_outer=lane//16, mn=lane%16) holds the 4 values S[kv = k_outer*4 + e, q = mn] for e=0..3
in its C-fragment. So for a FIXED query q=mn, its 16 kv values are split:
  - 4 of them are in THIS lane's own C-fragment registers (e=0..3, this lane's k_outer),
  - the other 12 are in lanes {mn+16, mn+32, mn+48} (the other k_outer groups).

### The reduction in two stages
1. **Intra-lane:** reduce this lane's own 4 registers — `fx.Vector.reduce("max"|"add")`.
2. **Cross-lane:** combine the 4 k_outer groups (lanes differing only in bits 4 and 5 of
   the lane id). `shuffle_xor(mask, width)` lets a lane read another lane's value where
   `other_lane = my_lane XOR mask`. k_outer = lane//16, so XOR 16 flips k_outer bit0
   (0<->1, 2<->3) and XOR 32 flips k_outer bit1 (0<->2, 1<->3). Doing both merges all 4
   groups. After the two shuffles, every lane in column mn holds the SAME reduced value.

This is exactly the production pattern (real kernel uses mfma_32x32x16 -> 16 regs/lane +
ONE shuffle_xor(32); here 4 regs/lane + TWO shuffles because the 16-wide MFMA splits kv
across 4 lane-groups instead of 2).

### Numerical stability
Subtract the max before exp (standard). We also carry the all-masked guard idea: if the
max is -inf (no valid kv), force it to 0 so exp(-inf - 0) = 0 instead of NaN. Not needed
for this dense tile but introduced here because Lesson 06's causal masking will need it.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_04_softmax.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BQ = 16  # queries in the tile
BKV = 16  # keys in the tile
HD = 64
MFMA_K = 16  # the MFMA's K per step; HD // MFMA_K = 4 K-steps
LOG2E = 1.4426950408889634
NEG_GUARD = -1.0e38  # "-inf" sentinel for the all-masked guard


@flyc.kernel(known_block_size=[64, 1, 1])
def softmax_kernel(Q: fx.Tensor, K: fx.Tensor, P: fx.Tensor, sm_scale: fx.Constexpr[float]):
    tid = fx.thread_idx.x

    # ── GEMM1 in the layout-algebra style: S[kv, q] = K @ Qᵀ ──────────────────
    # A = K[kv, hd] (m=kv), B = Q[q, hd] (n=q); the MFMA computes A @ Bᵀ -> S[kv, q].
    K = fx.rocdl.make_buffer_tensor(K)
    Q = fx.rocdl.make_buffer_tensor(Q)
    P = fx.rocdl.make_buffer_tensor(P)

    # flat_divide carves the MFMA-shaped tiles; fx.slice picks the single block tile
    # (None keeps a mode, an int selects it — same op the bracket sugar lowers to).
    gK = fx.slice(fx.flat_divide(K, (BKV, MFMA_K)), (None, None, 0, None))  # (kv, k, nK=4)
    gQ = fx.slice(fx.flat_divide(Q, (BQ, MFMA_K)), (None, None, 0, None))   # (q,  k, nK=4)
    gP = fx.slice(fx.flat_divide(P, (BKV, BQ)), (None, None, 0, 0))         # (kv, q)

    # One MFMA atom, one wave (atom-layout (1,1,1)); the tiled MMA derives the fragments.
    # Per the layout-algebra skill, bf16 on gfx942 uses the 16x16x16 atom + `fx.gemm`
    # (the raw `..._1k` i16-bitcast intrinsic is reserved for the 32x32x8 case).
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(BKV, BQ, MFMA_K, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    # Tiled copies matched to the MMA layout move global -> register fragments.
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((BKV * MFMA_K // 64) * 16), fx.BFloat16)
    tcK = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcQ = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)

    thr_gK = tcK.partition_S(gK)
    thr_gQ = tcQ.partition_S(gQ)
    frag_K = thr_mma.make_fragment_A(fx.slice(gK, (None, None, 0)))
    frag_Q = thr_mma.make_fragment_B(fx.slice(gQ, (None, None, 0)))
    frag_S = thr_mma.make_fragment_C(gP)
    frag_S.fill(0)

    for kt in fx.range_constexpr(HD // MFMA_K):
        fx.copy(cp_ab, fx.slice(thr_gK, (None, None, None, kt)), tcK.retile(frag_K))
        fx.copy(cp_ab, fx.slice(thr_gQ, (None, None, None, kt)), tcQ.retile(frag_Q))
        fx.gemm(mma_atom, frag_S, frag_K, frag_Q, frag_S)

    # frag_S now holds this lane's 4 scores: S[kv = k_outer*4+e, q = mn], e=0..3.
    # Scale all 4 at once (sm_scale = 1/sqrt(hd)); fold it in before softmax.
    sv = fx.Vector(frag_S.load()) * fx.Vector.filled(4, sm_scale, fx.Float32)

    # ── softmax over kv (the REDUCTION — stays direct, skill §7) ───────────────
    # 1) intra-lane max over this lane's own 4 kv
    m = fx.Float32(sv.reduce("max"))
    # 2) cross-lane: merge the 4 k_outer groups (lanes differ in bits 4,5): XOR 16 then 32
    for mask in (16, 32):
        m = m.maximumf(m.shuffle_xor(fx.Int32(mask), fx.Int32(64)))
    # all-masked guard (harmless here; needed once causal masking arrives in Lesson 06)
    safe_m = (m < fx.Float32(NEG_GUARD)).select(fx.Float32(0.0), m)

    # exp via fast rocdl.exp2: exp(x) = 2^(x*log2e)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((fx.Float32(sv[e]) - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
    pv = fx.Vector.from_elements(p, fx.Float32)
    # sum over kv: intra-lane reduce then the same cross-lane butterfly
    l = fx.Float32(pv.reduce("add"))
    for mask in (16, 32):
        l = l + l.shuffle_xor(fx.Int32(mask), fx.Int32(64))
    inv_l = fx.Float32(1.0) / l

    # ── store P[kv, q] through the tiled copy (layout algebra, no manual index) ─
    pw = fx.Vector.from_elements([p[e] * inv_l for e in range(4)], fx.Float32)
    frag_S.store(pw)
    cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcP = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
    fx.copy(cp_c, tcP.retile(frag_S), tcP.partition_S(gP))


@flyc.jit
def run_softmax(Q, K, P, sm_scale: fx.Constexpr[float], stream: fx.Stream = fx.Stream(None)):
    softmax_kernel(Q, K, P, sm_scale).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(BQ, HD, dtype=torch.bfloat16).cuda()
    K = torch.randn(BKV, HD, dtype=torch.bfloat16).cuda()
    P = torch.zeros(BKV, BQ, dtype=torch.float32).cuda()
    sm = 1.0 / HD**0.5

    run_softmax(Q, K, P, sm, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    S = (K.float() @ Q.float().T) * sm  # [kv, q]
    ref = torch.softmax(S, dim=0)  # softmax over kv
    err = (P - ref).abs().max().item()
    print(f"P = softmax_kv(S)   max abs err = {err:.5f}  ->  {'PASS' if err < 1e-2 else 'FAIL'}")
