# SPDX-License-Identifier: Apache-2.0
"""gemm_simple — the SAME idiomatic GEMM as `minimal_tiled_gemm.py`, but kept
deliberately UNIFORM for learning.

The only thing different from `minimal_tiled_gemm.py` is the inner MMA call:
instead of dropping to a raw MFMA opcode (which we had to do because `fx.gemm`
can't emit `32x32x8` on CDNA3), here we fix the instruction to `16x16x16` — the
one shape `fx.gemm` fully supports — and use `fx.gemm` directly. No `bitcast`,
no raw `fx.rocdl.mfma_*` intrinsic, no `acc_n` width to track.

That makes the whole kernel a clean 6-step recipe:
  1. make_mma_atom         -> name the instruction (defines A/B/C lane layouts)
  2. make_tiled_mma        -> wrap atom + wave tiling (one wave, no repeat here)
  3. thr_slice(tid)        -> this lane's view
  4. make_fragment_{A,B,C} -> registers in the layout the MMA wants
  5. tiled copies          -> move global -> register fragments
  6. fx.gemm(...)          -> the matrix-multiply-accumulate itself

ONE wavefront (64 lanes) computes ONE 16x16 output tile, looping over K.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_simple.py
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# The one MFMA shape fx.gemm supports on gfx942 bf16. Fixed on purpose.
M_MMA, N_MMA, K_MMA = 16, 16, 16
K = 64   # contraction length = K // K_MMA = 4 MFMA K-steps


@flyc.kernel(known_block_size=[64, 1, 1])
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # This block's tiles. Keep the K-tile dim on A/B for the loop.
    gA_k = fx.flat_divide(A, (M_MMA, K_MMA))[None, None, 0, None]   # (m, k, nK)
    gB_k = fx.flat_divide(B, (N_MMA, K_MMA))[None, None, 0, None]   # (n, k, nK)
    gC = fx.flat_divide(C, (M_MMA, N_MMA))[None, None, 0, 0]        # (m, n)

    # (1)(2)(3) name the instruction, tile it across waves, slice to this lane.
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(M_MMA, N_MMA, K_MMA, fx.BFloat16)),
        fx.make_layout((1, 1, 1), (0, 0, 0)),  # one wave owns the tile, no repeat
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # (5a) tiled copies: global -> register, in the fragment layout the MMA wants.
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((M_MMA * K_MMA // 64) * 16), fx.BFloat16)
    tcA = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)
    thr_gA = tcA.partition_S(gA_k)
    thr_gB = tcB.partition_S(gB_k)

    # (4) register fragments in the exact A/B/C layouts derived from tiled_mma.
    frag_A = thr_mma.make_fragment_A(gA_k[None, None, 0])
    frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0)

    # K-loop: (5b) copy this K-slice into registers, then (6) one fused MMA.
    for kt in fx.range_constexpr(K // K_MMA):
        fx.copy(cp_ab, thr_gA[None, None, None, kt], tcA.retile(frag_A))
        fx.copy(cp_ab, thr_gB[None, None, None, kt], tcB.retile(frag_B))
        fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)

    # register -> global. 32b copy so the strided C fragment stores correctly.
    cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcC = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
    fx.copy(cp_c, tcC.retile(frag_C), tcC.partition_S(gC))


@flyc.jit
def run(A, B, C, stream: fx.Stream = fx.Stream(None)):
    gemm(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(M_MMA, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N_MMA, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M_MMA, N_MMA, dtype=torch.float32, device="cuda")
    run(A, B, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    err = (C - A.float() @ B.float().T).abs().max().item()
    print(f"16x16x16 (fx.gemm): max abs err = {err:.5f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
