# SPDX-License-Identifier: Apache-2.0
"""gemm_2x2wave_rowmajor — IDENTICAL to `gemm_2x2wave.py` except the wave grid is
numbered ROW-MAJOR instead of column-major.  Side-by-side proof that the wave
numbering is a free relabeling: it changes which wave id sits in which seat, but
NOT the computed result.

gemm_2x2wave.py uses stride (1, WAVE_M, 0) -> ids run DOWN columns (M fastest):

        N-wave 0      N-wave 1                 N-wave 0      N-wave 1
      +----------+----------+                +----------+----------+
M-wave|  wave 0  |  wave 2  |     this file  |  wave 0  |  wave 1  |
  0   |          |          |     stride     |          |          |
      +----------+----------+   (WAVE_N,1,0) +----------+----------+
M-wave|  wave 1  |  wave 3  |      ------->  |  wave 2  |  wave 3  |
  1   |          |          |    ids run     |          |          |
      +----------+----------+    ACROSS rows +----------+----------+
        (column-major)                          (row-major, here)

The 2x2 PHYSICAL arrangement (2 waves along M, 2 along N) is fixed by the SHAPE
(WAVE_M, WAVE_N, 1) and is the same in both files. Only the STRIDE differs:
  - column-major: (1, WAVE_M, 0) -> wave_id = m*1 + n*WAVE_M  (M varies fastest)
  - row-major:    (WAVE_N, 1, 0) -> wave_id = m*WAVE_N + n*1  (N varies fastest)

Because make_tiled_mma re-derives every partition/replication layout from the
wave grid, the tiled copies route the correct A-rows / B-data to whichever wave
holds each seat. Result is byte-for-byte identical to the column-major version
(both give max abs err = 0.0).

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_2x2wave_rowmajor.py
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# One 16x16x16 bf16 atom (the shape fx.gemm supports), tiled 2x2 across waves.
M_MMA, N_MMA, K_MMA = 16, 16, 16
WAVE_M, WAVE_N = 2, 2                      # 2 warps along A (M), 2 along B (N)
TILE_M, TILE_N = WAVE_M * M_MMA, WAVE_N * N_MMA   # 32 x 32 output tile
K = 64                                     # contraction length -> K // K_MMA steps
NUM_THREADS = WAVE_M * WAVE_N * 64         # 4 waves * 64 lanes = 256


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # This block's tiles, now 32-wide. Keep the K-tile dim on A/B for the loop.
    gA_k = fx.flat_divide(A, (TILE_M, K_MMA))[None, None, 0, None]   # (32, 16, nK)
    gB_k = fx.flat_divide(B, (TILE_N, K_MMA))[None, None, 0, None]   # (32, 16, nK)
    gC = fx.flat_divide(C, (TILE_M, TILE_N))[None, None, 0, 0]       # (32, 32)

    # The ONLY change vs gemm_2x2wave.py: ROW-MAJOR wave numbering.
    #   shape  (2,2,1) = 2 waves along M, 2 along N, 1 along K   (same as before)
    #   stride (2,1,0) = M-stride 2, N-stride 1 -> waves numbered row-major
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(M_MMA, N_MMA, K_MMA, fx.BFloat16)),
        fx.make_layout((WAVE_M, WAVE_N, 1), (WAVE_N, 1, 0)),
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # tiled copies: global -> register. Replication across the shared wave dim
    # (A across N-waves, B across M-waves) is derived from tiled_mma for you.
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((M_MMA * K_MMA // 64) * 16), fx.BFloat16)
    tcA = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)
    thr_gA = tcA.partition_S(gA_k)
    thr_gB = tcB.partition_S(gB_k)

    frag_A = thr_mma.make_fragment_A(gA_k[None, None, 0])
    frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0)

    for kt in fx.range_constexpr(K // K_MMA):
        fx.copy(cp_ab, thr_gA[None, None, None, kt], tcA.retile(frag_A))
        fx.copy(cp_ab, thr_gB[None, None, None, kt], tcB.retile(frag_B))
        fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)

    cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcC = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
    fx.copy(cp_c, tcC.retile(frag_C), tcC.partition_S(gC))


@flyc.jit
def run(A, B, C, stream: fx.Stream = fx.Stream(None)):
    gemm(A, B, C).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(TILE_M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(TILE_N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(TILE_M, TILE_N, dtype=torch.float32, device="cuda")
    run(A, B, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    err = (C - A.float() @ B.float().T).abs().max().item()
    print(f"2x2 wave ROW-MAJOR numbering (fx.gemm): max abs err = {err:.5f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
