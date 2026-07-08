# SPDX-License-Identifier: Apache-2.0
"""gemm_2x2wave — the SAME uniform GEMM as `gemm_simple.py`, but with a 2x2 wave
grid instead of a single wave.  For learning what the wave layout does.

gemm_simple.py:  ONE wave (64 lanes)  -> one 16x16 atom -> a 16x16 output tile.
this file:       FOUR waves (256 lanes) arranged 2 (along M) x 2 (along N):

        N-wave 0      N-wave 1
      +------------+------------+
M-wave|  wave 0    |   wave 2   |   each wave owns ONE 16x16 atom
  0   | (rows 0-15,| (rows 0-15,|   so the block computes a 32x32 tile
      |  cols 0-15)|  cols16-31)|
      +------------+------------+
M-wave|  wave 1    |   wave 3   |
  1   |(rows16-31, |(rows16-31, |
      | cols 0-15) | cols16-31) |
      +------------+------------+

"2 warps vertical in A" = the 2 M-waves: they load DIFFERENT A rows.
"2 warps vertical in B" = the 2 N-waves: they load DIFFERENT B rows (B is [N,K]).
A is SHARED across the 2 N-waves, B is SHARED across the 2 M-waves — the tiled
copies handle that replication for you; you only declare the wave grid.

The ONLY change from gemm_simple.py is:
  - the wave layout: (1,1,1) -> (2,2,1) with stride (1,2,0)
  - the tile sizes:  16x16   -> 32x32   (2 atoms along each of M, N)
  - the launch block: 64 -> 256 threads (4 waves)
Everything else (fx.gemm, fragments, tiled copies) is byte-for-byte identical,
because make_tiled_mma re-derives every layout from the new wave grid.

NOTE on repetition: here each wave does exactly ONE atom (tile == waves*atom, no
repeat). To add repetition, just enlarge the tile (e.g. TILE_M=64) — the fragment
gains an extra mode and each wave loops the atom; no other code changes.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_2x2wave.py
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

import flydsl
from flydsl.utils.env import DebugEnvManager, RuntimeEnvManager
from flydsl._mlir import ir
import os

# # ---- debug preamble (from the slide, cache forced off) ----
# DebugEnvManager.enable_debug_info = True
# DebugEnvManager.dump_asm = True
# DebugEnvManager.dump_ir = True
# DebugEnvManager.dump_dir = os.path.join(os.path.dirname(__file__), "vadd_dbg")
# ir._globals.register_traceback_file_inclusion(__file__)
# ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
# ir._globals.set_loc_tracebacks_frame_limit(40)
# ir._globals.set_loc_tracebacks_enabled(True)
# RuntimeEnvManager.enable_cache = False
# os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"


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

    # The ONE change vs gemm_simple: a 2x2 wave grid instead of (1,1,1).
    #   shape  (2,2,1) = 2 waves along M, 2 along N, 1 along K
    #   stride (1,2,0) = M-stride 1, N-stride 2 -> waves numbered column-major
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(M_MMA, N_MMA, K_MMA, fx.BFloat16)),
        fx.make_layout((WAVE_M, WAVE_N, 1), (1, WAVE_M, 0)),
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
    print(f"2x2 wave 16x16x16 (fx.gemm): max abs err = {err:.5f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
