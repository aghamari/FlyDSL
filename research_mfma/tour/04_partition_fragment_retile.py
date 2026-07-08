# SPDX-License-Identifier: Apache-2.0
"""Tour 04 — the per-lane objects: partition_S vs make_fragment vs retile.

Inside a real device kernel (so thread_idx / device memrefs exist), build the tiled MMA and
the A-matched tiled copy, then print THREE things for lane `tid` (fire at trace time):

  thr_gA = tcA.partition_S(gA)      # this lane's SOURCE view of global A  (a Tensor view)
  frag_A = thr_mma.make_fragment_A  # this lane's REGISTER fragment (MMA operand layout)
  tcA.retile(frag_A)                # frag_A re-viewed in the COPY's layout (for fx.copy)

Compare the three layouts: partition_S = where I read; make_fragment = my registers;
retile = the fragment relabeled so a copy can fill it.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/tour/04_partition_fragment_retile.py
"""
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

M_MMA, N_MMA, K_MMA = 16, 16, 16
WAVE_M, WAVE_N = 2, 2
TILE_M, TILE_N = WAVE_M * M_MMA, WAVE_N * N_MMA   # 32 x 32
K = 32                                            # -> nK = 2 K-steps
NUM_THREADS = WAVE_M * WAVE_N * 64                # 256


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def demo(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    C = fx.rocdl.make_buffer_tensor(C)

    gA = fx.flat_divide(A, (TILE_M, K_MMA))[None, None, 0, None]   # (32, 16, nK=2)
    gC = fx.flat_divide(C, (TILE_M, TILE_N))[None, None, 0, 0]     # (32, 32)

    tm = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(M_MMA, N_MMA, K_MMA, fx.BFloat16)),
        fx.make_layout((WAVE_M, WAVE_N, 1), (1, WAVE_M, 0)),
    )
    thr_mma = tm.thr_slice(tid)

    cp = fx.make_copy_atom(fx.rocdl.BufferCopy(64), fx.BFloat16)
    tcA = fx.make_tiled_copy_A(cp, tm).get_slice(tid)

    thr_gA = tcA.partition_S(gA)                    # source view (Tensor)
    frag_A = thr_mma.make_fragment_A(gA[None, None, 0])   # register fragment
    print("thr_gA   = tcA.partition_S(gA)   :", thr_gA)
    print("frag_A   = thr_mma.make_fragment_A:", frag_A)
    print("retile   = tcA.retile(frag_A)    :", tcA.retile(frag_A))
    # keep frag_C alive so the kernel is well-formed
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0)


@flyc.jit
def run(A, B, C, stream: fx.Stream = fx.Stream(None)):
    demo(A, B, C).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


if __name__ == "__main__":
    A = torch.randn(TILE_M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(TILE_N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(TILE_M, TILE_N, dtype=torch.float32, device="cuda")
    run(A, B, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
