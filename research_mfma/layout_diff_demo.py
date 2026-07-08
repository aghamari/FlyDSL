# SPDX-License-Identifier: Apache-2.0
"""layout_diff_demo — a REAL FlyDSL demo that prints the actual layouts, so you
can see *why* the MMA fragment and the tiled-copy disagree (and what retile
fixes).

Everything printed here is STATIC (known at trace/compile time), so we just
`print(...)` from inside the kernel body while FlyDSL is tracing it.  Nothing is
launched for the layout dump; we also run the kernel afterwards to prove the
retile handoff is correct.

Two different consumers touch the SAME per-thread A registers:
  * fx.gemm  reads them in the MMA FRAGMENT layout  (dictated by the MFMA hw:
             how the instruction wants A spread over the 64 lanes + each lane's
             VGPRs).
  * fx.copy  writes them in the TILED-COPY layout    (dictated by how you load
             from global memory: contiguous / vectorized per lane).
These two constraints are independent, so the layouts differ.  retile builds a
view of the fragment in the copy's coordinates so the handoff lines up.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/layout_diff_demo.py
"""
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# The layout dump below runs during kernel TRACING. If the kernel is cached,
# FlyDSL skips the trace and you see no prints -- so force the cache off here.
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
from flydsl.utils.env import RuntimeEnvManager

RuntimeEnvManager.enable_cache = False

M_MMA, N_MMA, K_MMA = 16, 16, 16
WAVE_M, WAVE_N = 2, 2
TILE_M, TILE_N = WAVE_M * M_MMA, WAVE_N * N_MMA
K = 64
NUM_THREADS = WAVE_M * WAVE_N * 64


def _rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    gA_k = fx.flat_divide(A, (TILE_M, K_MMA))[None, None, 0, None]
    gB_k = fx.flat_divide(B, (TILE_N, K_MMA))[None, None, 0, None]
    gC = fx.flat_divide(C, (TILE_M, TILE_N))[None, None, 0, 0]

    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(M_MMA, N_MMA, K_MMA, fx.BFloat16)),
        fx.make_layout((WAVE_M, WAVE_N, 1), (1, WAVE_M, 0)),
    )
    thr_mma = tiled_mma.thr_slice(tid)

    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((M_MMA * K_MMA // 64) * 16), fx.BFloat16)
    tiled_copy_A = fx.make_tiled_copy_A(cp_ab, tiled_mma)
    print("dir(tiled_copy_A):", dir(tiled_copy_A))

    tcA = tiled_copy_A.get_slice(tid)

    # ---- operand A, the FRAGMENT layout (what fx.gemm expects) ----
    frag_A = thr_mma.make_fragment_A(gA_k[None, None, 0])

    # ---- operand A, the COPY view of that same fragment (what fx.copy writes) ----
    copy_view_A = tcA.retile(frag_A)

    # ---- the block-wide thread-value (TV) layouts these are derived from ----
    _rule("BLOCK-WIDE thread-value (TV) layouts for operand A")
    print("MMA  TV layout (tiled_mma.tv_layout_A_tiled):")
    print("   ", tiled_mma.tv_layout_A_tiled)
    print("COPY TV layout (tiled_copy_A.layout_dst_tv_tiled) -- where each lane's")
    print("     loaded values must land:")
    print("   ", tiled_copy_A.layout_dst_tv_tiled)
    print("COPY TV layout (tiled_copy_A.layout_src_tv_tiled) -- how the global")
    print("     source is read per lane:")
    print("   ", tiled_copy_A.layout_src_tv_tiled)
    print("COPY Layout TV tiled   ", tiled_copy_A.layout_tv_tiled)
    print("tile mn size: ", tiled_copy_A.tile_mn)
    print("dir mn size: ", dir(tiled_copy_A))

    _rule("PER-THREAD register layouts for operand A (tid = a symbolic lane)")
    print("frag_A.layout            (MMA fragment  = fx.gemm's view):")
    print("   ", frag_A.layout)
    print("tcA.retile(frag_A).layout (copy view     = fx.copy's view):")
    print("   ", copy_view_A.layout)
    print("thr_gA (global SOURCE partition, tcA.partition_S(gA_k)).layout:")
    print("   ", tcA.partition_S(gA_k).layout)
    print()
    print("NOTE: for A/B the tiled-copy is DERIVED from the mma (make_tiled_copy_A")
    print("      uses tiled_mma.tv_layout_A_tiled), so the register maps coincide")
    print("      and retile here is just a mode RESHAPE, not a permutation.")

    # ---- operand C: the accumulator, also derived from the mma ----
    frag_C_dbg = thr_mma.make_fragment_C(gC)
    cp_c_dbg = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcC_dbg = fx.make_tiled_copy_C(cp_c_dbg, tiled_mma).get_slice(tid)
    _rule("PER-THREAD register layouts for operand C (accumulator)")
    print("frag_C.layout             (MMA accumulator = fx.gemm's view):")
    print("   ", frag_C_dbg.layout)
    print("tcC.retile(frag_C).layout (copy view        = fx.copy's view):")
    print("   ", tcC_dbg.retile(frag_C_dbg).layout)

    # ---- an INDEPENDENT copy (NOT derived from the mma) to show a real mismatch ----
    # A plain tiled copy: 64 lanes, each lane owns 4 contiguous bf16 along the
    # 16x16 A tile in a *row-major* value order -- chosen independently of the
    # MMA's distribution.  Now the register maps genuinely disagree.
    indep = fx.make_tiled_copy_tv(
        cp_ab,
        fx.make_layout((16, 4), (1, 16)),   # 64 threads over the 16x16 tile
        fx.make_layout((1, 4), (0, 1)),     # 4 values per thread, contiguous along K
    )
    _rule("INDEPENDENT (non-derived) copy: the block-wide 'ways' really differ")
    print("independent copy dst TV layout (a DIFFERENT thread->element map):")
    print("   ", indep.layout_dst_tv_tiled)
    print("MMA               TV layout    (the hardware's map):")
    print("   ", tiled_mma.tv_layout_A_tiled)
    print()
    print("These two block-wide TV layouts are genuinely different mappings of")
    print("(thread, value) -> tile-coordinate.  THAT is the 'two different ways'.")
    print("retile's job is to bridge whichever way the copy uses into the")
    print("fragment's storage.  (In THIS kernel each thread's A fragment is a")
    print("flat 4-element vector, so the retiled *register* view can't visibly")
    print("permute -- any 4 values land on the same 4 regs.  A true permutation")
    print("shows up once a thread owns a multi-DIM value block, e.g. bigger")
    print("tiles / vectorized LDS loads -- that's the scramble in retile_demo.py.)")

    # ---- do the actual work so we can prove correctness of the handoff ----
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0)
    thr_gA = tcA.partition_S(gA_k)

    tiled_copy_B = fx.make_tiled_copy_B(cp_ab, tiled_mma)
    tcB = tiled_copy_B.get_slice(tid)
    thr_gB = tcB.partition_S(gB_k)
    frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])

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
    _rule(f"correctness: max abs err = {err:.5f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
