# SPDX-License-Identifier: Apache-2.0
"""research_mfma — "what happens if I just change the MFMA instruction?"  (idiomatic)

A deliberately MINIMAL register-resident GEMM, written in the *layout-algebra*
style of `examples/03-tiledMma.py` / `examples/04-preshuffle_gemm.py` — i.e. the
idiomatic FlyDSL way, with no `.ir_value()`, no `buffer_ops`, and no hand-written
lane↔element index math. Data movement is done with **tiled copy atoms** and the
MMA **fragment** tensors that `make_tiled_mma` derives for you.

Design constraints (on purpose, so the ONLY thing that changes is the MFMA):
  - One wavefront (64 lanes, wave layout (1,1,1)) computes ONE MFMA-shaped tile.
  - The K dimension is a plain unrolled loop that accumulates into one C fragment.
  - NO LDS, NO prefetch, NO double-buffer, NO scheduling barriers. A/B fragments
    are copied straight from global into registers each K-step. This is
    intentionally memory-heavy — the clean control whose only knobs are:
        (1) the MFMA instruction   (16x16x16  vs  32x32x8)
        (2) the block→tile mapping (row-major vs XCD-swizzled)

WHY the MFMA is a raw opcode instead of `fx.gemm` (the DSL reality check):
  `fx.gemm` lowers through the CDNA3 MMA-atom dispatch
  (`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`), which on gfx942 bf16 only really
  reaches `16x16x16` — `32x32x8` has no dispatch entry and `32x32x4` isn't even
  LLVM-selectable on this arch. So the *instruction comparison itself* lives
  below `fx.gemm`. We keep everything idiomatic (tiled copies build the
  fragments; the fragment layouts are exactly what the hardware op wants) and
  only drop to the raw intrinsic for the single instruction, calling it on the
  fragment vectors (`frag.load()` / `frag.store()`). For 16x16x16 you *could*
  instead write `fx.gemm(tiled_mma, fC, fA, fB, fC)` — see the note below.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_mfma.py
      HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_mfma.py --M 4096 --N 4096 --K 4096
"""
import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import do_bench


# ───────────────────────── grid -> tile mapping ─────────────────────────────
# Pure block-index remap (idiomatic index math, no codegen tricks). Returns
# (pid_m, pid_n) for a given linear block id.
#   NUM_XCDS=1, GROUP_SIZE_M=1  -> plain row-major (the baseline).
#   NUM_XCDS>1                  -> group consecutive blocks onto the same XCD so
#                                  they hit the same private L2 (HipKittens fact #5).
#   GROUP_SIZE_M>1              -> windowed traversal: walk M-first within a group
#                                  of GROUP_SIZE_M tile-rows so a reused B column
#                                  stays hot in L2.
# Adapted from ROCm/gfx9-gluon-tutorials (also in FlyDSL/examples/efficient-gemm.py).
def get_pids(pid, num_pid_m, num_pid_n, NUM_XCDS, GROUP_SIZE_M):
    GRID_MN = num_pid_m * num_pid_n  # compile-time python int

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % fx.Int32(NUM_XCDS)
        local_pid = pid // fx.Int32(NUM_XCDS)
        pid = (xcd < fx.Int32(tall_xcds)).select(
            xcd * fx.Int32(pids_per_xcd) + local_pid,
            fx.Int32(tall_xcds * pids_per_xcd) + (xcd - fx.Int32(tall_xcds)) * fx.Int32(pids_per_xcd - 1) + local_pid,
        )

    if GROUP_SIZE_M == 1:
        pid_m = pid // fx.Int32(num_pid_n)
        pid_n = pid % fx.Int32(num_pid_n)
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // fx.Int32(num_pid_in_group)
        first_pid_m = group_id * fx.Int32(GROUP_SIZE_M)
        rem_m = fx.Int32(num_pid_m) - first_pid_m
        group_size_m = (rem_m < fx.Int32(GROUP_SIZE_M)).select(rem_m, fx.Int32(GROUP_SIZE_M))
        pid_m = first_pid_m + ((pid % fx.Int32(num_pid_in_group)) % group_size_m)
        pid_n = (pid % fx.Int32(num_pid_in_group)) // group_size_m
    return pid_m, pid_n


# ───────────────────────────── the kernel ───────────────────────────────────
# C[M,N] = A[M,K] @ B[N,K]^T   (B stored row-major [N,K]; the MFMA does A @ B^T).
# bf16 in, f32 accumulator. `shape` picks both the MMA atom and its raw opcode.
MFMA = {
    #            (m,  n,  k), raw opcode,                         acc f32/lane
    "16x16x16": ((16, 16, 16), fx.rocdl.mfma_f32_16x16x16bf16_1k, 4),
    "32x32x8":  ((32, 32, 8),  fx.rocdl.mfma_f32_32x32x8bf16_1k, 16),
}


def compile_variant(shape, M, N, K, num_xcds, group_m):
    (m, n, k), opcode, acc_n = MFMA[shape]
    assert M % m == 0 and N % n == 0 and K % k == 0
    num_pid_m, num_pid_n = M // m, N // n

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        pid_m, pid_n = get_pids(fx.Int32(fx.block_idx.x), num_pid_m, num_pid_n, num_xcds, group_m)

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        # this block's tiles; keep the K-tile dim on A/B for the loop.
        gA_k = fx.flat_divide(A, (m, k))[None, None, pid_m, None]   # (m, k, nK)
        gB_k = fx.flat_divide(B, (n, k))[None, None, pid_n, None]   # (n, k, nK)
        gC = fx.flat_divide(C, (m, n))[None, None, pid_m, pid_n]    # (m, n)

        # name the instruction once; the tiled_mma derives all fragment layouts.
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(m, n, k, fx.BFloat16)),
            fx.make_layout((1, 1, 1), (0, 0, 0)),  # one wave owns the tile
        )
        thr_mma = tiled_mma.thr_slice(tid)

        # global -> register copies (no LDS), widths derived from the fragment.
        cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((m * k // 64) * 16), fx.BFloat16)
        tcA = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
        tcB = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)

        thr_gA = tcA.partition_S(gA_k)
        thr_gB = tcB.partition_S(gB_k)
        frag_A = thr_mma.make_fragment_A(gA_k[None, None, 0])
        frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])
        frag_C = thr_mma.make_fragment_C(gC)
        A_ret = tcA.retile(frag_A)
        B_ret = tcB.retile(frag_B)
        frag_C.fill(0)

        # K-loop: copy this K-slice into registers, then issue the MFMA.
        f32xacc = fx.typing.T.vec(acc_n, fx.typing.T.f32)
        for kt in fx.range_constexpr(K // k):
            fx.copy(cp_ab, thr_gA[None, None, None, kt], A_ret)
            fx.copy(cp_ab, thr_gB[None, None, None, kt], B_ret)
            # The bf16 MFMA takes raw <Nxi16> operands; the fragment vectors are
            # already in the exact lane layout the instruction expects.
            a = frag_A.load().bitcast(fx.Int16)
            b = frag_B.load().bitcast(fx.Int16)
            frag_C.store(fx.Vector(opcode(f32xacc, [a, b, frag_C.load()])))
            # (For 16x16x16 you could replace the 3 lines above with:
            #   fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)
            #  but fx.gemm can't emit 32x32x8 on CDNA3, so we stay uniform.)

        # register -> global. 32b copy so the strided C fragment stores correctly.
        cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcC = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
        fx.copy(cp_c, tcC.retile(frag_C), tcC.partition_S(gC))

    grid_mn = num_pid_m * num_pid_n

    @flyc.jit
    def run(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kern(A, B, C).launch(grid=(grid_mn, 1, 1), block=(64, 1, 1), stream=stream)

    return run


# ─────────────────────────────── harness ────────────────────────────────────
VARIANTS = [
    ("16x16x16",      "16x16x16", 1, 1),
    ("32x32x8",       "32x32x8",  1, 1),
    ("16x16x16+xcd",  "16x16x16", None, None),  # num_xcds/group filled from args
    ("32x32x8+xcd",   "32x32x8",  None, None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--xcds", type=int, default=4, help="NUM_XCDS for the swizzled variants (MI308X≈4)")
    ap.add_argument("--group-m", type=int, default=8, help="GROUP_SIZE_M windowed traversal for swizzled variants")
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()
    M, N, K = args.M, args.N, args.K

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = A.float() @ B.float().T

    flop = 2.0 * M * N * K
    min_bytes = (M * K + N * K) * 2 + M * N * 4  # if A,B read once (lower bound)

    print(f"GEMM  C[{M},{N}] = A[{M},{K}] @ B[{N},{K}]^T   (bf16 in, f32 acc)")
    print(f"  {flop/1e9:.1f} GFLOP,  min I/O {min_bytes/1e6:.0f} MB  (no-LDS kernel rereads A/B many times)")
    print(f"  XCD-swizzle variants: NUM_XCDS={args.xcds}, GROUP_SIZE_M={args.group_m}\n")
    print(f"  {'variant':<16}{'tile':<10}{'us':>10}{'TFLOPS':>10}{'GB/s(min)':>12}   acc")
    print("  " + "-" * 70)

    results = {}
    for label, shape, xcds, group in VARIANTS:
        xcds = args.xcds if xcds is None else xcds
        group = args.group_m if group is None else group
        run = compile_variant(shape, M, N, K, xcds, group)
        C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        # IMPORTANT: launch on the CURRENT stream. do_bench records its timing
        # events on the current stream; a fresh torch.cuda.Stream() would let
        # reps overlap and report impossible (>peak) TFLOPS.
        stream = torch.cuda.current_stream()
        run(A, B, C, stream=stream)
        torch.cuda.synchronize()
        err = (C - ref).abs().max().item()
        ok = "PASS" if err < 5e-1 else f"FAIL({err:.2f})"

        fn = lambda r=run, C=C, s=stream: r(A, B, C, stream=s)
        ms = do_bench(fn, warmup=20, rep=args.rep)
        us = ms * 1e3
        tflops = flop / (ms * 1e-3) / 1e12
        gbs = min_bytes / (ms * 1e-3) / 1e9
        tile = {"16x16x16": "16x16", "32x32x8": "32x32"}[shape]
        print(f"  {label:<16}{tile:<10}{us:>10.1f}{tflops:>10.2f}{gbs:>12.1f}   {ok}")
        results[label] = (us, tflops)

    print()
    base = results["16x16x16"][1]
    for label, (_, tf) in results.items():
        print(f"  {label:<16} {tf:6.2f} TFLOPS   ({tf/base:.2f}x vs 16x16x16 baseline)")


if __name__ == "__main__":
    main()
