# SPDX-License-Identifier: Apache-2.0
"""Atomic-add demo — why `UniversalAtomic(Add)` exists.

We sum an array by having EVERY thread add its element into a single global
accumulator `Out[0]`. Two kernels, same idea, one difference:

  - atomic_sum_kernel : each lane does an ATOMIC add  -> correct sum.
  - naive_sum_kernel  : each lane does a PLAIN store   -> lost-update race;
                        only the last writer survives, so you get garbage.

Each thread prints what it contributes so you can watch the kernel run, then the
host prints the final accumulator vs. the expected sum.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/atomic_add_demo.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

N = 8  # tiny so the per-thread prints are readable; one block of N threads


@flyc.kernel(known_block_size=[N, 1, 1])
def atomic_sum_kernel(A: fx.Tensor, Out: fx.Tensor):
    tid = fx.thread_idx.x

    # one element per thread: split A into N tiles of 1, this lane owns tile `tid`.
    tA = fx.logical_divide(A, fx.make_layout(1, 1))

    load_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    add_atom = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx.Float32), fx.Float32)

    rA = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(load_atom, fx.slice(tA, (None, tid)), rA)

    val = fx.memref_load_vec(rA)
    fx.printf("[atomic] thread {} adds {} into Out[0]\n", tid, val)

    # Out[0] += val, atomically across all racing lanes.
    tOut = fx.logical_divide(Out, fx.make_layout(1, 1))
    fx.copy_atom_call(add_atom, rA, fx.slice(tOut, (None, fx.Int32(0))))


@flyc.kernel(known_block_size=[N, 1, 1])
def naive_sum_kernel(A: fx.Tensor, Out: fx.Tensor):
    tid = fx.thread_idx.x

    tA = fx.logical_divide(A, fx.make_layout(1, 1))

    load_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    store_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    rA = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(load_atom, fx.slice(tA, (None, tid)), rA)

    val = fx.memref_load_vec(rA)
    fx.printf("[naive ] thread {} stores {} into Out[0] (overwrites!)\n", tid, val)

    # Plain store: every lane writes Out[0] = val, so they clobber each other.
    tOut = fx.logical_divide(Out, fx.make_layout(1, 1))
    fx.copy_atom_call(store_atom, rA, fx.slice(tOut, (None, fx.Int32(0))))


@flyc.jit
def run_atomic_sum(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    atomic_sum_kernel(A, Out).launch(grid=(1, 1, 1), block=(N, 1, 1), stream=stream)


@flyc.jit
def run_naive_sum(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    naive_sum_kernel(A, Out).launch(grid=(1, 1, 1), block=(N, 1, 1), stream=stream)


def main():
    A = torch.arange(N, dtype=torch.float32).cuda()  # [0, 1, 2, ..., 7]
    expected = A.sum().item()  # 0+1+...+7 = 28

    print(f"input A = {A.tolist()}")
    print(f"expected sum = {expected}\n")

    print("=== atomic add (correct) ===")
    out = torch.zeros(1, dtype=torch.float32).cuda()  # accumulator MUST start at 0
    run_atomic_sum(A, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    print(f"-> Out[0] = {out.item()}  (expected {expected})  "
          f"{'PASS' if abs(out.item() - expected) < 1e-3 else 'FAIL'}\n")

    print("=== naive plain store (racy / wrong) ===")
    out.zero_()
    run_naive_sum(A, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    print(f"-> Out[0] = {out.item()}  (NOT {expected}: only one lane's write survived)")


if __name__ == "__main__":
    main()
