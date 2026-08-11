# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os

import torch

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils.env import DebugEnvManager, RuntimeEnvManager
from flydsl._mlir import ir

# ---- debug preamble (from the slide, cache forced off) ----
DebugEnvManager.enable_debug_info = True
DebugEnvManager.dump_asm = True
DebugEnvManager.dump_ir = True
DebugEnvManager.dump_dir = os.path.join(os.path.dirname(__file__), "vadd_dbg")
ir._globals.register_traceback_file_inclusion(__file__)
ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
ir._globals.set_loc_tracebacks_frame_limit(40)
ir._globals.set_loc_tracebacks_enabled(True)
RuntimeEnvManager.enable_cache = False
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"


@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    A = fx.rocdl.make_buffer_tensor(A)

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    copyAtomBuffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    fx.copy_atom_call(copyAtomBuffer, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


def run_eager():
    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    is_closed = torch.allclose(C, A + B)
    print(f"[Eager] Result correct: {is_closed}")
    return is_closed


if __name__ == "__main__":
    ok = run_eager()
    print(f"All passed: {ok}")
