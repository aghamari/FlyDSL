# SPDX-License-Identifier: Apache-2.0
"""Row Reduction — Warp Reduce vs Block Reduce, the FlyDSL port of CK-Tile's
`tutorial_16_row_reduction/row_reduction.cpp`.

Computes  Y[M] = reduce(X[M, N], axis=1)  (one reduction per row), across the
SAME reduce-abstraction levels the CK tutorial demonstrates, but written with
FlyDSL's direct register + shuffle + LDS primitives (FlyDSL has no ReduceKernel
/ BlockReduce2d* atoms — reductions stay explicit, see lesson_04_softmax).

The reduce hierarchy (identical to CK / unified-attention / production reduce):
  Stage 1  In-thread    — each thread reduces its own strided slice of the row
  Stage 2  Warp shuffle — cross-lane XOR butterfly (shuffle_xor) over 64 lanes
  Stage 3  Cross-warp   — each warp's partial -> LDS -> barrier -> combine

Four approaches (mirroring the four in the .cpp):
  1. k_warp_max     — Warp-only row-max.   Stage 1 + Stage 2. No LDS.
                      Shape: NWARPS warps along M, 1 warp owns a whole row.
  2. k_block_max    — Cross-warp row-max.  Stage 1 + 2 + 3 (LDS). Stages spelled out.
                      Shape: all NWARPS warps cooperate on ONE row.
  3. k_wrapped_max  — Same row-max, but via the reusable `block_row_reduce`
                      wrapper (the FlyDSL analog of CK's host-only ReduceKernel
                      that bundles all three stages).
  4. k_block_sum    — Cross-warp row-SUM via the same wrapper, op = ADD.
                      Shows switching the reduce op is trivial.

CK uses runtime M,N + a tiled-window distribution; here M,N are compile-time and
the per-thread column mapping is explicit (strided), which keeps the focus on the
3-stage reduce hierarchy rather than the tile-distribution algebra.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/row_reduction.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# ── Problem + tile config (compile-time) ────────────────────────────────────
M = 1024
N = 2048
WARP = 64
NWARPS = 4
BLOCK = NWARPS * WARP          # 256 threads / block
NEG_INF = -3.0e38


# ── Reduce op: identity + a trace-time combiner (CK's ReduceOp::Max / ::Add) ─
class ReduceOp:
    def __init__(self, name, identity, combine):
        self.name = name
        self.identity = identity
        self.combine = combine


MAX = ReduceOp("max", NEG_INF, lambda a, b: a.maximumf(b))
ADD = ReduceOp("add", 0.0, lambda a, b: a + b)


# ── Stage 1: in-thread reduction over a strided slice of the row ────────────
def in_thread_reduce(rX, row_base, start, stride, n_iters, op):
    acc = fx.Float32(op.identity)
    for j in fx.range_constexpr(n_iters):
        col = start + fx.Int32(j * stride)
        v = fx.Float32(fx.buffer_ops.buffer_load(rX, row_base + col, vec_width=1, dtype=fx.Float32))
        acc = op.combine(acc, v)
    return acc


# ── Stage 2: cross-lane XOR butterfly over the 64 lanes of a warp ───────────
def warp_reduce(acc, op):
    for mask in (1, 2, 4, 8, 16, 32):
        acc = op.combine(acc, acc.shuffle_xor(fx.Int32(mask), fx.Int32(WARP)))
    return acc


# ── Stage 3: cross-warp reduction via LDS (one slot per warp) ───────────────
def cross_warp_reduce(acc, warp_id, lds_view, op):
    # After warp_reduce every lane in a warp holds that warp's partial, so all
    # lanes write the same value to LDS[warp_id] (idempotent).
    lds_view.store(acc.ir_value(), [warp_id])
    fx.gpu.barrier()
    total = fx.Float32(op.identity)
    for w in fx.range_constexpr(NWARPS):
        total = op.combine(total, fx.Float32(lds_view.load([w])))
    return total


# ── The reusable all-stages wrapper (FlyDSL analog of CK's ReduceKernel) ─────
def block_row_reduce(rX, row_base, tid, warp_id, lds_view, op):
    acc = in_thread_reduce(rX, row_base, tid, BLOCK, N // BLOCK, op)
    acc = warp_reduce(acc, op)
    acc = cross_warp_reduce(acc, warp_id, lds_view, op)
    return acc


# LDS allocators (one global per cross-warp kernel; NWARPS f32 slots each).
def _make_alloc(name):
    a = SmemAllocator(None, arch="gfx942", global_sym_name=name)
    a.ptr = NWARPS * 4
    return a


_alloc_bmax = _make_alloc("row_reduction_block_max_smem")
_alloc_wrap = _make_alloc("row_reduction_wrapped_max_smem")
_alloc_bsum = _make_alloc("row_reduction_block_sum_smem")


def _lds(alloc):
    return SmemPtr(alloc.get_base(), 0, fx.typing.T.f32, shape=(NWARPS,))


# ── Approach 1 — Warp-only row-max (Stage 1 + 2, no LDS) ─────────────────────
@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def k_warp_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    warp_id = tid // fx.Int32(WARP)
    lane = tid % fx.Int32(WARP)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)

    row = bid * fx.Int32(NWARPS) + warp_id          # one row per warp
    row_base = row * fx.Int32(N)

    acc = in_thread_reduce(rX, row_base, lane, WARP, N // WARP, MAX)
    acc = warp_reduce(acc, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, row.ir_value())


# ── Approach 2 — Cross-warp row-max (Stage 1 + 2 + 3, spelled out) ───────────
@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def k_block_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    warp_id = tid // fx.Int32(WARP)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    lds_view = _lds(_alloc_bmax)

    row = bid                                        # one row per block
    row_base = row * fx.Int32(N)

    acc = in_thread_reduce(rX, row_base, tid, BLOCK, N // BLOCK, MAX)
    acc = warp_reduce(acc, MAX)
    acc = cross_warp_reduce(acc, warp_id, lds_view, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, row.ir_value())


# ── Approach 3 — Same row-max via the reusable wrapper ───────────────────────
@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def k_wrapped_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    warp_id = tid // fx.Int32(WARP)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    lds_view = _lds(_alloc_wrap)

    row_base = bid * fx.Int32(N)
    acc = block_row_reduce(rX, row_base, tid, warp_id, lds_view, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── Approach 4 — Cross-warp row-SUM via the same wrapper (op = ADD) ──────────
@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def k_block_sum(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    warp_id = tid // fx.Int32(WARP)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    lds_view = _lds(_alloc_bsum)

    row_base = bid * fx.Int32(N)
    acc = block_row_reduce(rX, row_base, tid, warp_id, lds_view, ADD)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── JIT launchers ────────────────────────────────────────────────────────────
@flyc.jit
def run_warp_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    k_warp_max(X, Y).launch(grid=(M // NWARPS, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@flyc.jit
def run_block_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc_bmax.finalize()
    k_block_max(X, Y).launch(grid=(M, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@flyc.jit
def run_wrapped_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc_wrap.finalize()
    k_wrapped_max(X, Y).launch(grid=(M, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@flyc.jit
def run_block_sum(X, Y, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc_bsum.finalize()
    k_block_sum(X, Y).launch(grid=(M, 1, 1), block=(BLOCK, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(M, N, dtype=torch.float32, device="cuda")
    ref_max = X.amax(dim=1)
    ref_sum = X.sum(dim=1)

    print(f"\nRow reduction  X[{M}, {N}] -> Y[{M}]   (BLOCK={BLOCK}, NWARPS={NWARPS})\n")

    def check(name, runner, ref, tol):
        Y = torch.zeros(M, dtype=torch.float32, device="cuda")
        runner(X, Y, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        err = (Y - ref).abs().max().item()
        print(f"  {name:<34} max abs err = {err:.6f}  ->  {'PASS' if err < tol else 'FAIL'}")

    check("1. warp-only   row-max", run_warp_max, ref_max, 1e-5)
    check("2. cross-warp  row-max", run_block_max, ref_max, 1e-5)
    check("3. wrapped     row-max", run_wrapped_max, ref_max, 1e-5)
    check("4. cross-warp  row-sum", run_block_sum, ref_sum, 5e-2)
    print()
