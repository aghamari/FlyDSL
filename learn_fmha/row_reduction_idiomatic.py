# SPDX-License-Identifier: Apache-2.0
"""Row Reduction (idiomatic FlyDSL) — companion to `row_reduction.py`.

Same four approaches as the CK-Tile `tutorial_16_row_reduction` port, but written
in the idiomatic `fx` style the layout-algebra skill + learn_fmha lessons prescribe
for REDUCTIONS specifically:

  * Loads are VECTORIZED `buffer_load` wrapped in `fx.Vector` (the skill's §7 rule:
    "reduction = buffer_load + warp shuffle + LDS" — tiled-copy/`zipped_divide` is for
    GEMM/transpose, not reductions).
  * The in-thread stage uses `fx.Vector.reduce("add"|"max")` instead of a scalar loop.
  * The cross-lane stage stays DIRECT (`shuffle_xor` butterfly) — lessons 04/06: the
    reduction stays direct register + shuffle code.
  * One reusable `block_reduce(acc, op)` bundles warp-shuffle + cross-warp LDS, exactly
    mirroring CK's reduce hierarchy and `kernels/reduce.py`'s `make_block_reduce`.
  * A small `ReduceOp` carries `(identity, combine, vname)` so swapping max<->sum is one line.

Mapping (one row per block; warp-only = a single-warp block, no LDS):
  1. k_warp_max    — warp-only row-max   (BLOCK=64, 1 warp): in-thread + warp shuffle
  2. k_block_max   — cross-warp row-max  (BLOCK=256): + LDS cross-warp, stages spelled out
  3. k_wrapped_max — cross-warp row-max via the reusable `block_reduce` wrapper
  4. k_block_sum   — cross-warp row-SUM via the same wrapper (op = ADD)

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/row_reduction_idiomatic.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# ── Problem + tile config ────────────────────────────────────────────────────
M = 1024
N = 2048
WARP = 64
VEC = 4                         # 128-bit vectorized f32 loads
NEG_INF = -3.0e38


# ── Reduce op: identity + scalar combiner + the fx.Vector.reduce name ────────
class ReduceOp:
    def __init__(self, name, identity, combine, vname):
        self.name = name
        self.identity = identity
        self.combine = combine          # (Float32, Float32) -> Float32
        self.vname = vname              # fx.Vector.reduce kind


MAX = ReduceOp("max", NEG_INF, lambda a, b: a.maximumf(b), "max")
ADD = ReduceOp("add", 0.0, lambda a, b: a + b, "add")


# ── Stage 1: in-thread — vectorized loads + fx.Vector.reduce ────────────────
def in_thread_reduce(rX, row_base, tid, block, op):
    chunk = block * VEC
    n_chunks = N // chunk
    acc = fx.Float32(op.identity)
    for ct in fx.range_constexpr(n_chunks):
        col = fx.Int32(ct * chunk) + tid * fx.Int32(VEC)
        v = fx.Vector(fx.buffer_ops.buffer_load(rX, row_base + col, vec_width=VEC, dtype=fx.Float32))
        acc = op.combine(acc, fx.Float32(v.reduce(op.vname)))
    return acc


# ── Stage 2: warp shuffle — cross-lane XOR butterfly (stays direct) ─────────
def wave_reduce(acc, op):
    for sh in (32, 16, 8, 4, 2, 1):
        acc = op.combine(acc, acc.shuffle_xor(fx.Int32(sh), fx.Int32(WARP)))
    return acc


# ── Stage 3: cross-warp — per-warp partial -> LDS -> barrier -> combine ─────
# After wave_reduce every lane holds its warp's partial, so all lanes writing the
# same LDS slot is idempotent (avoids a stateful runtime-if the frontend can't thread).
def cross_warp_reduce(acc, tid, s_red, nwarps, op):
    wave = tid // fx.Int32(WARP)
    s_red.store(acc.ir_value(), [wave])
    fx.gpu.barrier()
    total = fx.Float32(op.identity)
    for w in fx.range_constexpr(nwarps):
        total = op.combine(total, fx.Float32(s_red.load([w])))
    return total


# ── The reusable all-stages wrapper (FlyDSL analog of CK's ReduceKernel) ─────
def block_reduce(acc, tid, s_red, nwarps, op):
    return cross_warp_reduce(wave_reduce(acc, op), tid, s_red, nwarps, op)


def _make_alloc(name, nwarps):
    a = SmemAllocator(None, arch="gfx942", global_sym_name=name)
    a.ptr = nwarps * 4
    return a


_alloc_bmax = _make_alloc("row_red_idiom_block_max_smem", 4)
_alloc_wrap = _make_alloc("row_red_idiom_wrapped_max_smem", 4)
_alloc_bsum = _make_alloc("row_red_idiom_block_sum_smem", 4)


def _lds(alloc, nwarps):
    return SmemPtr(alloc.get_base(), 0, fx.typing.T.f32, shape=(nwarps,))


# ── Approach 1 — Warp-only row-max (single-warp block, no LDS) ───────────────
@flyc.kernel(known_block_size=[WARP, 1, 1])
def k_warp_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)

    acc = in_thread_reduce(rX, bid * fx.Int32(N), tid, WARP, MAX)
    acc = wave_reduce(acc, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── Approach 2 — Cross-warp row-max (stages spelled out) ─────────────────────
@flyc.kernel(known_block_size=[4 * WARP, 1, 1])
def k_block_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    s_red = _lds(_alloc_bmax, 4)

    acc = in_thread_reduce(rX, bid * fx.Int32(N), tid, 4 * WARP, MAX)
    acc = wave_reduce(acc, MAX)
    acc = cross_warp_reduce(acc, tid, s_red, 4, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── Approach 3 — Same row-max via the reusable wrapper ───────────────────────
@flyc.kernel(known_block_size=[4 * WARP, 1, 1])
def k_wrapped_max(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    s_red = _lds(_alloc_wrap, 4)

    acc = in_thread_reduce(rX, bid * fx.Int32(N), tid, 4 * WARP, MAX)
    acc = block_reduce(acc, tid, s_red, 4, MAX)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── Approach 4 — Cross-warp row-SUM via the same wrapper (op = ADD) ──────────
@flyc.kernel(known_block_size=[4 * WARP, 1, 1])
def k_block_sum(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    s_red = _lds(_alloc_bsum, 4)

    acc = in_thread_reduce(rX, bid * fx.Int32(N), tid, 4 * WARP, ADD)
    acc = block_reduce(acc, tid, s_red, 4, ADD)
    fx.buffer_ops.buffer_store(acc.ir_value(), rY, bid.ir_value())


# ── JIT launchers ────────────────────────────────────────────────────────────
@flyc.jit
def run_warp_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    k_warp_max(X, Y).launch(grid=(M, 1, 1), block=(WARP, 1, 1), stream=stream)


def _finalize(alloc):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalize()


@flyc.jit
def run_block_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    _finalize(_alloc_bmax)
    k_block_max(X, Y).launch(grid=(M, 1, 1), block=(4 * WARP, 1, 1), stream=stream)


@flyc.jit
def run_wrapped_max(X, Y, stream: fx.Stream = fx.Stream(None)):
    _finalize(_alloc_wrap)
    k_wrapped_max(X, Y).launch(grid=(M, 1, 1), block=(4 * WARP, 1, 1), stream=stream)


@flyc.jit
def run_block_sum(X, Y, stream: fx.Stream = fx.Stream(None)):
    _finalize(_alloc_bsum)
    k_block_sum(X, Y).launch(grid=(M, 1, 1), block=(4 * WARP, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(M, N, dtype=torch.float32, device="cuda")
    ref_max = X.amax(dim=1)
    ref_sum = X.sum(dim=1)

    print(f"\nRow reduction (idiomatic)  X[{M}, {N}] -> Y[{M}]\n")

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
