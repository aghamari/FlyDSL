# SPDX-License-Identifier: Apache-2.0
"""Tile playground — what a `make_tile` (a *tiler*) is and what you can do with it.

Companion to `divide_playground.py`. That file focused on the four *divide flavors*
(how the (Tile, Rest) modes get grouped); this one focuses on the TILER itself — the
`Tile` you build with `fx.make_tile(...)` and hand to `zipped_divide` /
`logical_divide` / `flat_divide` (and to `make_tiled_copy` as its `tile_mn`).

### What a Tile is
A `Tile` is an ordered list of per-axis *modes* — a cookie-cutter that says how to
carve each dimension of a tensor. Each mode is one of:

  * int N              -> contiguous run of N. Shorthand for the layout <N:1>.
  * Layout <N:s>       -> N elements spaced by s (strided / interleaved sub-tile).
  * Layout <(a,b):..>  -> a nested (hierarchical) sub-tile of that one axis.

And the *number* of modes matters: a tiler with FEWER modes than the tensor's rank
leaves the trailing axes untiled — that is how you "skip" an axis. (A bare Python
`None` mode and a bare nested int-tuple like `(2,2)` are NOT accepted by these
divides — use a short tiler to skip, and a nested Layout for hierarchy.)

### What you do with it
Pass it to a divide to split a tensor into (Tile, Rest):
  `fx.zipped_divide(A, tiler)` / `fx.logical_divide(...)` / `fx.flat_divide(...)`.
A plain Python tuple is auto-wrapped, so `fx.zipped_divide(A, (2, 4))` is exactly
`fx.zipped_divide(A, fx.make_tile(2, 4))`. Building the tile explicitly is what lets
you MIX int / Layout / None / nested modes per axis. The same object also names the
per-block footprint (`tile_mn`) inside `make_tiled_copy_{A,B,C}`.

This file does NO real compute — like `divide_playground.py`, it just prints each
resulting `<shape:stride>` so you can watch how each tiler mode carves the tensor.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/tile_playground.py
"""

from typing import Any

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


def _render(x: Any, ir_values: list) -> str:
    """Render a (possibly nested) shape/stride into a printf format string."""
    if isinstance(x, (tuple, list)):
        return "(" + ",".join(_render(v, ir_values) for v in x) + ")"
    if isinstance(x, int):
        return str(x)
    ir_values.append(x)
    return "{}"


def print_layout(t: fx.Tensor, label: str) -> None:
    """Print one tensor's layout as `<label> <shape:stride>` from inside a kernel."""
    ir_values: list = []
    shape_fmt = _render(t.shape.to_py_value(), ir_values)
    stride_fmt = _render(t.stride.to_py_value(), ir_values)
    fx.printf(f"{label:<34}<{shape_fmt}:{stride_fmt}>\n", *ir_values)


@flyc.kernel(known_block_size=[1, 1, 1])
def tile_demo():
    # ── 1-D: the three scalar-axis modes (int, blocked Layout, strided Layout) ──
    # An 8-element contiguous vector; carve it with a size-4 tile three ways.
    v = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
    print_layout(v, "vector <8:1>")

    # int 4  ==  <4:1> : contiguous  => tile {0,1,2,3}
    print_layout(fx.logical_divide(v, fx.make_tile(4)), "  make_tile(4)      int == <4:1>")
    # <4:1> spelled out: identical to the int form above
    print_layout(fx.logical_divide(v, fx.make_tile(fx.make_layout(4, 1))), "  make_tile(<4:1>)  blocked")
    # <4:2> : stride-2  => tile {0,2,4,6}  (interleaved, not contiguous)
    print_layout(fx.logical_divide(v, fx.make_tile(fx.make_layout(4, 2))), "  make_tile(<4:2>)  interleaved")

    # ── 2-D: one mode PER AXIS; mix ints, Layouts, short tilers, nested Layouts ─
    # An 8x8 row-major matrix (<(8,8):(8,1)>). Tiler modes line up with axes (row, col).
    a = fx.make_rmem_tensor(fx.make_layout((8, 8), (8, 1)), fx.Float32)
    print_layout(a, "matrix 8x8 row-major")

    # int per axis: 2 rows x 4 cols, both contiguous
    print_layout(fx.zipped_divide(a, fx.make_tile(2, 4)), "  make_tile(2, 4)       both axes")
    print_layout(
        fx.zipped_divide(a, fx.make_tile(fx.make_layout(2, 1), fx.make_layout(4, 2))),
        "  make_tile(<2:1>,<4:2>) mixed",
    )
    print_layout(fx.zipped_divide(a, fx.make_tile(2)), "  make_tile(2)          rows only (short tiler)")
    print_layout(fx.zipped_divide(a, fx.make_tile(fx.make_layout((2, 2), (1, 2)), 4)), "  make_tile(<(2,2):(1,2)>,4) nested")

    # ── The auto-wrap: a plain tuple IS make_tile(...) ──────────────────────────
    print_layout(fx.zipped_divide(a, (2, 4)), "  (2, 4)  tuple  == make_tile(2,4)")


@flyc.jit
def run_tile_demo(stream: fx.Stream = fx.Stream(None)):
    tile_demo().launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


if __name__ == "__main__":
    run_tile_demo(stream=torch.cuda.Stream())
    torch.cuda.synchronize()
