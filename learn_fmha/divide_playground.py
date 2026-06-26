# SPDX-License-Identifier: Apache-2.0
"""Divide playground — see the four `*_divide` flavors print their layouts.

This is a companion to lesson 00. It does NO real compute. We build a couple of
register tensors with known layouts, run `logical_divide` / `zipped_divide` /
`tiled_divide` / `flat_divide` on them, and print the resulting `<shape:stride>`
so you can watch how each flavor groups the (Tile, Rest) modes.

Two things worth holding in your head:
  - A layout is written `<shape:stride>`. e.g. `<8:1>` = 8 elements, step 1.
  - `*_divide(A, tiler)` splits A into a Tile (inside one tile) and a Rest (which
    tile). The flavors differ ONLY in how those modes are grouped:

      Layout Shape : (M, N)        Tiler : <TileM, TileN>
      logical_divide : ((TileM,RestM), (TileN,RestN))
      zipped_divide  : ((TileM,TileN), (RestM,RestN))
      tiled_divide   : ((TileM,TileN), RestM, RestN)
      flat_divide    : (TileM, TileN, RestM, RestN)

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/divide_playground.py
"""

from typing import Any

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


def _render(x: Any, ir_values: list) -> str:
    """Render a (possibly nested) shape/stride into a printf format string.

    Static ints are inlined directly; dynamic ir.Values become `{}` placeholders
    and are appended to `ir_values` for fx.printf to fill in at runtime.
    """
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
    fx.printf(f"{label:<26}<{shape_fmt}:{stride_fmt}>\n", *ir_values)


@flyc.kernel(known_block_size=[1, 1, 1])
def divide_demo():
    # --- 1-D: the stride lever (blocked vs interleaved) ---------------------
    # An 8-element vector, contiguous. Split into tiles of 4.
    v = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
    print_layout(v, "vector")

    # tiler <4:1> -> contiguous tile  => tile0={0,1,2,3} tile1={4,5,6,7}
    print_layout(fx.logical_divide(v, fx.make_layout(4, 1)), "  logical <4:1> blocked")
    # tiler <4:2> -> strided tile      => tile0={0,2,4,6} tile1={1,3,5,7}
    print_layout(fx.logical_divide(v, fx.make_layout(4, 2)), "  logical <4:2> interleaved")

    # --- 2-D: the four divide flavors on the SAME split ---------------------
    # An 8x8 row-major matrix; tile it with a 2x4 sub-tile.
    a = fx.make_rmem_tensor(fx.make_layout((8, 8), (8, 1)), fx.Float32)
    print_layout(a, "matrix 8x8 row-major")

    print_layout(fx.logical_divide(a, (2, 4)), "  logical_divide <2,4>")
    print_layout(fx.zipped_divide(a, (2, 4)), "  zipped_divide  <2,4>")
    print_layout(fx.tiled_divide(a, (2, 4)), "  tiled_divide   <2,4>")
    print_layout(fx.flat_divide(a, (2, 4)), "  flat_divide    <2,4>")


@flyc.jit
def run_divide_demo(stream: fx.Stream = fx.Stream(None)):
    divide_demo().launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


if __name__ == "__main__":
    run_divide_demo(stream=torch.cuda.Stream())
    torch.cuda.synchronize()
