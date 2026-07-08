# SPDX-License-Identifier: Apache-2.0
"""Tour 03 — make_copy_atom + make_tiled_copy_A: the copy side.

- make_copy_atom(BufferCopy(bits), dtype) names ONE hardware copy instruction (how many
  bits a lane moves). It's the data-movement analog of the MMA atom.
- make_tiled_copy_A(cp, tiled_mma) builds a TiledCopy whose thread-value layout is taken
  FROM the MMA's A operand (tv_layout_A_tiled), so loads land where the MFMA expects.

Prints the copy atom, then the A-matched TiledCopy's tile + TV/src/dst layouts. No tid here
(that comes from .get_slice(tid), see tour 04).

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/tour/03_copy_atom.py
"""
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def demo():
    cp = fx.make_copy_atom(fx.rocdl.BufferCopy(64), fx.BFloat16)  # 64 bits = 4 bf16 / lane
    print("CopyAtom = BufferCopy(64), bf16 :", cp)

    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tm = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    tcA = fx.make_tiled_copy_A(cp, tm)   # a TiledCopy (team-wide), matched to MMA operand A
    print("TiledCopy (A-matched)")
    print("  tile_mn            :", tcA.tile_mn)
    print("  layout_tv_tiled    :", tcA.layout_tv_tiled)
    print("  layout_src_tv_tiled:", tcA.layout_src_tv_tiled)
    print("  layout_dst_tv_tiled:", tcA.layout_dst_tv_tiled)
    print(dir(tcA))


demo()
