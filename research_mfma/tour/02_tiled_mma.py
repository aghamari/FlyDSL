# SPDX-License-Identifier: Apache-2.0
"""Tour 02 — make_tiled_mma: replicate the atom across the block's waves.

make_tiled_mma(atom, atom_layout) takes the single MFMA and an (Mrep,Nrep,Krep) wave grid,
and derives:
  - tile_size_mnk     : the M x N x K this tiled MMA covers  (atom x atom_layout)
  - thr_layout_vmnk   : (lane, m-wave, n-wave, k-wave) -> thread id
  - tv_layout_?_tiled : the TILED (thread,value)->pos layouts (the CK *_dstr_encode analog)

Compare (1,1,1) (one wave) with (2,2,1) (four waves) to see the layouts grow.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/tour/02_tiled_mma.py
"""
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def demo():
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    for name, al in [("(1,1,1)", fx.make_layout((1, 1, 1), (0, 0, 0))),
                     ("(2,2,1)", fx.make_layout((2, 2, 1), (1, 2, 0)))]:
        tm = fx.make_tiled_mma(atom, al)
        print(f"TiledMma  atom_layout = {name}")
        print("  tile_size_mnk     :", tm.tile_size_mnk)
        print("  thr_layout_vmnk   :", tm.thr_layout_vmnk)
        print("  tv_layout_A_tiled :", tm.tv_layout_A_tiled)
        print("  tv_layout_B_tiled :", tm.tv_layout_B_tiled)
        print("  tv_layout_C_tiled :", tm.tv_layout_C_tiled)
        print()


demo()
