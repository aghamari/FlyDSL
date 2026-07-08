# SPDX-License-Identifier: Apache-2.0
"""Tour 01 — make_mma_atom: name ONE MFMA instruction + its per-wave operand layouts.

An MmaAtom is "which instruction". It carries the fixed hardware thread-value layouts for
each operand (layout_A/B/C_tv) — the lane<->element map print_typst draws. Nothing is tiled
across waves yet.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/tour/01_mma_atom.py
"""
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def demo():
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    print("MmaAtom = MFMA(16,16,16, bf16)")
    print("  shape_mnk   :", atom.shape_mnk)     # (M,N,K) of one instruction
    print("  thr_layout  :", atom.thr_layout)    # 64 lanes in a wave
    print("  layout_A_tv :", atom.layout_A_tv)   # (thread,value) -> A[m,k]
    print("  layout_B_tv :", atom.layout_B_tv)   # (thread,value) -> B[n,k]
    print("  layout_C_tv :", atom.layout_C_tv)   # (thread,value) -> C[m,n]


demo()
