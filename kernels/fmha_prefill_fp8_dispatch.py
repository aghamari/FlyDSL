# SPDX-License-Identifier: Apache-2.0
"""Per-shape dispatch wrapper for the FMHA prefill kernel (CK-style trait heuristic).

The kernel `fmha_prefill_fp8_ck_hk5` bakes KT and DIAG as compile-time constexpr read
from env at import. The full-grid autotune sweep (tests/kernels/sweep_fmha.py, device-fair
on MI308X) found the per-seqlen optimum is purely (KT, DIAG) — KPAD/VPAD/NWAVES/NBUF stay
at their defaults (8/8/4/2). This wrapper picks (KT, DIAG) by seqlen and returns the matching
compiled kernel module.

Measured best (device-fair graph-replay TF, 2026-06-18; per-seqlen BASE + KT/DIAG):
    sq<=1024  : log2dom KT=64 DIAG=0 -> 26 TF  (CK 30)
    sq<=2048  : log2dom KT=64 DIAG=1 -> 55 TF  (CK 62)
    sq<=16384 : hk5     KT=32 DIAG=0 -> 129 TF (CK 141)  # log2dom regresses to 106 here
    else      : hk5     KT=32 DIAG=1 -> 142 TF (CK 146)
The log2dom stack (LOG2E/exp-bias/kdlds) wins at small seq but regresses at large seq,
so get_kernel() now also picks the base kernel per seqlen (see best_base()).

CONSTRAINT: KT/DIAG are read at import and FlyDSL's SmemAllocator finalizes once per process,
so one process holds ONE (KT,DIAG) instance. `get_kernel(sq)` sets env then (re)imports the
kernel module fresh. Call it ONCE per process for a given seqlen class (mirrors how the tests
fork a subprocess per shape).
"""
from __future__ import annotations

import importlib
import os
import sys

_KERNEL_MODULE = "fmha_prefill_fp8_ck_hk5"  # legacy default; see best_base() below

# Per-seqlen BASE kernel. The log2dom stack (LOG2E-into-descale + exp-bias hoist + kdlds)
# WINS at small seq but REGRESSES at large seq. Measured device-fair (graph-replay) 2026-06-18
# at each seqlen's optimal (KT,DIAG):
#     sq1024  KT64 DIAG0 : log2dom 26  vs hk5 25
#     sq2048  KT64 DIAG1 : log2dom 55  vs hk5 52
#     sq16384 KT32 DIAG0 : log2dom 106 vs hk5 129   <- hk5 wins big
#     sq32768 KT32 DIAG1 : log2dom 131 vs hk5 142   <- hk5 wins
# So dispatch the BASE per seqlen too (not just KT/DIAG): log2dom <=2048, hk5 above.
# Net best-of-both: 26/55/129/142 TF (CK-Tile fp8 ref: 30/62/141/146).
_BASE_SMALL = "fmha_prefill_fp8_ck_log2dom"
_BASE_LARGE = "fmha_prefill_fp8_ck_hk5"


def best_base(sq: int) -> str:
    """Return the fastest base kernel module for this seqlen (measured, see note above)."""
    return _BASE_SMALL if sq <= 2048 else _BASE_LARGE


def best_kt_diag(sq: int) -> tuple[int, int]:
    """Return (KT, DIAG) tuned per seqlen from the autotune sweep."""
    if sq <= 1024:
        return 64, 0
    if sq <= 2048:
        return 64, 1
    if sq <= 16384:
        return 32, 0
    return 32, 1


def get_kernel(sq: int):
    """Set the (KT,DIAG) env for this seqlen, pick the per-seqlen base kernel, (re)import.

    The returned module exposes `run_attn(...)` and `BM` exactly like the base kernel,
    so existing call sites work unchanged:
        K = get_kernel(sq)
        grid = b * nq * ((sq + K.BM - 1) // K.BM)
        K.run_attn(*args, ..., grid)
    """
    kt, diag = best_kt_diag(sq)
    mod = best_base(sq)
    os.environ["FMHA_KT"] = str(kt)
    os.environ["FMHA_DIAG"] = str(diag)
    # Force a fresh import so the new env constexpr take effect (KT/DIAG are read at import).
    sys.modules.pop(mod, None)
    return importlib.import_module(mod)
