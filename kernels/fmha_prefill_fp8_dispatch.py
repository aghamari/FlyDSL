# SPDX-License-Identifier: Apache-2.0
"""Per-shape dispatch wrapper for the FMHA prefill kernel (CK-style trait heuristic).

The kernel `fmha_prefill_fp8_ck_hk5` bakes KT and DIAG as compile-time constexpr read
from env at import. The full-grid autotune sweep (tests/kernels/sweep_fmha.py, device-fair
on MI308X) found the per-seqlen optimum is purely (KT, DIAG) — KPAD/VPAD/NWAVES/NBUF stay
at their defaults (8/8/4/2). This wrapper picks (KT, DIAG) by seqlen and returns the matching
compiled kernel module.

Measured best (device-fair TF vs colleague fixed-config / CK-Tile):
    sq<=1024  : KT=64 DIAG=0  -> 25 TF  (was 19, CK 30)
    sq<=2048  : KT=64 DIAG=1  -> 52 TF  (was 48, CK 62)
    sq<=16384 : KT=32 DIAG=0  -> 128 TF (was 123, CK 141)
    else      : KT=32 DIAG=1  -> 142 TF (default, CK 146)

CONSTRAINT: KT/DIAG are read at import and FlyDSL's SmemAllocator finalizes once per process,
so one process holds ONE (KT,DIAG) instance. `get_kernel(sq)` sets env then (re)imports the
kernel module fresh. Call it ONCE per process for a given seqlen class (mirrors how the tests
fork a subprocess per shape).
"""
from __future__ import annotations

import importlib
import os
import sys

_KERNEL_MODULE = "fmha_prefill_fp8_ck_hk5"


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
    """Set the (KT,DIAG) env for this seqlen, (re)import, and return the kernel module.

    The returned module exposes `run_attn(...)` and `BM` exactly like the base kernel,
    so existing call sites work unchanged:
        K = get_kernel(sq)
        grid = b * nq * ((sq + K.BM - 1) // K.BM)
        K.run_attn(*args, ..., grid)
    """
    kt, diag = best_kt_diag(sq)
    os.environ["FMHA_KT"] = str(kt)
    os.environ["FMHA_DIAG"] = str(diag)
    # Force a fresh import so the new env constexpr take effect (KT/DIAG are read at import).
    sys.modules.pop(_KERNEL_MODULE, None)
    return importlib.import_module(_KERNEL_MODULE)
