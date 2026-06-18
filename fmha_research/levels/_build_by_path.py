# SPDX-License-Identifier: Apache-2.0
"""Kernel loader for the FMHA reproduction levels.

Structural analog of ``fused_mega_moe/levels/_build_by_path.py``. In the MoE example the
loader exists to import a *curated snapshot* file without it shadowing the production kernel
(it loads the file by path under a unique module name and rewrites its smem global symbol).

For THIS kernel almost every level is reachable from an EXISTING ``kernels/`` module plus
``FMHA_*`` env overrides (the levers are flag-tunable — see reproduce_levels.LEVELS), so the
common path is simply ``importlib.import_module(name)`` after the env is set. We only need
the path-based, symbol-renamed load for the rare snapshot that is NOT reachable by an
existing module + flag (none are required today; the helper is here for parity + future use).

WHY env-before-import: every FMHA_* knob (KT/DIAG/VCOL/KPAD/VPAD/NWAVES/NBUF/XCD) is read at
IMPORT time, and FlyDSL's module-global SmemAllocator finalizes ONCE per process. The driver
therefore forks a subprocess per (level, seqlen) and sets env before calling load_kernel.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_KERNELS = _REPO / "kernels"


def load_kernel(name: str) -> ModuleType:
    """Import an EXISTING kernel module by name from ``kernels/`` (the common case).

    Env (FMHA_*) MUST already be set in this process before calling — it is read at import.
    Returns the module exposing ``run_attn``, ``BM`` and (optionally) ``V_COL``.
    """
    if str(_KERNELS) not in sys.path:
        sys.path.insert(0, str(_KERNELS))
    return importlib.import_module(name)


def load_snapshot(path: str | Path, smem_sym: str) -> ModuleType:
    """Load a curated snapshot file BY PATH under a unique module name + smem symbol.

    Use this ONLY for a level that is not reachable from an existing module + flag (so the
    snapshot can coexist with the production kernel in one process without their module-global
    SmemAllocator symbols colliding). `smem_sym` becomes the unique global symbol name.

    NOTE: not needed by any current level — all 14 map to an existing module + env. Kept for
    structural parity with the MoE loader and for future bespoke snapshots.
    """
    path = Path(path)
    mod_name = f"_fmha_level_{path.stem}"
    if str(_KERNELS) not in sys.path:
        sys.path.insert(0, str(_KERNELS))
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    # The snapshot is expected to read its smem global symbol from this attribute if present,
    # so two snapshots / the production kernel don't share a SmemAllocator global_sym_name.
    module.__dict__["_SMEM_SYM_OVERRIDE"] = smem_sym
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
