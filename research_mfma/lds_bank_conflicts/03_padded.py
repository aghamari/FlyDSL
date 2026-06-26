# SPDX-License-Identifier: Apache-2.0
"""Scenario 3 — Row-major + padding (LOW conflicts, ~1.5% extra LDS).

LDS layout:  make_layout((TM, TK), (TK + PAD, 1))   ->  odd row stride (TK+PAD)

With PAD=1 (f32), the row stride is 65 elements. The transpose read walks
addresses m*65 + k; bank = (m*65 + k) % 32 = (m + k) % 32  ->  consecutive m hit
consecutive banks  ->  spread across all 32 banks (vs all-same-bank in scenario 1).
Padding makes the stride coprime-ish with 32 so the access cycles every bank.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_bank_conflicts/03_padded.py
"""

import flydsl.expr as fx
from lds_common import TM, TK, run_case

PAD = 1

if __name__ == "__main__":
    run_case(
        "padded",
        f"Scenario 3: row-major + {PAD}-element padding",
        TM * (TK + PAD),
        lambda: fx.make_layout((TM, TK), (TK + PAD, 1)),
        expect="LOW conflicts (odd stride spreads banks); +~1.5% LDS",
    )
