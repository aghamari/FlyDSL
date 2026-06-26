# SPDX-License-Identifier: Apache-2.0
"""Scenario 1 — Row-major LDS (baseline, HIGH bank conflicts).

LDS layout:  make_ordered_layout((TM, TK), (1, 0))   ->  stride (TK, 1)  (k contiguous)

Transpose read: consecutive lanes read consecutive m at a fixed k -> addresses
m*64 + k.  bank = (m*64 + k) % 32 = k % 32  for ALL m  ->  every lane in the wave
targets the SAME bank -> worst-case conflict. This is the baseline.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_bank_conflicts/01_row_major.py
"""

import flydsl.expr as fx
from lds_common import TM, TK, run_case

if __name__ == "__main__":
    run_case(
        "row_major",
        "Scenario 1: row-major (baseline)",
        TM * TK,
        lambda: fx.make_ordered_layout((TM, TK), (1, 0)),
        expect="HIGH conflicts on transpose read (all lanes -> same bank)",
    )
