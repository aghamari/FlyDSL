# SPDX-License-Identifier: Apache-2.0
"""Scenario 2 — Column-major LDS (transpose read conflict-free, write strided).

LDS layout:  make_ordered_layout((TM, TK), (0, 1))   ->  stride (1, TM)  (m contiguous)

Now the transposed read (consecutive lanes -> consecutive m at fixed k) walks
addresses k*TM + m, i.e. stride 1  ->  consecutive banks  ->  conflict-free read.
The trade-off: the WRITE (consecutive lanes -> consecutive k at fixed m) now
strides by TM=64  ->  the conflict moves to the store side.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_bank_conflicts/02_column_major.py
"""

import flydsl.expr as fx
from lds_common import TM, TK, run_case

if __name__ == "__main__":
    run_case(
        "column_major",
        "Scenario 2: column-major",
        TM * TK,
        lambda: fx.make_ordered_layout((TM, TK), (0, 1)),
        expect="LOW conflicts on read (contiguous); conflict moves to the write",
    )
