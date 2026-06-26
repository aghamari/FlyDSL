# SPDX-License-Identifier: Apache-2.0
"""Scenario 4 — Row-major + XOR swizzle (LOW conflicts, NO extra LDS).

LDS layout:  make_composed_layout(Swizzle, row-major)

The swizzle permutes the physical address so the conflicting column read is spread
across banks, without padding (no wasted LDS). `SwizzleType.get(B, M, S)` is CK's
`make_xor_transform` analog: it XORs B bits at bit position M with the bits at
position M+S. With Swizzle(5, 0, 6) on a 64-wide f32 row:
    extract bits [6, 11) of the offset  = m (since k < 64)
    XOR them into bits [0, 5)           = the bank bits
=> physical bank = (k ^ (m & 31)), i.e. each row's columns are rotated to a
   different bank set. Consecutive-m reads now sweep all 32 banks (2-way floor),
   the same win as padding but with zero extra LDS.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_bank_conflicts/04_xor.py
"""

import flydsl.expr as fx
from lds_common import TM, TK, run_case


def xor_layout():
    base = fx.make_ordered_layout((TM, TK), (1, 0))            # row-major
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(5, 0, 6)), base)


if __name__ == "__main__":
    run_case(
        "xor",
        "Scenario 4: row-major + XOR swizzle",
        TM * TK,
        xor_layout,
        expect="LOW conflicts (swizzle spreads banks); ZERO extra LDS",
    )
