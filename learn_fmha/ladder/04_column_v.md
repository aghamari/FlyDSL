# Ladder 04 — Column-major V (Lesson 17)

One change vs 03: V is stored **column-major `[d, kv]`** in global memory. GEMM2 contracts over kv,
so its V operand — 8 contiguous kv for a fixed d — becomes a single **wide load** instead of the
8-byte row-major gather. The transpose isn't moved (rung 03), it is **deleted**, because the layout
makes the contraction axis already contiguous.

This is the punchline of Part D: when a cost is irreducible in the current layout, change the layout
so the op disappears. Neutral at single-wave (~29.5 us, still occupancy-bound); the ~20% win shows
at large seqlen with a filled grid (structural tier). Assumes `sk % 32 == 0`.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/04_column_v.py`
