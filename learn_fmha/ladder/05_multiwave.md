# Ladder 05 — Multiwave (Lesson 08)

First structural rung. The local tier ran one wave on one of 80 CUs. Here a workgroup holds
`NWAVES` waves (default 4), each owning its own 32 q-rows, so `BM = NWAVES*32` and
`grid = ceil(sq/BM)`. Each wave still loads K/V independently (redundant across waves that share a
kv range) -> rung 06 fixes that.

Benched in TFLOPS at seqlens (single head): sq=1024 -> 2.56 TF, sq=2048 -> 5.24 TF. The occupancy
win only fully appears with the head grid (rung 07). `NWAVES` is env-overridable (`FMHA_NWAVES`).

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/05_multiwave.py`
