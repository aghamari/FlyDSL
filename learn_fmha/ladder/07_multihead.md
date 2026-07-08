# Ladder 07 — Multihead (Lessons 22b/22c) — ENDPOINT

Final rung. Everything in 06 plus a **head dimension**. Occupancy is a grid-mapping property: a
single head's `ceil(sq/BM)` workgroups can't fill 80 CUs, but `nq` heads give
`grid = nq * ceil(sq/BM)`, which does. A 2-D grid (`block_idx.x` = q-tile, `block_idx.y` = head)
offsets Q/K/V/O per head.

This unlocks the throughput the earlier rungs were starved of:

```
sq=256   nq=8   4.5 TFLOPS
sq=1024  nq=8  24.9 TFLOPS
sq=2048  nq=8  50.8 TFLOPS   (vs 6.56 single-head at sq=2048)
```

The full layout-algebra endpoint of the ladder (the sibling of `lesson_22c`). Inputs are per-head:
`Q[nq,sq,HD]`, `K[nq,sk,HD]`, `V[nq,HDV,sk]` (column-major), `O[nq,sq,HDV]`.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/07_multihead.py 2048 2048 1 8`
