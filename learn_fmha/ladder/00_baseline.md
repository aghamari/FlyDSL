# Ladder 00 — Baseline fused fp8 attention

The bottom rung: a correct, un-optimized fused causal fp8 attention kernel that every later rung
optimizes one step at a time.

- Single head, a **grid of single-wave workgroups** (BM=32 q-rows, grid=ceil(sq/32)).
- 32x32x16 fp8 MFMA, runtime kv-loop, online softmax.
- Both GEMMs via **`fx.gemm`** (typed `make_mma_atom` + `make_tiled_mma` + `make_fragment_A/B/C`).
- Deliberately un-optimized (each becomes a rung): generic `Float32.exp2()`, loops all kv-tiles,
  P-transpose through LDS, row-major V byte-gather.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/00_baseline.py 256 256 1`
Measured (sq=256, causal): PASS, ~31 us (single-wave, occupancy-starved).
