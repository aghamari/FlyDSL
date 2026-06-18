<!-- SPDX-License-Identifier: Apache-2.0 -->
# FMHA Prefill — fp8 paged causal attention (FlyDSL, gfx942)

A single-launch **fp8 e4m3 FNUZ paged causal FMHA prefill** kernel authored in FlyDSL
(Python → MLIR → HSACO) for AMD MI308X / gfx942, modeled on AMD's CK-Tile
`BlockFmhaBatchPrefillPipelineQRKSVSAsync`. This is a standalone optimization case study
with its own reproduction driver, structurally mirroring
`ck_dsl/examples/gfx950/fused_mega_moe`.

The dataflow is one kernel, no HBM round-trip for the scores:

```
GEMM1  S = K @ Qᵀ   ([kv, q])  ->  online softmax (max / exp2 / sum, P register-resident
                                   via ds_bpermute)  ->  GEMM2  O = Vᵀ @ P   ([d, q])
```

> For the precise algorithm, data layout, and per-threadgroup steps (from the math up), see
> [ALGORITHM.md](./ALGORITHM.md). This file is the optimization history: every lever, the
> gotchas, and the results.

## What it shows

* A correct single-kernel paged causal FMHA in fp8 e4m3 FNUZ with per-token-head Q/K
  descale, per-head V descale, GQA (nq8/nk1), `vec_k_col_v` paged KV, and bf16 output.
* A runbook-disciplined path: hypothesis → parity gate (`ck_check.py`, err<6e-2) →
  device-fair best-of-N perf (`bench_fmha_fair.py`) → keep/revert → record, every level
  reproducible from one driver (`reproduce_levels.py`).
* The two load-bearing structural wins — **column-V (delete the transpose)** and
  **diagonal-pair causal tiling** — plus a stack of VALU/scheduling throughput levers.

## Hardware / software

| GPU    | AMD Instinct MI308X (gfx942 / CDNA3), 4 XCDs, MFMA fp8 |
| ------ | ----------------------------------------------------- |
| ROCm   | flydsl 0.2.0 wheel (VERIFY-ON-GPU: exact ROCm version on the box) |
| shapes | bs=1, nq=8, nk=1, hd=128, causal, page_size=16; seqlens 1024 / 2048 / 16384 / 32768 |
| dtype  | fp8 e4m3 **FNUZ** Q/K/V, f32 per-token-head (Q,K) & per-head (V) scales, bf16 output |
| atom   | `mfma_f32_32x32x16_fp8_fp8` for both GEMMs |

> All timings are **device-fair** (CUDA-graph replay via `bench_fmha_fair.py`, NOT `do_bench`
> which adds ~0.3 ms host dispatch that dwarfs small-seq kernels). Only **same-session**
> ratios are meaningful (the box thermally throttles, so absolute ms drift between runs).

## Result

Device-fair TF, this kernel (per-seqlen dispatch over `fmha_prefill_fp8_ck_log2dom`) vs the
CK-Tile fp8 reference. **All TF cells are placeholders — run `reproduce_levels.py` to fill
them.** The expectation column is the handoff's last-measured numbers (confirm, don't trust).

| seqlen | this kernel (TF) | CK-Tile fp8 (TF) | gap | expectation (this / CK) |
| ------ | ---------------- | ---------------- | --- | ----------------------- |
| 1024   | TBD (run reproduce_levels.py) | TBD | TBD | ~26 / ~30 (1.12x) |
| 2048   | TBD (run reproduce_levels.py) | TBD | TBD | ~55 / ~62 (1.13x) |
| 16384  | TBD (run reproduce_levels.py) | TBD | TBD | ~116 / ~141 (1.22x) |
| 32768  | TBD (run reproduce_levels.py) | TBD | TBD | ~131 / ~146 (1.11x) |

The residual ~1.1–1.22x is the **0.2.0-scheduler structural wall**: the backend won't fill
the softmax `exp` shadow with independent MFMA, so the VALU between the two MFMA bursts is
exposed. It is *not* memory- or transpose-bound (column-V already removed that).

## Optimization log (summary)

Every kept lever, in order. Reproduced by `reproduce_levels.py`. Levers fall into four
families: **baseline**, **structural** (remove wasted work), **throughput** (do the work
faster), **dispatch** (launch / schedule only what's needed). `before→after` TF is at the
largest measured seqlen unless noted; **fill from the driver** (`results.tsv`).

| #  | lever | family | A → B kernel | before→after TF | x |
|----|-------|--------|--------------|-----------------|---|
| 0  | baseline naive fp8 | baseline | `fmha_prefill_fp8` | TBD (run reproduce_levels.py) | — |
| 1  | multiwave BM=128 / 4-wave | throughput | `fmha_prefill_fp8` → `_8wave` (`NWAVES=4`) | TBD | TBD |
| 2  | cooperative K/V → LDS (ping-pong) | structural | embedded in `_8wave` | TBD | TBD |
| 3  | register-P ds_bpermute transpose | throughput | embedded in `_8wave` | TBD | TBD |
| 4  | fast `exp2` softmax | throughput | embedded in `_8wave` | TBD | TBD |
| 5  | causal masked/unmasked loop split | structural | embedded in `_8wave` | TBD | TBD |
| 6  | diagonal-pair causal tiling | structural | `_8wave` → `_v7` (`DIAG`) | TBD | TBD |
| 7  | column-V (delete transpose) | structural | `_v7` → `_ck` (`VCOL`) | TBD | TBD |
| 8  | LDS row padding (bank conflicts) | throughput | `_ck` → `_ck_hk5` (`KPAD/VPAD=8`) | TBD | TBD |
| 9  | kdlds: K-descale → LDS | throughput | `_ck_hk5` → `_combined` | TBD | TBD |
| 10 | LOG2E-descale + exp-bias hoist | throughput | `_combined` → `_ck_log2dom` | TBD | TBD |
| 11 | XCD/chiplet block-ID remap | dispatch | `FMHA_XCD=0→1` (C=4) on `_ck_log2dom` | TBD | TBD |
| 12 | softmax VALU fold (`v_max3`+p_scale) | throughput | baked into `_ck_log2dom` | TBD | TBD |
| 13 | **per-seqlen (KT,DIAG) dispatch** | dispatch | dispatch over `_ck_log2dom` | TBD | TBD |

The two biggest wins are **structural**: column-V (lever 7, removes the V-transpose DS-wait,
LDS-wait 54%→18%) and the diagonal-pair causal balancer (lever 6, +8–24% at sq≥2048).
The **dispatch** family (11, 13) closes the per-shape gap — KT=64 helps small seq but
regresses large, so it must be chosen per shape. **Top of the ledger (L13) is the current best.**

## Each lever, in depth

### 0 — Baseline naive fp8 (`fmha_prefill_fp8`, baseline)
Correctness-first port of the PyISA `f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv`:
BM=32, 1 wave (64 threads), P transposed through an LDS scratch. Establishes the parity
contract (signature, layouts, descale convention) every later level must keep.

### 1 — Multiwave BM=128 / 4-wave (`_8wave`, throughput)
BM=128 with 4 waves / 256 threads. BM=128 measured uniformly fastest on MI308X (better
occupancy / latency-hiding than the 256×128 / 8-wave variant). **Gotcha:** smaller BM ⇒ more
q-tiles ⇒ more workgroups, which fills all CUs at small seqlen — but NWAVES∈{2,8} are
measured dead-ends, so 4 is fixed. A/B: `FMHA_NWAVES` 2/4/8 on the same module.

### 2 — Cooperative K/V → LDS, double-buffered (`_8wave`, structural)
The KV tile is loaded cooperatively into LDS and **ping-ponged** (buffer `b` at
`_K_OFF + b*_K_BYTES`) so the next tile's load overlaps the current tile's compute.
**Gotcha:** no clean standalone flag A/B — it is the load path itself, so its effect appears
as part of the L0→L1 delta, not a toggle.

### 3 — Register-resident P via ds_bpermute (`_8wave`, throughput)
The L0 baseline transposes P (the softmax probabilities) through an LDS scratch before GEMM2.
The 8wave path keeps P in registers and transposes it with `ds_bpermute` (a DS-unit lane
shuffle), deleting the LDS P buffer. **Gotcha:** embedded — the win is the L0→L1/L3 delta.

### 4 — Fast `exp2` softmax (`_8wave`, throughput)
Softmax uses `rocdl.exp2` with scores pre-scaled by `LOG2E`, replacing a slower `exp`.
This is the seed of lever 10, which later folds `LOG2E` all the way into the descale so the
per-element `*LOG2E` disappears entirely.

### 5 — Causal masked/unmasked loop split (`_8wave`, structural)
Split the KV loop into a masked tail (the diagonal tiles that need the per-element causal
mask) and an unmasked interior (fully-visible tiles that skip the mask VALU). CK does the
same. **Effect:** VALU:MFMA 24→19 (+13%). Documented on 8wave; it is foundational to the CK
line below.

### 6 — Diagonal-pair causal tiling (`_v7`, structural)
Each CTA processes q-tile `t` **and** its causal mirror `num_q_tiles-1-t`, so a light early
tile and a heavy late tile share a workgroup — the CK causal load-balancer. **+8–24% at
sq≥2048, a LOSS at sq1024** (too few tiles to pair), which is exactly why DIAG is chosen
per-shape at L13. A/B: `_8wave` (one tile/WG) → `_v7`, or `FMHA_DIAG=0/1` on the CK base.
`BM` exported to the bench becomes `2*TILE_BM` when DIAG is on (the grid divisor halves).

### 7 — Column-V, delete the transpose (`_ck`, structural; biggest structural win)
CK's true `vec_k_col_v` stores V **column-major** (`[pages, nk, hd, page_size]`) so the GEMM2
contraction dim (kv) is contiguous per `(head, d)`. The V→LDS copy becomes one 128-bit store
per slot instead of the 16× `ds_write_b8` scatter-transpose the row-major path needs — the
gfx942 V-transpose DS-wait **disappears entirely**. PMC: LDS-wait 54%→18% of busy cycles.
**Gotcha:** requires the matching col-V pool — `pack_paged_cache(v_col=True)`; the harness
reads `K.V_COL` to pack it, so a kernel that sets `VCOL` MUST export `V_COL`. This disproves
the earlier claim that the transpose DS-wait is irreducible / needs gfx950 `ds_read_tr`.
A/B: `_v7` → `_ck`, or `FMHA_VCOL=0/1`.

### 8 — LDS row padding for bank conflicts (`_ck_hk5`, throughput)
Baseline `SQ_LDS_BANK_CONFLICT` = 68% of busy cycles: K LDS rows have stride HD=128 B =
32 banks×4 B, so consecutive kv rows alias to the SAME bank (up to 32-way conflict on the
`ds_read`). Pad each LDS row by `FMHA_KPAD`/`FMHA_VPAD` bytes so the row stride is coprime-ish
with the 32-bank (128 B) period. **Swept optimum: KPAD=VPAD=8** (108.5 TF @ sq16384, LDS
18944 — beats 16/16's 106.8 with *less* LDS). The response is **non-monotonic** (KPAD=0 →
conflicts return; VPAD=4 → 59; VPAD=32 → 75). Padding affects only LDS buffer strides; global
strides keep HD/KT. A/B: `_ck` → `_ck_hk5`, or `FMHA_KPAD/VPAD=0` vs `8`.

### 9 — kdlds: stage K-descale in LDS (`_combined`, throughput)
The per-token K descale is staged into LDS (ping-ponged with the K/V tiles) so it is off the
score-scaling critical path during the MFMA. A/B: `_ck_hk5` → `_combined`.

### 10 — LOG2E-into-descale + exp-bias hoist (`_ck_log2dom`, throughput)
Fold `LOG2E` into the descale constant so the **entire score domain is in log2 units**,
removing the per-element `*LOG2E` before `exp2`; and hoist the per-tile exp bias out of the
inner loop. This is the **measured-peak underlying kernel**; `fmha_prefill_fp8_layout` and
`fmha_prefill_fp8_reorder` are its readability rewrites (same lowering). Device-fair it
reaches ~109/129 TF graph-timed (116/131 rocprof) @ sq16384/32768 vs hk5 103/122.
A/B: `_combined` → `_ck_log2dom`.

### 11 — XCD / chiplet block-ID remap (`_ck_log2dom`, dispatch)
MI308X has 4 XCDs; HW routes physical block `b` → XCD `b % 4`. Invert that round-robin so
`XCD_C=4` consecutive *logical* blocks (ordered qhead-fast ⇒ all GQA q-heads of one q-tile
share the identical causal K/V range) land on the SAME XCD's private L2. **Small win because
the kernel is VALU-bound** (memory off the critical path): sq16384 110→117, sq32768 138→140,
L2 hit 95.1%→95.7%. C∈{3,4,5} all tie. A/B: `FMHA_XCD=0` vs `1` (C=4).

### 12 — Softmax VALU fold (`_ck_log2dom`, throughput)
Fold the running-max update into `v_max3` (3-input max in one VALU op) and fold `p_scale`
into the descale. **Baked into log2dom**; no clean standalone flag, so its effect is part of
the L10/L11 delta.

### 13 — Per-seqlen (KT, DIAG) dispatch (dispatch; current best)
`KT` and `DIAG` are compile-time constexpr (read from env at import), and the sweep found the
per-seqlen optimum is purely `(KT, DIAG)`:

| seqlen | KT | DIAG | note |
|--------|----|------|------|
| ≤1024  | 64 | 0    | KT=64 amortizes softmax VALU at small seq |
| ≤2048  | 64 | 1    | diag pairing helps once there are enough tiles |
| ≤16384 | 32 | 0    | KT=64 REGRESSES large seq (116→86), back to 32 |
| else   | 32 | 1    | diag + KT32 = large-seq peak |

**Gotcha / discrepancy:** the production `kernels/fmha_prefill_fp8_dispatch.py` currently
dispatches over `fmha_prefill_fp8_ck_hk5`, **not** `_ck_log2dom` (the measured peak). The
driver therefore takes `--base` to choose the underlying module and defaults to `_ck_log2dom`.
Because env is read at import and the SmemAllocator finalizes once per process, each seqlen
class must be built in a fresh process — the driver forks one per (level, seqlen).

## Ready / cheap unbenchmarked variants

These modules exist in `kernels/` but were **never measured against the current best**
(`_ck_log2dom` dispatch). Each is a candidate A/B that could be wired into `LEVELS`:

| module | what it is |
|--------|-----------|
| `fmha_prefill_fp8_ck_kdlds` | standalone K-descale-to-LDS (pre-`_combined` isolation) |
| `fmha_prefill_fp8_ck_expfma` | exp via FMA-fused bias path |
| `fmha_prefill_fp8_log2` | log2-domain scores (pre-log2dom variant) |
| `fmha_prefill_fp8_jit` | JIT-config variant |
| `fmha_prefill_fp8_noprefetch` | prefetch ablation (isolates the prefetch win) |
| `fmha_prefill_fp8_v9` / `_v10` / `_v11` | later PyISA-line experiments |

## Gotchas & nuances (cross-cutting)

* **Process isolation is mandatory.** Every `FMHA_*` knob is read at import, and FlyDSL's
  module-global `SmemAllocator` finalizes once per process. One process = one (level, seqlen)
  build. The driver forks a subprocess per cell and sets env *before* the child imports.
* **Measurement.** Device-fair graph replay only (`bench_fmha_fair.py`); `do_bench` adds
  ~0.3 ms host dispatch that makes small-seq kernels look far slower than they are. Only
  same-session ratios are valid (thermal throttle).
* **Small shapes are noisy.** sq1024/2048 have tiny grids; parallel multi-GPU runs are
  unreliable for them. The driver measures small seqlens **isolated on one GPU, repeated**,
  and keeps the best; large seqlens fan out across GPUs.
* **GPU policy.** Default GPUs `0,1,3,4,5,6,7`; **GPU 2 is never used.**
* **Parity contract.** A kernel that sets `VCOL`/`V_COL` MUST export `V_COL` so the harness
  packs the matching column-V pool; otherwise GEMM2 reads a transposed/garbage V. Parity gate
  is `ck_check.py` with err<6e-2.
* **Levers couple.** Diagonal pairing helps only with enough tiles (sq≥2048); KT=64 helps
  small but regresses large — both are why L13 dispatches per shape rather than fixing one config.

## Dead ends (reverted — kept so they aren't re-tried)

| lever | why |
|-------|-----|
| XOR swizzle (vs padding) | +27 VGPR, worse occupancy; padding wins |
| pad + XOR together | no gain over padding alone |
| split-K | wash / regress (`fmha_prefill_fp8_ck_splitk`) |
| 8-wave + 8-wave ping-pong | regressed 101→72 TF |
| v5 128-kv softmax | slower |
| v6 / v13 2-rep & GEMM2 sw-pipeline | +31 VGPR, no MFMA↔VALU interleave from the scheduler |
| v14 `v_perm` transpose | no win over ds_bpermute / column-V |
| KT > 32 (at large seq) | VGPR / occupancy regress (sq16384 116→86) |
| ck_async KT=128 + DMA | only 32/35 TF (`fmha_prefill_fp8_ck_async`) |
| v12 async-on-padded | wrong results (err 2.48); `buffer_load_to_lds` broken in 0.2.0 |
| hot_loop_scheduler / sched_group_barrier | scheduler interleaves MFMA↔MEM, not MFMA↔VALU |
| wider 128-bit O store | impossible (output layout) |
| NWAVES ∈ {2,8} | slower than 4 |
| maxnreg / waves-per-eu caps | dead in the 0.2.0 wheel |

## Reproduce

`reproduce_levels.py` is the single, self-contained entry point. From the FlyDSL repo root:

```bash
# whole ledger (parity + device-fair perf, all 4 seqlens, all 14 levels)
python3 fmha_research/reproduce_levels.py

# the current best only, at the large shape
python3 fmha_research/reproduce_levels.py --levels 13 --seqs 16384

# a subset of levers, two shapes, also measure the CK-Tile reference row
python3 fmha_research/reproduce_levels.py --levels 7,10,13 --seqs 1024,16384 --ck

# parity only (no perf), useful to validate every level still passes err<6e-2
python3 fmha_research/reproduce_levels.py --no-perf

# pin GPUs (GPU 2 is excluded even if listed); override the L13 dispatch base
python3 fmha_research/reproduce_levels.py --gpus 0,1,3 --base fmha_prefill_fp8_ck_hk5
```

Each run appends rows to [`results.tsv`](./results.tsv) (the machine-readable ledger) and
prints the numeric per-level table.

## File map

| path | purpose |
|------|---------|
| `README.md` | this document (optimization history + gotchas + results) |
| `ALGORITHM.md` | the precise algorithm, data layout, and per-threadgroup steps |
| `reproduce_levels.py` | self-contained per-level driver (parity + device-fair perf) |
| `results.tsv` | machine-readable ledger (appended by the driver) |
| `levels/_build_by_path.py` | kernel loader (import-by-name; path-load for future snapshots) |
| `levels/README.md` | level → module/env mapping; why no snapshot files are needed |
| `../kernels/fmha_prefill_fp8_ck_log2dom.py` | the measured-peak underlying kernel (L13 base) |
