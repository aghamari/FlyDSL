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
* The dominant throughput win — **LDS row-padding (hk5, 2.03× at sq32768)** — riding on the
  two load-bearing structural wins (**column-V**, delete the transpose; and **diagonal-pair
  causal tiling**), plus a stack of VALU/scheduling levers and a **per-seqlen base dispatch**
  that picks the faster of {log2dom, hk5} for each shape.

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

Device-fair TF, this kernel (**per-seqlen-BASE dispatch**, `best_base()` in
`kernels/fmha_prefill_fp8_dispatch.py`) vs the CK-Tile fp8 reference. Each seqlen runs the
faster base at its own optimal `(KT, DIAG)`: **log2dom** for sq≤2048, **hk5** for sq≥16384.
Measured 2026-06-18 (graph-replay, bs=1 nq8 nk1 causal); see `results.tsv` for the raw rows.

| seqlen | this kernel (TF) | CK-Tile fp8 (TF) | gap | base @ (KT,DIAG) |
| ------ | ---------------- | ---------------- | ----- | ----------------- |
| 1024   | 26  | 30  | 1.15× | log2dom, KT=64 DIAG=0 |
| 2048   | 55  | 62  | 1.13× | log2dom, KT=64 DIAG=1 |
| 16384  | 129 | 141 | 1.09× | hk5, KT=32 DIAG=0 |
| 32768  | 142 | 146 | 1.03× | hk5, KT=32 DIAG=1 |

The large-seq gap is nearly closed — **1.03× at sq32768, 1.09× at sq16384** — and is the
residual **0.2.0-scheduler structural wall**: the backend won't fill the softmax `exp` shadow
with independent MFMA, so the VALU between the two MFMA bursts is exposed. It is *not* memory-
or transpose-bound (column-V already removed that). The small-seq gap (1.13–1.15× at
sq1024/2048) is instead **grid-fill / dispatch-bound** — the grid is < 80 CU at sq1024 — which
is why the small-seq dispatch family is the next frontier (see below).

## Optimization log (summary)

Every kept lever, in order. Reproduced by `reproduce_levels.py`. Levers fall into four
families: **baseline**, **structural** (remove wasted work), **throughput** (do the work
faster), **dispatch** (launch / schedule only what's needed). Device-fair graph-replay TF,
measured 2026-06-18; `x_prev` / `x_ck` are taken at **sq32768** (the largest shape).

| #  | lever | family | sq1024 | sq2048 | sq16384 | sq32768 | x_prev | x_ck |
|----|-------|--------|--------|--------|---------|---------|--------|------|
| 0  | baseline naive fp8 | baseline | 11 | 17 | 28 | 29 | — | 0.20× |
| 1  | multiwave BM=128 / 4-wave | throughput | 15 | 20 | 47 | 53 | 1.83× | 0.36× |
| 2  | cooperative K/V → LDS (ping-pong) | structural | 15 | 20 | 47 | 53 | 1.00× | 0.36× |
| 3  | register-P ds_bpermute transpose | throughput | 15 | 20 | 47 | 53 | 1.00× | — |
| 4  | fast `exp2` softmax | throughput | 15 | 20 | 47 | 53 | 1.00× | — |
| 5  | causal masked/unmasked loop split | structural | 15 | 20 | 47 | 53 | 1.00× | — |
| 6  | diagonal-pair causal tiling | structural | 13 | 30 | 52 | 57 | 1.08× | 0.39× |
| 7  | column-V (delete transpose) | structural | 14 | 33 | 63 | 70 | 1.23× | 0.48× |
| 8  | **LDS row padding (bank conflicts) = hk5** | throughput | 19 | 48 | **123** | **142** | **2.03×** | 0.97× |
| 9  | kdlds: K-descale → LDS | throughput | 19 | 48 | 107 | 121 | **0.85×** ↓ | 0.83× |
| 10 | LOG2E-descale + exp-bias hoist = log2dom | throughput | 18 | 46 | 115 | 131 | 1.08× | 0.90× |
| 11 | XCD/chiplet block-ID remap | dispatch | 18 | 46 | 116 | 131 | 1.00× | — |
| 12 | softmax VALU fold (`v_max3`+p_scale) | throughput | 18 | 46 | 116 | 131 | 1.00× | — |
| 13 | per-seqlen (KT,DIAG) dispatch / log2dom | dispatch | 26 | 55 | 106 | 131 | — | 0.90× |
| —  | **CK-Tile fp8 (reference)** | reference | 30 | 62 | 141 | 146 | — | — |

> **Levels 8–13 above were measured at the driver default (KT=32, DIAG=1, except L13 which
> dispatches `(KT,DIAG)` per shape over the log2dom base).** At each seqlen's *optimal*
> `(KT,DIAG)`, plain **hk5** hits **25 / 52 / 129 / 142** and **log2dom** hits
> **26 / 55 / 106 / 131**. The production **current best = per-seqlen-BASE dispatch**
> (`best_base()`): pick the faster base per shape — log2dom for sq≤2048, hk5 for sq≥16384 —
> giving **26 / 55 / 129 / 142** (the headline `Result` table above).

Two findings reshape the old story:

* **L8 (LDS row-padding = hk5) is the single dominant win — 2.03× at sq32768** — and it is the
  large-seq peak. The two structural levers that enable it are column-V (L7, removes the
  V-transpose DS-wait, LDS-wait 54%→18%) and the diagonal-pair causal balancer (L6, +8–24% at
  sq≥2048).
* **The kdlds → log2dom stack REGRESSES at large seq.** L9 (kdlds) drops sq16384 123→107
  (0.85×) and sq32768 142→121; L10 (log2dom) only partly recovers (115/131), still below hk5's
  123/142. log2dom *wins only at sq≤2048*. This **overturns the prior "log2dom is the peak"
  claim** — hence `best_base()` runs hk5, not log2dom, at large seq.

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

### 9 — kdlds: stage K-descale in LDS (`_combined`, throughput; LARGE-SEQ REGRESSION)
The per-token K descale is staged into LDS (ping-ponged with the K/V tiles) so it is off the
score-scaling critical path during the MFMA. **This is a measured regression at large seq:**
sq16384 123→107 (0.85×) and sq32768 142→121, while sq≤2048 are flat (19/48). The extra LDS
traffic outweighs the saved scalar work once the kv loop is long. A/B: `_ck_hk5` → `_combined`.

### 10 — LOG2E-into-descale + exp-bias hoist (`_ck_log2dom`, throughput)
Fold `LOG2E` into the descale constant so the **entire score domain is in log2 units**,
removing the per-element `*LOG2E` before `exp2`; and hoist the per-tile exp bias out of the
inner loop. `fmha_prefill_fp8_layout` and `fmha_prefill_fp8_reorder` are its readability
rewrites (same lowering). It **partly recovers the L9 regression** (sq16384 107→115, sq32768
121→131) **but is still below the hk5 peak** (123/142) at large seq, and only *wins* at
sq≤2048 (where it beats hk5 26/55 vs 25/52 at each shape's optimal `(KT,DIAG)`). This
**overturns the earlier "log2dom is the measured peak" claim** — log2dom is the small-seq base,
hk5 is the large-seq base, and `best_base()` (L13) dispatches between them. A/B: `_combined` →
`_ck_log2dom`.

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

### 13 — Per-seqlen (KT, DIAG) dispatch + per-seqlen BASE dispatch (dispatch; current best)
`KT`, `DIAG`, and the underlying base are compile-time constexpr (read from env at import).
The sweep found the per-seqlen optimum is `(base, KT, DIAG)`, and the production
`best_base()` in `kernels/fmha_prefill_fp8_dispatch.py` encodes it:

| seqlen | base | KT | DIAG | this kernel | note |
|--------|------|----|------|-------------|------|
| ≤1024  | log2dom | 64 | 0 | 26 | log2dom wins small seq; KT=64 amortizes softmax VALU |
| ≤2048  | log2dom | 64 | 1 | 55 | diag pairing helps once there are enough tiles |
| ≤16384 | hk5 | 32 | 0 | 129 | hk5 beats log2dom at large seq; KT=64 REGRESSES (116→86), back to 32 |
| else   | hk5 | 32 | 1 | 142 | diag + KT32 on hk5 = large-seq peak |

The earlier story dispatched only `(KT,DIAG)` over a single base; the **new finding is that
the base itself must be chosen per shape** (log2dom for sq≤2048, hk5 for sq≥16384), because
the kdlds/log2dom stack regresses large seq (see L9/L10). Pinning to one base costs ~10% at
sq16384 (log2dom 106 vs hk5 129). The driver takes `--base` to force a single underlying
module for A/B; `best_base()` is the per-shape mix that produces the `Result` headline.
**Gotcha:** env is read at import and the `SmemAllocator` finalizes once per process, so each
seqlen class must be built in a fresh process — the driver forks one per (level, seqlen).

## Ready / cheap unbenchmarked variants

These modules exist in `kernels/` but were **never measured against the current best**
(`best_base()` per-seqlen base dispatch). Each is a candidate A/B that could be wired into
`LEVELS`:

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

Measured **this session** (2026-06-18), top group; pre-existing below.

| lever | why |
|-------|-----|
| **F2 vectorized score-descale** (`v_pk_mul` via `fx.Vector` + `from_elements`, pack 16 scalar muls) | correct (err 0.039) but **109 TF vs hk5 128 @ sq16384 — regression**; the `from_elements` assembly adds VGPR/moves that outweigh the saved scalar muls (variant `kernels/fmha_prefill_fp8_vdescale.py`, worktree `fmha-vdescale-d3eda332`) |
| **F4 partial `lgkmcnt(2)`** in the ds_bpermute P-transpose (relax the full LDS drain) | correct (err 0.039/0.043) but **129 TF vs hk5 128 — neutral**; the full drain wasn't a binding stall (`kernels/fmha_prefill_fp8_waitcnt.py`, worktree `fmha-waitcnt-785de923`) |
| **F3 K=32 atom** `mfma_f32_16x16x32_fp8_fp8` | NOT built — it is a *smaller* atom than hk5's 32×32×16 (8192 vs 16384 MAC/inst), so it would **double the MFMA instruction count** and worsen the softmax cross-lane reduction (2 butterflies vs 1). Structural non-win (scaffold `kernels/fmha_prefill_fp8_mfma16.py`) |
| **K=128 scaled atom** `mfma_scale_f32_16x16x128_f8f6f4` | **gfx950/CDNA4-only** — wired only in `CDNA4/MmaAtom.cpp` (OCP `Float8E4M3FN`, not gfx942's E4M3FNUZ); no gfx942 selection pattern, cannot run on MI308X. The widest gfx942 fp8 atom (32×32×16) is already in use (scaffold `kernels/fmha_prefill_fp8_mfma128.py`) |
| **split-KV / flash-decoding** (S=2 on log2dom, grid 64→128, fused FlyDSL combine) | correct (err 0.043/0.047) but **device-fair 20/43 TF vs 26/55 @ sq1024/2048 — regression**; the combine pass (8192/16384 wg) + per-split Q-reload/prologue exceeds the CU-fill gain, and the VALU-bound kernel amortizes softmax over fewer MFMAs (`kernels/fmha_prefill_fp8_splitkv.py`, worktree `splitkv-57c5be53`). Refines `ck_splitk`, same verdict |
| **persistent kernel** (small seq) | provable no-op: sq1024/2048 grid (64 items) ≤ 80 CUs so it is not oversubscribed; wall-time is floored by the single heaviest *indivisible* q-tile, and per-launch overhead is already removed by the graph-replay metric (`kernels/fmha_prefill_fp8_persist.py`, worktree `persist-fmha-9d371c3e`) |
| XOR swizzle (vs padding) | +27 VGPR, worse occupancy; padding wins |
| pad + XOR together | no gain over padding alone |
| split-K | wash / regress (`fmha_prefill_fp8_ck_splitk`) |
| 8-wave + 8-wave ping-pong | regressed 101→72 TF |
| v5 128-kv softmax | slower |
| v6 / v13 2-rep & GEMM2 sw-pipeline | +31 VGPR, no MFMA↔VALU interleave from the scheduler |
| v14 `v_perm` transpose | no win over ds_bpermute / column-V |
| KT > 32 (at large seq) | VGPR / occupancy regress (sq16384 116→86) |
| kdlds K-descale → LDS (at large seq) | regresses sq16384 123→107 (0.85×); kept only as the sq≤2048 base (L9) |
| ck_async KT=128 + DMA | only 32/35 TF (`fmha_prefill_fp8_ck_async`) |
| v12 async-on-padded | wrong results (err 2.48); `buffer_load_to_lds` broken in 0.2.0 |
| hot_loop_scheduler / sched_group_barrier | scheduler interleaves MFMA↔MEM, not MFMA↔VALU |
| wider 128-bit O store | impossible (output layout) |
| NWAVES ∈ {2,8} | slower than 4 |
| maxnreg / waves-per-eu caps | dead in the 0.2.0 wheel |

## Status / remaining gap (structural ceiling reached)

All DSL/wheel levers are measured and exhausted (the dead-ends above). Current best =
per-seqlen-base dispatch **26 / 55 / 129 / 142 TF** vs CK-Tile 30/62/141/146 (**0.87–0.97×**).
The residual is the structural ceiling for FlyDSL 0.2.0 on gfx942, confirmed by ISA
(hk5 @ sq16384: VGPR **157 → 3 waves/SIMD**, VALU:MFMA **~36:1**, 0 spills, LDS 18.9 KB):

* **Large seq** — VALU/softmax-bound with no independent MFMA to hide it. The MFMA axis is
  exhausted (widest gfx942 fp8 atom 32×32×16 already used; K=128 is gfx950-only), and the only
  occupancy lever (`maxnreg`/waves-per-eu, to reach 4 waves) is dead in the wheel.
* **Small seq** — grid-fill-bound (grid < 80 CU at sq1024), but both grid-fill levers are
  measured dead: split-KV regresses (combine + per-split overhead), persistent is a no-op
  under graph-replay (grid not oversubscribed).

Closing the last gap requires capabilities **outside the DSL/wheel**: gfx950 (the K=128 scaled
atom) or external-LLVM occupancy control — the same structural residual the `fused_mega_moe`
example hit at small batch.

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
| `../kernels/fmha_prefill_fp8_dispatch.py` | **production current best** — `best_base()` per-seqlen base dispatch (log2dom ≤2048, hk5 ≥16384) |
| `../kernels/fmha_prefill_fp8_ck_hk5.py` | LDS-padded base (L8); the large-seq peak |
| `../kernels/fmha_prefill_fp8_ck_log2dom.py` | log2-domain base (L10); the small-seq base |
| `../kernels/fmha_prefill_fp8_vdescale.py` | F2 vectorized score-descale (dead end — regression) |
| `../kernels/fmha_prefill_fp8_waitcnt.py` | F4 partial `lgkmcnt(2)` P-transpose drain (dead end — neutral) |
| `../kernels/fmha_prefill_fp8_mfma16.py` | F3 K=32 atom scaffold (structural non-win, not built) |
