# FMHA fp8 Optimization Ladder (grow-to-production, layout-algebra)

A **series of real, standalone kernels**, each = the previous kernel plus **one** optimization,
climbing from a correct fused fp8 attention kernel to a multiwave/multihead endpoint. This
realizes the tutorial's original intent that lessons 08+ describe but never delivered as a
per-step kernel ladder. Every rung is self-checking (err < 6e-2 vs a torch fp8 reference) and
timed with **CUDA graphs** (launch overhead stripped, so the real kernel delta is visible).

All kernels are fp8 e4m3fnuz, 32x32x16 MFMA, `HD=HDV=128`, online softmax, causal-capable.

## Layout-algebra scope (how much of the layout API we actually use)
- **Both GEMMs** run through the layout API: one typed `fx.make_mma_atom(MFMA(32,32,16,f8))` feeds
  a `make_tiled_mma`, whose derived A/B/C fragment layouts drive **`fx.gemm`**.
- **Global Q/K loads** (rungs 00-05) use **`make_tiled_copy_A/B` + `partition_S` + `retile`** — real
  tiled copies, not hand-rolled `buffer_load` byte math.

What genuinely **stays direct** (and *why* — these are the skill's "algebra does not fit" cases):
- fp8 pack/unpack (`cvt_pk_fp8_f32`, i64 bitcasts) — bespoke register packing.
- The online-softmax reduction (`shuffle_xor` butterfly) — a reduction, not a layout op.
- The P-transpose bridge (LDS reload, then `ds_bpermute`) — GEMM1's C-layout != GEMM2's B-layout,
  which tiled-copy/fx.gemm cannot express (the chained-GEMM problem).
- The cooperative-LDS K/V staging (rungs 06-07) — a bespoke coop layout; it IS the optimization.
- The epilogue store — GEMM2 outputs `O[d,q]` but O is stored `[q,d]`, an f32->bf16 transpose-scatter
  that `tiled_copy_C` cannot express cleanly.

**This is the same hybrid scope as the production layout rewrite `kernels/fmha_prefill_fp8_layout.py`**
— which is the key point: the *best* FlyDSL kernels are also hybrid (direct `buffer_ops` +
`mma_atom_call_ssa`), because tiled-copy/`fx.gemm` cannot express the chained attention GEMMs, the
transposes, or the paged gathers. More tiled-copy algebra than this does not exist for this kernel
class; it would not compile or would not be faster.

## The ladder

### Local tier (single-wave workgroups; BM=32, grid=ceil(sq/32))
| rung | file | optimization | lesson |
|---|---|---|---|
| 00 | `00_baseline.py` | correct fused fp8 attention (LDS P-transpose, row-major V gather, generic exp2) | 05-07 |
| 01 | `01_fast_exp2.py` | `rocdl.exp2` instead of `Float32.exp2()` | 13 |
| 02 | `02_causal_bound.py` | cap the kv loop at the causal limit | 14 |
| 03 | `03_register_p.py` | `ds_bpermute` P-transpose (no LDS) | 12 |
| 04 | `04_column_v.py` | column-major V -> transpose deleted | 17 |

### Structural tier (waves + grid so occupancy wins are real)
| rung | file | optimization | lesson |
|---|---|---|---|
| 05 | `05_multiwave.py` | NWAVES waves/workgroup (BM=NWAVES*32) | 08 |
| 06 | `06_cooperative_lds.py` | share one K/V LDS tile across waves | 10 |
| 07 | `07_multihead.py` | head grid (`grid = nq * ceil(sq/BM)`) — ENDPOINT | 22b/22c |

## Measured scoreboard (GPU 2, CUDA-graph timing)

Local tier is occupancy-starved (single wave on 1 of 80 CUs), so the local tricks are ~neutral
on wall time — exactly the tutorial's point (a good trick only helps if the kernel is bound by
what it improves). The wins appear in the structural tier once the grid fills the machine.

Local tier (sq=256, causal, single head):
```
00_baseline      31.1 us
01_fast_exp2     29.6 us   (VALU micro-opt, ~neutral here)
02_causal_bound  28.3 us   (less work, but under-occupied)
03_register_p    28.1 us   (moves the transpose off LDS store/reload)
04_column_v      29.5 us   (transpose deleted; neutral until grid fills)
```

Structural tier (sq=2048, causal):
```
05_multiwave          single-head   5.24 TFLOPS
06_cooperative_lds    single-head   6.56 TFLOPS   (+25% vs 05: no redundant K/V loads)
07_multihead (nq=8)   50.8  TFLOPS  (head grid fills the 80 CUs)
```

Endpoint `07_multihead` across seqlens (nq=8, causal):
```
sq=256     4.5 TFLOPS
sq=1024   24.9 TFLOPS
sq=2048   50.8 TFLOPS
sq=4096  102.5 TFLOPS
```

## Comparison to the production kernels (fair CUDA-graph timing)

Measured device-fair (all CUDA-graph; the standard `do_bench` harness includes ~0.3ms host
overhead that hides the real kernel, so it is NOT used here). Shape: b1, nq=8, causal.

```
                              sq=2048        sq=4096
this ladder  07_multihead     50.8 TF        102.5 TF
production   fmha_prefill_fp8_ck        48 TF        (~ck-family)
production   fmha_prefill_fp8_ck_log2dom  68 TF      158 TF   (the BEST)
```

So: **the ladder endpoint is on par with the mid production kernel (`_ck`) and reaches ~65-75% of
the best (`_ck_log2dom`)**. The remaining gap to the best is NOT layout algebra — it is the extra
structural/codegen tricks the best kernel stacks (LDS row padding, K-descale-in-LDS, exp-bias hoist,
diagonal-pair tiling, per-shape KT/DIAG dispatch) plus paged-KV/GQA machinery. This matches the
tutorial's capstone finding: past the algorithm+layout wins, the ceiling is instruction scheduling
and register allocation, which the DSL abstracts away.

(Caveat: the ladder is non-paged, per-tensor-scale, single-batch multihead, so it carries less
memory-system overhead than the paged/GQA/per-token production kernels — same FLOPs, simpler I/O.)

## Pushing further: `08_tuned` and where the wall is (a PMC-driven pass)

Following the optimization-runbook, I profiled the endpoint with `rocprofv3` at sq=2048, nq=8:

```
SQ_WAIT_INST_LDS / SQ_BUSY_CU_CYCLES = 0.6%     -> NOT LDS-bound
SQ_INSTS_VALU    / SQ_INSTS_MFMA     = 18.4 : 1 -> VALU-bound (drowning in ALU per matrix op)
```

This is *why* the best production kernel beats the ladder: the gap is **VALU/structural, not layout
algebra**. `08_tuned.py` applies what helped and documents what didn't:

- KEPT: LOG2E-into-descale (log2 domain, the `log2dom` idea), `maxnreg=96` (+~4%), `fast_fp_math`
  (+~1%). -> ~51 TF @2048, ~105 TF @4096.
- DEAD END (measured): **LDS row padding** (`hk5`'s +66% win) does nothing here (0%) — LDS is not
  our bottleneck at this shape. **Diagonal-pair** (Lesson 16) *regresses* -9% — with nq=8 the grid
  (128 wg) already fills the 80 CUs, so halving it to 64 wg under-fills. (Diagonal-pair only helps
  single-head / very large seq where the causal imbalance dominates.)

Honest verdict: `08_tuned` is **on par with the mid production kernel** (`_ck`, 48 TF) and reaches
**~65-75% of the very best** (`_ck_log2dom`, 68/158 TF). We did NOT beat the best. The remaining gap
is deeper VALU reduction (e.g. the `hk_fexp` Schraudolph fast-exp *approximation*, which trades
accuracy) plus per-shape KT/DIAG dispatch and the paged/hk5 large-seq regime — optimizations that
either don't transfer to this shape/config or trade correctness for speed. Per the tutorial's own
capstone: past the algorithm+layout wins, the ceiling is instruction scheduling and register
allocation, which the DSL abstracts away.

## How to run
```bash
# one rung, one shape
HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/04_column_v.py 256 256 1
HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/07_multihead.py 2048 2048 1 8

# a rung across its default shapes (forks one process per shape for the LDS finalize gotcha)
HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/06_cooperative_lds.py
```

## The transferable method (from the tutorial)
1. Build correct first; verify every layer (each rung re-checks err < 6e-2).
2. Measure with CUDA graphs; classify the bottleneck before optimizing.
3. Change ONE thing per rung; re-measure the binding number.
4. Prefer layout wins (column-V deletes the transpose) over micro-ops (fast-exp2).
5. Occupancy is a grid-mapping property — the multihead grid, not any single trick, unlocks the
   throughput.

## Notes / constraints
- `_bench.py` holds the shared torch fp8 reference, correctness check, and CUDA-graph timer.
- Rungs that use LDS (00, 02 via 01 lineage no; 00/06/07 use LDS; 03/04/05 are LDS-free except
  where noted) fork one process per shape (the module-global `SmemAllocator.finalize()` runs once
  per process — the lesson_07 gotcha).
- Column-V rungs assume `sk % 32 == 0` (full kv-tiles) so the wide column-V load never crosses a
  d-row boundary; the ladder's shapes satisfy this.
- Deferred (further production steps, not pure layout-algebra): paged KV cache, GQA, diagonal-pair
  tiling, ping-pong prefetch. These live in `kernels/fmha_prefill_fp8_*`.
