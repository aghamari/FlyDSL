---
name: fmha-prefill-autoresearch
description: Autonomous edit->run->measure->keep/discard optimization loop for the FlyDSL FP8 paged causal FMHA batch-prefill kernel on AMD MI308X / gfx942, targeting the gap to CK-Tile (currently ~1.2-1.4x) above the canonical hk5 baseline. Use when asked to autoresearch / autonomously optimize the fp8 FMHA prefill kernel, run the FMHA optimization loop overnight, add+measure an attention lever, or "do autoresearch on the prefill attention". Fuses Karpathy's autoresearch loop discipline with the FMHA lever menu + measured dead-end catalog, scoped to this kernel and hardware.
argument-hint: [a lever to try, a seqlen to focus, or nothing to run the loop]
---

# FMHA Prefill Autoresearch — optimize fp8 batch-prefill attention on MI308X (gfx942)

An **autonomous, measured optimization loop** for the FlyDSL **FP8 paged causal
FMHA batch-prefill** kernel on **AMD MI308X (gfx942 / CDNA3)**. It fuses two methods:

- **autoresearch** (`/workspaces/amir/autoresearch/program.md`, Karpathy): the
  never-stop edit -> commit -> run -> measure -> keep-if-better / revert-if-not loop,
  logged to a `results.tsv`.
- **flydsl-fmha-prefill-opt** (the sibling skill): the diagnose-before-you-change
  FMHA lever menu, the structural wins already in hk5, and the **measured dead-end
  catalog** so the loop never re-tries a ruled-out lever.

**Goal:** close the gap to CK-Tile. The canonical baseline **hk5** is at 5/18/103/123
TF @ sq 1024/2048/16384/32768; CK-Tile fp8 is 30/62/141/146; PyISA asm is higher
still. The realistic prize is the **~1.2-1.4x** to CK-Tile at large seq — but that gap
is structural (VALU/softmax scheduling + VGPR occupancy), so see *Honest ceiling*
before expecting a multiplier.

> The baseline kernel `kernels/fmha_prefill_fp8_ck_hk5.py` is already heavily tuned
> (= ck + LDS padding). This loop's job is to find a NEW lever that beats it, OR to
> confirm a hypothesis is a dead-end and record it. Most "obvious" levers are already
> in the dead-end catalog — read it first (§ Dead-ends) or you'll waste iterations.

---

## The metric and the rules
- **Primary metric:** kernel time in **ms** / **TFLOPS**, from `do_bench`
  (warmup=10, rep=50, median), via `tests/kernels/bench_fmha_compare.py`. Higher TF /
  lower ms is better. Headline is **TF at the 4 customer seqlens** (bs=1, nq8, nk1,
  causal), and the **ratio to hk5** and to CK-Tile.
- **Correctness gate FIRST, every time:** `ck_check.py` must print `OK` (err < 6e-2,
  fp8 tolerance) on the shape(s) you touched, BEFORE any timing counts. A fast wrong
  kernel is a crash.
- **One lever at a time**, gated for correctness, measured, then kept or reverted.
  **Never stack two unverified levers.**
- **NEVER claim a speedup/regression from reasoning alone — let the measurement
  decide** (FlyKAT empirical-loop rule). And **never produce two consecutive hypotheses
  without an intervening measurement**: if you've reasoned twice in a row, run the
  cheapest command that can disprove the current hypothesis before reasoning again.
- **Cheapest-falsifier escalation ladder** (run the cheapest test that can kill the
  lever, escalate only when it can't decide):
  **(a) ISA-diff** — did the compiled kernel even change? Identical ISA = no-op, stop
  early and discard. Changed != win, escalate. ->
  **(b) bench** — is it actually faster (TF up at the touched seqlen)? ->
  **(c) PMC/profile** — why/where (did the binding counter move?).
- **Confirm structural changes in the ISA / PMC**, not just the stopwatch — a change
  that doesn't move VGPR/spills, the instruction histogram, or the binding PMC counter
  almost certainly did nothing (see § ISA/PMC check).
- **Category effort order** (FlyKAT canonical, cheapest-leverage-first):
  `algorithmic > fusion > tuning > wrapper`. **Drop a lever after 2 failed attempts**
  within a category; **stop a whole category after 3 consecutive no-improvement
  attempts** and switch to the next. (For this kernel `algorithmic`/`fusion` = the
  cross-tile-pipeline / VALU-reduction levers; `tuning` = KT/NWAVES/pad sweeps;
  `wrapper` = launch/dispatch. The big gap lives in `algorithmic`.)
- **A stall is never a reason to stop a lever or the loop** — only *measured* diminishing
  returns trigger a drop/category-stop. When stalled with no falsifiable hypothesis, run
  `profile collect` / `ir dump` to gather evidence, then re-form the hypothesis.
- **Measurement noise rule (learned the hard way):** parallel-GPU sweeps are
  UNRELIABLE for small/fast shapes (sq1024/2048 swung 2.3x run-to-run). Small shapes
  need **isolated + repeated** single-GPU runs. Parallel sweep is OK only for the
  large/slow shapes (sq16384/32768).

---

## Environment (read before running anything)
- **All 8 GPUs (0-7) work** — pick whichever is FREE at run time (don't hardcode a GPU).
  Check freeness: `rocm-smi --showmeminfo vram` (idle ≈ 297 MB used) + `rocm-smi
  --showpids` (no KFD PIDs = free), then `HIP_VISIBLE_DEVICES=<free id>`. (GPU 0/1 used
  to be broken — fixed 2026-06-15; obsolete rule.) For fan-out, put each concurrent run
  on a different free GPU.
- FlyDSL is a **wheel** (0.2.0) at `/opt/venv/.../site-packages/flydsl` — **no C++
  rebuild**, so any lever needing a backend/codegen change (e.g. real maxnreg) is out.
- Work in the FlyDSL repo: `/workspaces/amir/FlyDSL`.
- HW: 80 CU, 4 SIMD/CU, 256 VGPR/thread, 512 VGPR-banks/SIMD (occupancy = 512/vgpr_count
  waves/SIMD), 64KB LDS, ~5.3 TB/s HBM, ~1.3 PFLOPS fp8.
- **FlyDSL module-global SmemAllocator finalizes once per process -> ONE shape per
  process.** `ck_check.py` and the bench already fork a subprocess per shape; respect
  that (don't try to loop shapes in one python process).

```bash
cd /workspaces/amir/FlyDSL
# canonical bench (default kernel = hk5; add your variant module to compare):
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 <your_variant> > run.log 2>&1
# large shapes only, no PyISA (fast iteration on the meaningful gap):
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py \
  --kernels <your_variant> --seqs 16384 32768 --no-pyisa > run.log 2>&1
# also CK-Tile (the target; needs aiter built):
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py --ck ...
# correctness, ONE shape per process: module b sq sk nk gqa causal page_size [pscale]
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/ck_check.py <your_variant> 1 16384 16384 1 8 1 16
```
Read results out of the log without flooding context:
```bash
grep -E "ms/|TF|ERR|OK|FAIL" run.log | tail -20
```
Bench row format: `b1 sq16384 nq8 nk1   X.XXXms/YTF ...`. Check row:
`<module> ... -> ERR x.xxxx OK  (KT=.. NBUF=.. BM=..)`.

### Prefer FlyKAT for measurement when the kernel is registered as a FlyKAT op
**FlyKAT** (`/workspaces/amir/flykat`, the FlyDSL Kernel Authoring Toolkit) is a
stateless verify/bench/profile framework. It standardizes exactly this loop and gives
the cheap falsifiers the ladder above wants. Use it **if/when the FMHA prefill kernel is
registered as a FlyKAT op** (today only `hstu_attention` exists — a template, not our
paged-fp8 prefill; see the `flydsl-fmha-prefill-opt` skill's note). Command map:
```bash
cd /workspaces/amir/flykat   # uv pip install -e . ; uv pip install flydsl
flykat verify  <op> --kernel flydsl:<variant> --set sq=16384 ...   # correctness oracle
flykat bench   <op> --kernel flydsl:<variant> --trials 3 ...        # stable timing
flykat ir resources <op> --kernel flydsl:<variant> --set ...        # VGPR/LDS/occupancy
flykat ir check <op> --kernel flydsl:<variant> --baseline <tag> ... # ★ ladder step (a): ISA changed?
flykat profile collect <op> --kernel flydsl:<variant> ...           # ladder step (c): counters
flykat profile diff <base.json> <mine.json>                         # quantitative "did it help"
flykat inspect <op> --kernel flydsl:<variant> --ir --profile --out ...  # one-shot deep pass
```
`flykat ir check ... --baseline` IS ladder step (a) (identical ISA = no-op, stop early);
`flykat bench` is (b); `flykat profile collect`/`diff` is (c). Until the kernel is a
FlyKAT op, the raw `bench_fmha_compare.py` / `ck_check.py` / `FLYDSL_DUMP_IR` path above
is the fallback that does the same three jobs.

---

## The loop (autoresearch, scoped to FMHA prefill)
Work on a dedicated branch off the canonical kernel. **First establish the hk5
baseline number on THIS machine** (autoresearch "first run is the baseline" rule) —
don't trust the table's numbers blindly; re-measure.

**LOOP FOREVER (until interrupted):**
1. **Baseline first.** Bench hk5 on the 4 seqlens (large shapes isolated + repeated),
   record TF + ms + the ratio to CK-Tile into `results.tsv`. Also dump hk5's
   VGPR/LDS/spills and the binding PMC so you know which bottleneck you're attacking.
2. **Pick ONE lever** from the menu (§ Lever menu). Prefer the diagnosis-driven one
   (read the PMC/ISA first; don't guess). **Skip anything in § Dead-ends.**
3. **Implement it as a NEW variant file** — copy `fmha_prefill_fp8_ck_hk5.py` to a new
   `fmha_prefill_fp8_<lever>.py`, give it a unique `SmemAllocator global_sym_name`, and
   make the minimal isolated change. (Don't edit hk5 in place — keep the baseline
   pristine and the diff reviewable.) Prefer an **env-guarded knob** when the lever is
   a toggle, so on/off is one process apart.
4. **Correctness gate:** `ck_check.py` on the touched shape(s). If not `OK`, the lever
   is wrong — fix or discard. Don't measure speed yet.
5. **Measure** with the bench (large shapes isolated + repeated; small shapes need
   their own isolated repeats). Compare TF to hk5 on the same machine, same session.
6. **Confirm structural levers in ISA/PMC** (§ ISA/PMC check). If the histogram /
   binding counter didn't move, the change is inert — discard.
7. **Keep or discard:**
   - Beats hk5 (higher TF) with `OK` correctness -> commit
     ("[Opt] FMHA prefill: <lever> (+X% @ sqN)"), update `results.tsv`, and consider
     promoting into hk5 (or a new canonical) once you trust it across all 4 seqlens.
   - Equal/worse, or helps one seqlen but hurts another -> discard the variant (or keep
     the file as a documented negative, matching the repo's convention of keeping
     `_splitk`/`_8wave`/`_async` as negatives), log the reason.
8. **Record both outcomes** — wins AND informative failures — so the loop never
   re-tries a dead lever. Then go to 2.

**NEVER STOP** once the loop has begun. If you run out of ideas: re-read the PMC to
re-classify the bottleneck, combine two near-misses that touch disjoint mechanisms,
read the CK-Tile / PyISA reference for a structural idea not yet tried, or read the
`learn_fmha/` lessons for an angle. The loop runs until the human interrupts.

**Timeout:** a bench over a couple seqlens is well under a minute once compiled; first
JIT compile of a new variant can take a couple minutes. If a run hangs past ~10 min,
kill it and treat the lever as a failure.

**Crashes:** dumb+easy (typo, missing import) -> fix and re-run. Fundamentally broken
idea -> log "crash"/"discard" and move on.

---

## Fan-out then fold (parallelizing levers)
Lever attempts are independent experiments — run them concurrently, serialize only the
decision. Pick up to **4 mutually-exclusive attempts** (different levers, or grid points
of one lever needing source edits), one **subagent per attempt** in a single message,
each with `isolation: "worktree"` so concurrent edits don't collide (see the
`git-worktree-workflow` skill).

**CRITICAL environment constraint:** subagents here are **sandbox-blocked from running
python/GPU** (they return "BASH BLOCKED"). So subagents can only PREPARE isolated
variant files in their worktrees; **all correctness + timing must run from the main
session**, serially. Plan the fan-out as "N agents each author one variant" then "main
session benches them one by one and folds the winners." Folding: keep only variants that
pass `OK` AND beat hk5; partition by compatibility (two levers that both grow LDS can
blow the 64KB ceiling; two that both add VGPR can't both fit 3-wave occupancy); apply
compatible winners together on a fresh copy and **RE-MEASURE** (combined != product).

---

## The op (one-paragraph spec)
FP8 paged causal FMHA prefill, HD=128, fp8 e4m3 **FNUZ**, bf16 out, paged KV
**vec_k_col_v**, per-token-head Q/K descale, per-head V descale, p_scale, GQA,
page_size=16. Customer: HunyuanVideo 3.0 (AITERKER-112), **bs=1**. GEMM1 = K@Q^T
(S as [kv,q]), softmax (online, register-resident P via ds_bpermute, fast exp2),
GEMM2 = V^T@P (O as [d,q]), all `mfma_f32_32x32x16_fp8_fp8`. Headline seqlens:
**1024 / 2048 / 16384 / 32768** (bs=1 nq8 nk1 causal).

---

## The lever menu (each a gfx942 hypothesis — diagnose, then try)
Ordered by where the *measured* bottleneck is (VALU:MFMA ~19:1, VGPR-pinned at 3
waves/SIMD). Most memory/LDS/transpose levers are already won or dead — the open
frontier is **hiding softmax VALU behind MFMA** and **occupancy**.

| # | Lever | Hypothesis | Status going in |
|---|---|---|---|
| 1 | **Cross-tile software pipeline w/ INDEPENDENT MFMA** | issue GEMM1(tile i+1) MFMAs into softmax(i) VALU to hide the 19:1 VALU | the one structural idea that *could* close the gap; hard in 0.2.0 (scheduler won't auto-interleave, +VGPR risk). Prior naive 2-rep pipeline regressed (+31 VGPR) — needs a smarter, occupancy-neutral formulation. |
| 2 | **Reduce softmax VALU** | fewer ops in the exp/max/sum/rescale path; cheaper rescale skip when corr~1 | VALU is the wall; any real VALU cut that doesn't add VGPR is a direct win. (maxnumf-vs-maximumf was tried, slower.) |
| 3 | **KT (outer kv tile) sweep** | bigger KT amortizes barriers/loads | KT>32 regressed (VGPR/occupancy) — but re-test if a VALU/occupancy lever first frees headroom. |
| 4 | **NWAVES / TILE_BM** | occupancy vs reuse | NWAVES!=4 didn't help; re-test only if occupancy changes. |
| 5 | **p_scale / exp bias folding** | fold more constants out of the inner loop | small VALU wins compound. |
| 6 | **K-descale-to-LDS (kdlds)** | stage K-descale in LDS off the score-scaling critical path | `fmha_prefill_fp8_ck_kdlds.py` exists, correct, UNBENCHMARKED — measure it vs hk5 as a ready first experiment. |
| 7 | **Per-seqlen dispatch** | DIAG on for sq>=2048, off for sq1024 (grid-halving loss) | diagonal-pair loses at sq1024; a seqlen-gated default could lift the small shape. |

Lever #1 and #2 attack the actual bottleneck; prioritize them. #6 is the cheapest
ready experiment (a file already exists). #3/#4 only after occupancy changes.

---

## ★ DEAD-ENDS (measured & ruled out — do NOT spend loop iterations here)
From the `flydsl-fmha-prefill-opt` skill + auto-memory. Each was built and measured:
- **XOR swizzle** (instead of padding): +27 VGPR, net loss (VGPR-bound). Padding wins.
- **pad + XOR combined**: worse than padding alone.
- **split-K**: wash. **8-wave**: regressed 101->72.
- **sched_group_barrier interleave (hk8)**: neutral. **GEMM2 sw-pipeline (hk10/v13)**:
  regressed. **per-shape config tuning**: defaults already optimal.
- **wider output store 64->128b**: impossible (MFMA lane layout) AND irrelevant (O
  written once). Our reads are already widest (28x dwordx4).
- **async global->LDS DMA (FMHA_BUFK)**: broken (err 2.48) — incompatible with our LDS
  padding (gfx942 DMA scatters lane i -> m0+i*4, can't hit padded rows). Mutually
  exclusive with the padding that is our biggest win.
- **scheduler `hot_loop_scheduler` port**: exists but weaves MFMA<->MEMORY; our gap is
  MFMA<->VALU and there is NO sched_valu primitive. hk4 regressed 61->58.
- **maxnreg / waves-per-eu occupancy cap**: BLOCKED — the wheel lowers the hint to
  `--amdgpu-num-vgpr`/`--amdgpu-waves-per-eu` which DON'T EXIST in LLVM (silently
  dropped; VGPR never moved). The real mechanism is a C++ function attribute,
  unreachable from a wheel.

If you think you have a new angle on one of these, state explicitly what NEW
information changes the verdict before building it.

---

## ISA / PMC check (for structural levers)
```bash
# VGPR / LDS / spills:
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/x FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/ck_check.py <module> 1 16384 16384 1 8 1 16
# /tmp/x/<kernel>_0/19_gpu_module_to_binary.mlir -> vgpr_count, vgpr_spill_count,
#   group_segment_fixed_size (LDS bytes); 21_final_isa.s for the instruction mix.
# occupancy = 512 / vgpr_count waves/SIMD (166 -> 3; need <=128 for 4).
# PMC via rocprofv3 (pmc.txt lines): SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_LDS
#   SQ_BUSY_CU_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_LDS ; then sqlite:
#   SELECT n.name,SUM(e.value) FROM rocpd_pmc_event e
#     JOIN rocpd_info_pmc n ON e.pmc_id=n.id GROUP BY n.name;
```
Diagnostic ratios: VALU:MFMA (is it still ~19:1?), LDS-bank-conflict %, LDS-wait/busy,
VGPR vs the 128/166 occupancy cliffs. **A lever that doesn't move the binding counter
did nothing**, regardless of the stopwatch.

---

## results.tsv (keep updated; leave untracked by git)
Tab-separated (commas break descriptions). One row per (variant, seqlen):
```
commit	variant	seqlen	tflops	ms	vgpr	status	description
```
`status` in {keep, discard, crash}. `keep` only after `OK` correctness AND beating hk5
at that seqlen. This table (measured on THIS machine) is the deliverable.

---

## Honest ceiling
hk5 is a strong local optimum; the residual ~1.2-1.4x to CK-Tile is **structural**:
softmax VALU between the two MFMA bursts with no independent MFMA to hide it (lever #1),
and VGPR-pinned 3-wave occupancy with the only cap-lever broken in the wheel. A real win
almost certainly comes from lever #1 (a smart cross-tile pipeline that stays
occupancy-neutral) or a genuine VALU reduction (#2) — not from re-trying memory/LDS
levers, which are won or dead. **Profile to confirm the bottleneck before investing in
any one lever.** If a full batch of fan-out levers returns no composable win, re-profile
and re-classify before the next batch.

---

## Related
- **flydsl-fmha-prefill-opt** — the static knowledge (lineage, wins, full dead-end
  catalog, harness reference). This skill is its autonomous-loop instance.
- **git-worktree-workflow** — fan-out isolation (+ the subagent-can't-run-GPU caveat).
- **cdna-kernel-opt / lds-optimization / gemm-optimization / kernel-trace-analysis**
  (KB skills) — generic CDNA lever method.
- **autoresearch** (`/workspaces/amir/autoresearch/program.md`) — the loop discipline.
- Auto-memory: `feedback-fmha-perf-lessons`, `feedback-hk-amd-kernel-tricks`,
  `feedback-fmha-global-access-levers`, `project-fmha-flydsl-port`.

## One-sentence takeaway
> Run the autoresearch keep/discard loop over the FMHA lever menu, scoped to the fp8
> prefill kernel on gfx942: re-baseline hk5 first, one new-variant lever at a time,
> correctness (ck_check OK) before speed, large shapes measured isolated+repeated,
> structural wins confirmed in ISA/PMC, dead-ends never re-tried — and aim the effort
> at hiding softmax VALU / buying occupancy, because that (not memory) is the measured
> wall between hk5 and CK-Tile.
