---
name: flydsl-fmha-prefill-opt
description: Optimize the FlyDSL FP8 paged causal FMHA prefill kernel on AMD MI308X / gfx942 (CDNA3), the port of the PyISA / CK-Tile batch-prefill attention for the HunyuanVideo customer. Use when asked to speed up the fp8 FMHA prefill kernel, add/measure an attention lever, pick the best FMHA variant, diagnose its bottleneck, or understand why a given attention optimization is a dead-end here. Carries the ck->hk5 lineage, the structural wins, the full measured dead-end catalog, and the bench/check harnesses so you don't re-investigate settled questions.
argument-hint: [a lever to try, a shape to focus, or nothing to summarize state]
---

# FlyDSL FP8 FMHA Prefill Optimization (MI308X / gfx942)

Optimization knowledge for the **FP8 paged causal FMHA prefill** kernel in FlyDSL,
a port of the hand-written PyISA assembly kernel
(`f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv`) and AMD's CK-Tile
`BlockFmhaBatchPrefillPipelineQRKSVSAsync`. Customer: HunyuanVideo 3.0
(AITERKER-112), **bs=1 only**. Target HW: **MI308X, gfx942, CDNA3.**

This skill is the distilled result of a long optimization session. Its highest value
is the **dead-end catalog** — many plausible attention levers were built, measured,
and ruled out with evidence. Do not re-investigate those without new information.

## Feature set (full parity, all variants preserve this)
HD=128 · fp8 e4m3 **FNUZ** · causal · bf16 out · paged KV **vec_k_col_v** ·
per-token-head Q/K descale · per-head V descale · p_scale · GQA · page_size=16.
All variants share the same `run_attn` signature + tensor layouts, so the bench /
correctness harnesses are drop-in across them.

## ★ THE BEST KERNEL: `kernels/fmha_prefill_fp8_ck_hk5.py`
This is the canonical, fastest FlyDSL FMHA kernel. Hand colleagues **hk5**.

| kernel | sq1024 | sq2048 | sq16384 | sq32768 | note |
|---|---|---|---|---|---|
| `fmha_prefill_fp8_ck` (base) | 5 | 16 | 61 | 70 | column-V + diagonal-pair, no padding |
| **`fmha_prefill_fp8_ck_hk5`** | **5** | **18** | **103** | **123** | ck + LDS row padding = BEST |
| CK-Tile fp8 (reference) | 30 | 62 | 141 | 146 | aiter production |
| PyISA asm (reference) | — | ~83 | higher | higher | hand-written |

TFLOPS, bs=1 nq8 nk1 causal. hk5 = `ck` + **one** win (LDS padding). Residual gap to
CK-Tile is ~1.2-1.4x and is the structural VALU/softmax-scheduling ceiling (below).

## The wins that got us to hk5 (in order of leverage)
1. **★ LDS row PADDING (the hk5 win, +66-73% at large seq).** Pad each K/V LDS row
   +16 bytes (`FMHA_KPAD=FMHA_VPAD=8`). K/V rows at stride HD=128B aliased all 32 LDS
   banks; PMC `SQ_LDS_BANK_CONFLICT` was **68% of busy**. Padding drops it to 15%,
   busy -43%. This OVERTURNED an earlier "irreducible gfx942 LDS-wait ceiling" claim —
   the "LDS-wait" was bank conflicts, fixed cheaply. Swept: KPAD=VPAD=8 is the optimum
   (better perf AND less LDS than the original 16/16).
2. **Column-V (`vec_k_col_v`, FMHA_VCOL=1).** Store V column-major so the GEMM2
   contraction dim (kv) is contiguous => **NO transpose**. V->LDS is one 128-bit store/
   slot vs a 16x `ds_write_b8` scatter. PMC LDS-wait 54%->18%. Disproves the "needs
   gfx950 ds_read_tr" claim — layout choice beats transpose tricks.
3. **Diagonal-pair tiling (FMHA_DIAG=1).** Each CTA does q-tile `t` + causal mirror
   `num_tiles-1-t` (shared live state, cheap VGPR). +8-24% at sq>=2048; a *loss* at
   sq1024 (grid halving) -> motivates per-shape dispatch.
4. **Masked/unmasked loop split** (CK does this): interior tiles skip per-element
   causal-mask VALU. VALU:MFMA 24->19, +13%.
5. Register-resident P (ds_bpermute transpose), fast `rocdl.exp2`, Q-loaded-once.

## Env knobs (defaults are the swept optimum — don't tune without reason)
`FMHA_NWAVES`(4 -> TILE_BM=128) · `FMHA_KT`(32, outer kv tile) · `FMHA_VCOL`(1) ·
`FMHA_DIAG`(1) · `FMHA_KPAD`(8) · `FMHA_VPAD`(8) · `FMHA_NBUF`(2) · `FMHA_BUFK`(0,
async K DMA — BROKEN, see dead-ends).

## ★★★ DEAD-END CATALOG (measured; do NOT re-investigate without new info)
Each was built as an isolated variant, measured, and ruled out.

- **XOR swizzle (instead of padding)** — eliminates bank conflicts too, but costs
  **+27 VGPR** for address math and we're VGPR-bound (3 waves/SIMD) -> net loss.
  Padding wins on gfx942 because LDS is abundant and addressing stays simple.
- **pad + XOR combined** — worse than padding alone (re-introduces conflicts).
- **split-K** — wash. (Also a `@flyc.jit` trap: `=None` default tensor args break the
  cache key; scratch tensors must be REQUIRED positional, pass tiny dummies when unused.)
- **8-wave** — regressed 101->72 (latency chain).
- **sched_group_barrier MFMA/VALU/EXP interleave (hk8)** — NEUTRAL. Single-tile
  dependency; the 0.2.0 scheduler already orders reads->MFMA tightly.
- **GEMM2 software-pipeline / cross-tile pipeline (hk10/v13)** — regressed (+31 VGPR,
  occupancy drop; compiler won't interleave across the softmax dependency).
- **per-shape config tuning** — defaults already optimal everywhere swept.
- **wider output store 64b->128b (dwordx4)** — IMPOSSIBLE + irrelevant. mfma_f32_32x32x16
  gives each lane d-indices {j*8 + half*4 + e} (4-wide runs spaced 8 apart, gap owned by
  the other half-wave); no lane owns 8 contiguous d. AND O is written once vs K/V read
  O(sq) times, so output store width is negligible. Our reads are already the widest of
  the 3 kernels (28x buffer_load_dwordx4).
- **async global->LDS DMA (FMHA_BUFK / buffer_load_to_lds)** — BROKEN (err 2.48 vs
  0.043), and NOT a wheel codegen bug (ISA emits correct `buffer_load_dword ... lds` with
  uniform m0). gfx942 DMA scatters lane i -> `m0 + i*4` (uniform base + fixed lane
  stride); the LDS dest is NOT per-lane controllable, so it's **incompatible with our
  padded LDS layout** (_K_LDSW=34 dwords/row != clean 64-lane stride). To use DMA you'd
  have to drop padding for a swizzle (loses more than DMA gains) AND we're VGPR-bound so
  freeing VGPRs can't buy a 4th wave. PyISA/CK use DMA because they use XOR swizzles
  (compatible with m0+lane*4), not padding.
- **scheduler interleave / `hot_loop_scheduler` port** — the FlyDSL `hot_loop_scheduler`
  DOES exist (sched_mfma/sched_dsrd/sched_vmem/sched_dswr/sched_barrier — NOT hidden by
  the DSL). But it weaves MFMA<->MEMORY to hide global-load latency; our bottleneck is
  MFMA<->VALU (softmax exp2/ds_bpermute/max-sum between the two MFMA bursts). There is
  **no sched_valu primitive**, and K/Q load once so there are no independent memory ops
  to weave in. hk4 (the interleave) regressed 61->58.
- **maxnreg / waves-per-eu occupancy cap** — BLOCKED in the 0.2.0 wheel. The API exists
  (`CompilationContext.compile_hints({"maxnreg":N,"waves_per_eu":N})`,
  `autotune.Config(maxnreg=,waves_per_eu=)`) but the rocm backend lowers them to CLI opts
  `--amdgpu-num-vgpr` / `--amdgpu-waves-per-eu` which **DO NOT EXIST in LLVM** (confirmed:
  absent from `llc --help-hidden`; only `--amdgpu-function-calls` is real). Silently
  dropped: VGPR stayed 166 across all caps. The real mechanism is the LLVM function
  ATTRIBUTE `"amdgpu-waves-per-eu"="4,4"` (verified: llc reports NumVGPRsForWavesPerEU:97),
  attached in C++ — unreachable from Python since flydsl is a wheel (no C++ rebuild) and
  the bundled LLVM (23.0git) differs from system ROCm 7.2's llc.
- **fast-exp2 + correct max-freeze/rollback combined (`hk_fzx`, FMHA_FEXP×FMHA_FREEZE)** —
  the genuinely-unexplored combined lever: Schraudolph fast-exp2 (exp body) + a CORRECT
  MI350-style max-freeze (peel-tile-0 exact seed, frozen interior with corr≡1, wave-uniform
  overflow gate on the pre-cvt fma float, rare reconstruct-from-P rollback: v_min_u32 2^120
  clamp + per-row-gated delta=1/mx + FA_max+=log2(mx), no QK re-run). CORRECT (err 0.043–0.047
  at sq1024/16384/32768, and 0.047 on an adversarial P≈7e11 overflow input where the
  constant-seed probe goes NaN; transcendentals are HW-interlocked by LLVM on gfx942 — no
  gfx950 s_nop needed). DEVICE-FAIR NEUTRAL: +0% — interleaved within-file freeze on/off =
  118/118 (sq16384), 125/125 (sq32768); no combined cell clears +5% vs hk5 129/142. The
  freeze removes the o_acc corr-rescale but must add an equal-cost overflow-gate max reduction
  + peeled loops + 13 VGPR (157→170, occupancy edge), netting zero — re-confirms
  VALU/occupancy-ceiling bound, not rescale-bound. Verdict: keep
  `kernels/fmha_prefill_fp8_ck_hk_fzx.py` as an OFF-by-default reference impl of correct
  freeze+rollback; hk5+dispatch remains the optimum.
- **Clean-room PyISA rewrite from scratch (`fmha_prefill_fp8_pi_{base,mi300,mi350}.py`) — DEAD
  END, the wall is the WHEEL not hk5's code.** Built 3 from-scratch variants to the exact ABI
  (all CORRECT, err<6e-2 incl. adversarial overflow). (1) `pi_base`: minimal clean online-softmax
  baseline, 167 VGPR, 111/123 TF. (2) `pi_mi300`: faithful two-band 8-wave (waves0-3/4-7 over a
  256-row tile, shared K/V LDS) + resumable-softmax `s_setprio`/`sched_group_barrier` interleave —
  REGRESSES to 85/99 (8-wave −18% from LDS contention with no independent role; sched hints −8%,
  ISA shows the two MFMA bursts stay contiguous because there is no `sched_valu` and the intra-tile
  GEMM1→softmax→GEMM2 dependency leaves no MFMA to hide the softmax). (3) `pi_mi350`: Schraudolph
  fast-exp2 + correct max-freeze/rollback — fast-exp2 NEUTRAL (ISA `v_exp_f32` 34→2, VGPR 164→158,
  yet 0% device-fair → exp was never the bottleneck), max-freeze a −4/−6% regression (overflow-gate
  + rollback re-add the VALU the freeze removed, VGPR 158→169). None beats hk5 (123/142). Verdict:
  a from-scratch rewrite re-derives hk5's exact wheel-bound ceiling; do not re-attempt without a
  C++/external-LLVM toolchain.
- **External-LLVM occupancy escape hatch (`FLYDSL_COMPILE_LLVM_DIR`) — UNAVAILABLE on this box.**
  Phase-0 re-probe: `compile_hints({maxnreg:128, waves_per_eu:4})` on the default in-process path
  leaves VGPR at 157 (unchanged) — confirmed no-op. The flags DO get embedded into the
  `gpu-module-to-binary{opts=...}` fragment AND would reach codegen via the external path
  (`compiler/external_llvm.py`), but that path requires `<prefix>/bin/mlir-opt`, which is ABSENT in
  `/opt/rocm/llvm`, `/opt/rocm-7.2.0/llvm`, and the flydsl wheel. So 4 waves/SIMD cannot be forced
  here at all — the occupancy ceiling (the real binding constraint) is environmentally locked.

## The remaining gap (~1.2-1.4x to CK-Tile) is STRUCTURAL
- VALU:MFMA ~19:1; softmax VALU sits between the two MFMA bursts with no independent
  MFMA to hide it. Overlapping needs cross-tile pipelining with INDEPENDENT MFMA streams,
  which the 0.2.0 scheduler won't auto-do and the primitives can't express (no sched_valu).
- Occupancy is VGPR-pinned at 3 waves/SIMD (166 VGPR; 4 waves needs <=128) and the only
  lever to force it (maxnreg) is broken in the wheel.
- This is codegen/regalloc control the DSL+wheel don't expose, NOT an algorithm gap.

## Environment & how to run (gfx942)
- **All 8 GPUs (0-7) work** — pick whichever is FREE at run time (don't hardcode GPU 2).
  Check: `rocm-smi --showmeminfo vram` (idle ≈ 297 MB) and `rocm-smi --showpids` (no KFD
  PIDs = free), then `HIP_VISIBLE_DEVICES=<free id>`. (GPU 0/1 used to be broken — fixed
  2026-06-15; that rule is obsolete.) Examples below use `HIP_VISIBLE_DEVICES=<g>`.
- FlyDSL is a **wheel** (0.2.0) at `/opt/venv/.../site-packages/flydsl` — no C++ rebuild.
- HW: 80 CU, 4 SIMD/CU, 256 VGPR/thread, 512 VGPR-banks/SIMD (occupancy=512/vgpr_count
  waves/SIMD), 64KB LDS, ~5.3 TB/s HBM, ~1.3 PFLOPS fp8.

```bash
cd <FlyDSL repo>
# Benchmark hk5 vs PyISA (default kernel = hk5):
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py --seqs 1024 16384 --no-pyisa
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_compare.py --ck   # also CK-Tile (needs aiter built)
# Correctness, ONE shape per process (module-global smem can't re-finalize):
#   args: module b sq sk nk gqa causal page_size [pscale]
HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck_hk5 1 16384 16384 1 8 1 16
```

### Profiling (the measure->classify->one-change->re-measure loop)
```bash
# VGPR / LDS / spills:
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/x FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/ck_check.py <module> 1 16384 16384 1 8 1 16
# read /tmp/x/<kernel>_0/19_gpu_module_to_binary.mlir -> vgpr_count/sgpr_count/
#   group_segment_fixed_size/vgpr_spill_count ; 21_final_isa.s for the instruction mix.
# PMC via rocprofv3 (pmc.txt: SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_LDS SQ_BUSY_CU_CYCLES
#   SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_LDS), then query the sqlite results.db:
#   SELECT n.name,SUM(e.value) FROM rocpd_pmc_event e
#     JOIN rocpd_info_pmc n ON e.pmc_id=n.id GROUP BY n.name;
```

## Measurement discipline (learned the hard way)
- **Parallel-GPU sweeps are unreliable for small/fast shapes.** sq2048 read 29.2 then
  12.9 TF (2.3x swing) across parallel runs; isolated single-GPU repeated runs showed no
  real difference. RULE: parallel sweep OK for large/slow shapes only; small shapes need
  isolated + repeated measurement.
- **Cross-file / cross-process TFLOPS swings of ~20% are noise, not signal** — the same
  kernel can read e.g. 129 vs 142 TF depending on file/process; only trust a lever measured
  *interleaved within one file/process* (on/off in the same run). See the
  `flydsl-kernel-bench-discipline` skill for the device-fair methodology.
- One lever at a time; correctness (err < ~6e-2 for fp8) before speed; confirm structural
  changes in the ISA/PMC, not just the stopwatch.
- Probe the upper bound before building (is the bottleneck even what you'd be fixing?).

## Key files
| File | Role |
|---|---|
| `kernels/fmha_prefill_fp8_ck_hk5.py` | **BEST / canonical.** ck + LDS padding. |
| `kernels/fmha_prefill_fp8_ck.py` | column-V base (61/70), kept. |
| `kernels/fmha_prefill_fp8_ck_colv.py` | frozen 69TF fallback. |
| `kernels/fmha_prefill_fp8_ck_{splitk,8wave,async}.py` | documented negatives. |
| `kernels/fmha_prefill_fp8_ck_hk_fzx.py` | validated-negative reference: correct fast-exp2 + max-freeze/rollback, OFF by default. |
| `kernels/fmha_prefill_fp8_ck_hk_fexp.py` | fast-exp2 only (device-fair neutral). |
| `kernels/fmha_prefill_fp8_pi_base.py` | clean-room from-scratch baseline (correct, 111/123 TF); foundation for the pi variants. |
| `kernels/fmha_prefill_fp8_pi_mi300.py` | clean-room MI300 two-band + resumable-softmax — validated-negative (85/99 TF). |
| `kernels/fmha_prefill_fp8_pi_mi350.py` | clean-room MI350 fast-exp2 + max-freeze/rollback — validated-negative (neutral/regression). |
| `kernels/fmha_prefill_fp8_ck_kdlds.py` | K-descale-to-LDS experiment (unbenchmarked). |
| `tests/kernels/bench_fmha_compare.py` | unified bench (default kernel hk5). |
| `tests/kernels/ck_check.py` | single-shape correctness (packs col-V via V_COL). |
| `tests/kernels/fmha_prefill_fp8_ref.py` | torch reference + quant/pack helpers. |

## Related
- KB skills: `flydsl-kernel-authoring`, `flydsl-tile-programming`, `lds-optimization`,
  `gemm-optimization`, `cdna-kernel-opt`, `kernel-trace-analysis` — generic FlyDSL/CDNA
  method this specializes.
- Auto-memory: `feedback-fmha-perf-lessons`, `feedback-hk-amd-kernel-tricks`,
  `feedback-fmha-global-access-levers`, `feedback-fmha-optimization-ideas`,
  `reference-flydsl-fmha-tutorial`, `project-fmha-flydsl-port` (the source of these numbers).
- Tutorial: the `learn_fmha/` 23-lesson series rebuilds this kernel bottom-up.

## One-sentence takeaway
> hk5 (= ck + LDS row padding) is the best FlyDSL FMHA prefill kernel at 103/123 TF;
> the remaining ~1.3x gap to CK-Tile is the VALU/softmax-scheduling + VGPR-occupancy
> ceiling that the 0.2.0 wheel doesn't let us reach, and every global-access / scheduler /
> occupancy lever to close it has been measured and ruled out — optimize the measured
> bottleneck, not the plausible one.
