---
name: flydsl-findings-log
description: Running log of FlyDSL performance + compiler-codegen findings on AMD MI308X/gfx942. After EVERY FlyDSL kernel / optimization / profiling / debug task, append what was learned — perf levers that worked or regressed, and where the FlyDSL→MLIR→LLVM compiler generates good vs bad code for specific instructions/ops — to FINDINGS.md. Use whenever working on FlyDSL kernels, layout algebra, profiling (ATT/PMC), or compiler codegen: READ it first to avoid re-deriving known results, and UPDATE it at the end of the task. Current optimization focus is SMALL shapes (sq1024/2048).
argument-hint: [nothing = read + append at end of task, or a specific finding to record now]
---

# FlyDSL Findings Log

A durable, append-only knowledge base of what actually works (and what the compiler
actually does) in this FlyDSL / gfx942 project — so no finding is discovered twice.

**The document:** [FINDINGS.md](FINDINGS.md) — read it, add to it.

## Mandatory protocol (every FlyDSL task)
1. **Start:** read [FINDINGS.md](FINDINGS.md) before profiling or editing. If a lever
   is already in "Dead-ends" or a codegen fact is already recorded, don't re-derive it.
2. **End:** append what you learned this task to the matching section of FINDINGS.md.
   Do this even for "nothing new" tasks — record the negative ("re-confirmed X").
   One or two lines per finding, dated, with the measured evidence.

## What counts as a finding (record all of these)
- **Compiler codegen quality** — the primary ask. Whenever you learn that FlyDSL/MLIR
  **does** or **does NOT** generate good code for a specific construct, record it:
  e.g. "`//`/`%` by a power of 2 is already lowered to shift/mask (explicit shift is a
  no-op)", "`fx.math.fma` → single `v_fma_f32`", "`rocdl.ballot` → SGPR (wave-uniform)",
  "`fx.select`/`take` on a Layout/Tensor hard-aborts", "`maxnreg` is dropped silently".
  Note the instruction(s) it lowers to (from `21_final_isa.s`) as evidence.
- **Performance levers** — wins AND regressions, with device-fair TF deltas + the
  structural confirmation (ATT/PMC counter that moved, VGPR/occupancy).
- **Dead-ends** — measured + ruled out, so the loop never retries them.
- **Open ideas** — untried hypotheses worth a lever attempt.

## Measurement discipline (don't record a number that violates this)
- **Device-fair only:** `tests/kernels/bench_fmha_fair.py <module> <seqs...>` (CUDA-graph
  replay). `do_bench`/`bench_fmha_compare.py` distorts small shapes (~0.3 ms host overhead).
- **Correctness first:** `tests/kernels/ck_check.py <module> 1 <sq> <sq> 1 8 1 16` must
  print `OK` (err < 6e-2) before any timing counts.
- **Cross-file deltas < ~20% are inconclusive** on the stopwatch alone (per-module
  artifact) — confirm with a within-file toggle or ISA/PMC structural evidence.
- **Codegen evidence:** `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/x FLYDSL_RUNTIME_ENABLE_CACHE=0`
  → `/tmp/x/<k>_0/21_final_isa.s` (instruction mix) + `19_gpu_module_to_binary.mlir`
  (`vgpr_count`, `group_segment_fixed_size`). Occupancy = `512 // vgpr` waves/SIMD.
- ATT capture + hotspot analysis: see the `capture-kernel-trace` / `kernel-trace-analysis`
  skills.

## GPU policy
- Multiple GPUs may be used in parallel for experiments when free.
- **Avoid GPU 2** — a colleague sometimes uses it. Prefer GPUs 0, 1, 3, 4, 5, 6, 7.
- Check free: `rocm-smi --showpids` (no KFD pid) + `rocm-smi --showmeminfo vram`
  (idle ≈ 298 MB). Then `HIP_VISIBLE_DEVICES=<free id>`.

## Current focus: SMALL shapes (sq1024 / sq2048)
The large shapes (sq16384/32768) already meet/beat CK-Tile; the remaining gap is at
**sq1024/2048** (currently ~70–87% of CK). Prioritize levers that cut per-tile / launch
overhead and softmax VALU at short sequence lengths. The **scale-fold trick** (fold
`scaleQ·scaleK` into the MFMA's native `scale·(A@B)+C`, and the Schraudolph exp bias into
the same path) is a priority small-shape idea — see Open Ideas in FINDINGS.md.
