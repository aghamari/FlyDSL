---
name: flydsl-kernel-bench-discipline
description: Measurement and experiment methodology for FMHA/GPU-kernel optimization on the FlyDSL MI308X / gfx942 (CDNA3) project — how to time, gate, and isolate experiments so a reported speedup is real and not a benchmarking artifact. Use when running, comparing, or trusting FMHA/kernel perf numbers; when a do_bench/TFLOPS delta looks too good (or too bad) to be true; when comparing FlyDSL vs CK/PyISA; or before believing any "this lever is faster" claim. Distilled from a session where a +3.7/6.2% do_bench "win" was actually a device-fair regression. Companion to flydsl-fmha-prefill-opt (lever catalog) and fmha-prefill-autoresearch (the loop).
argument-hint: [a kernel/variant to measure, a shape, or a claimed speedup to verify]
---

# FlyDSL Kernel Bench Discipline (MI308X / gfx942)

How to **measure FMHA/kernel optimization experiments so the result is trustworthy** on
this project (FlyDSL, AMD **MI308X / gfx942 / CDNA3**, fp8 paged causal FMHA prefill).
This is the *methodology* half of the FMHA work — the **what to optimize** lives in
`flydsl-fmha-prefill-opt` (lever + dead-end catalog) and **the autonomous loop** lives in
`fmha-prefill-autoresearch`. Read this before you believe any speedup.

> **The cautionary tale that produced this skill:** a fast-exp2 variant (`hk_fexp`,
> `FMHA_FEXP=2`) looked like **+3.7 / +6.2%** under `do_bench` (113/137 TF vs a depressed
> 109/129 hk5). Device-fair graph-replay timing showed it was a **small REGRESSION**
> (126/139 vs real hk5 **129/142**). The "win" was host-dispatch noise. Don't ship a win
> the device-fair bench hasn't confirmed.

---

## Rule 1 — Trust device-fair graph-replay timing, not `do_bench`
`do_bench` (`tests/kernels/bench_fmha_compare.py`) times the `@flyc.jit` wrapper, which
includes **~0.3 ms host/JIT dispatch per call**. At small/fast shapes that overhead
*dwarfs* the kernel and **distorts comparisons** (a slower kernel can look faster because
its dispatch noise landed lower that run).

**The trustworthy number:** `tests/kernels/bench_fmha_fair.py` — compiles once,
CUDA-graph-captures, times the **median of 50 graph replays** (host cost paid at capture)
= true device time, comparable to CK-Tile's device-side numbers.

```bash
HIP_VISIBLE_DEVICES=<free> python3 tests/kernels/bench_fmha_fair.py <module> [seqs...]
# -> "<module> sqN: X.XXXXms / YTF (device, graph)"
```

**RULE:** `do_bench` is fine for a quick smell test, but **always confirm a win with the
device-fair bench before believing it.** Treat any `do_bench` delta on a fast shape
(sq1024/2048) as suspect until graph-replay agrees.

---

## Rule 1b — Only WITHIN-FILE comparisons are valid (per-module artifact)
Even the device-fair graph-replay bench (Rule 1) carries an **intrinsic per-module
(per-compiled-artifact) effect**: algorithmically-equivalent kernels living in *different
files* measured **118 / 128 / 141 TF — a ~20% swing — despite byte-identical ISA**, and the
swing survived a 3-second settle warmup. So it is **not** warmup and **not** a real
algorithmic difference; it is a module-level artifact (module load / code-object placement /
allocator state). **A cross-file delta below ~20% is therefore NOT evidence of a real win or
loss** — comparing `hk5.py` vs `hk_fzx.py` directly can swing that much on noise alone.

**RULE:** Make the comparison **within a single compiled artifact.** Put the variants behind
env-toggled compile-time knobs in the SAME file (e.g. `FMHA_FEXP` / `FMHA_FREEZE` toggling
inside `hk_fzx.py`) and compare the toggles (FREEZE=0 vs FREEZE=1) — same module, same
code-object, so the per-module artifact cancels.

- **Worked example:** within-file, freeze on vs off = **118 vs 119 TF = neutral** (the
  trustworthy result). The cross-file read of the "same" kernel in another file showed
  **129 TF** — that +9% is the artifact, not a real win.
- **Method:** **interleaved repeats on a single idle GPU** — alternate the two toggles
  back-to-back, repeated — *not* separate back-to-back batches, so slow drift cancels.
  (Still obey Rule 4: free GPU, small shapes repeated.)
- **When a lever genuinely needs a separate file** (a structural change that can't be
  env-gated), treat any cross-file delta < ~20% as **inconclusive**: find a within-file
  proxy, or confirm via **ISA/PMC structural evidence** (Rule 7 step 4) instead of the
  stopwatch.

---

## Rule 2 — Bench each kernel with its NATIVE harness (never a foreign ABI)
A kernel benched through a harness wired for a *different* kernel's ABI produces
physically implausible numbers — it may *load and launch* yet **mislaunch**.

- **FlyDSL kernels** → `bench_fmha_fair.py` (device-fair) / `bench_fmha_compare.py`.
- **CK-Tile** → aiter path: `bench_fmha_compare.py --ck` (needs aiter built).
- The C++ `.co` harness `tests/fmha/src/bench/fmha_bench.cpp` is **hard-wired to the
  asm/PyISA ABI** (512 threads, grid `gdx = ceil(ceil(sq/ts_qo)/2)`, `tg_div=2`). Use it
  **only for kernels that share that ABI (the asm family).**

**Cautionary tale:** loading CK's `.co`
(`aiter/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_fp8_causal.co`, symbol
`_ZN5aiter25fmha_fwd_hd128_fp8_causalE`) into the asm C++ harness *loaded* but
**mislaunched** under that grid → inflated **172/200 TF** vs CK's true **141/145**.

**RULE:** match harness↔ABI. An implausibly high TF number is the tell that you crossed
ABIs — re-bench with the kernel's native path before recording anything.

### Benching the asm/PyISA kernel without the CPU check
The PyISA host binary `asm/fwd_fp8` **always computes a slow CPU reference** and has **no
flag to disable it** (`check=0`, `no_check=1`, `skip_check=1`, `validate=0`, `verify=0` are
all ignored). To get pure perf, **bypass it**: bench the asm `.co` directly through the C++
harness (`tests/fmha/src/bench/fmha_bench.cpp`), which uses random inputs + `hipEvent`
timing and does **no correctness check**. This is valid here because the asm family **shares
that harness's ABI** (unlike CK — see Rule 2 caveat above; never trust CK through this harness).

```bash
# Build once:
hipcc -O3 --offload-arch=gfx942 -I /workspaces/amir/tests/fmha/src \
  /workspaces/amir/tests/fmha/src/bench/fmha_bench.cpp -o /tmp/fmha_bench

# Run (free GPU; both fwd_causal.co and fwd_causal_sched.co use the SAME symbol):
SYM=_ZN5aiter36fmha_fwd_hd128_fp8_causal_qkptph_vphE
HIP_VISIBLE_DEVICES=<free> /tmp/fmha_bench --co /workspaces/amir/asm/fwd_causal.co --sym $SYM \
  --batch 1 --seqlen-q <sq> --nhead-q 8 --nhead-k 1 --hd-qk 128 --hd-v 128 \
  --paged --page-size 16 --vec-k-col-v --scatter-pages --descale-ptph \
  --ts-qo 256 --warmup 20 --repeat 50
```

Verified kernel symbol (both `.co`, via `llvm-objdump -t <co> | grep .kd`, drop the `.kd`):
`_ZN5aiter36fmha_fwd_hd128_fp8_causal_qkptph_vphE`.

Measured asm TFLOPS (median ms), device, this harness — stable across reruns; both `.co`
within noise of each other:

| seqlen | fwd_causal | fwd_causal_sched |
|---|---|---|
| sq1024  | 33 TF (0.066 ms)  | 33 TF (0.065 ms)  |
| sq2048  | 80 TF (0.108 ms)  | 80 TF (0.108 ms)  |
| sq16384 | 238 TF (2.312 ms) | 238 TF (2.314 ms) |
| sq32768 | 291 TF (7.557 ms) | 291 TF (7.562 ms) |

These are the asm-kernel native-ABI numbers; they are far above the FlyDSL/CK figures in
Rule 6 because the asm family runs a different (non-device-fair-comparable) data contract —
use them only as the asm baseline, not as a cross-stack comparison against FlyDSL/CK.

---

## Rule 3 — One process per (module, seqlen)
Knobs like `FMHA_KT` / `FMHA_DIAG` / `FMHA_FEXP` are read as **import-time constexpr**,
and the FlyDSL module-global `SmemAllocator` **finalizes once per process**. So:

- **Set the env BEFORE importing** the kernel module.
- **Fork a fresh process per config / per seqlen** — never loop shapes or toggle a knob
  inside one Python process.
- Mirror the existing pattern in `kernels/fmha_prefill_fp8_dispatch.py:get_kernel(sq)`
  (sets `FMHA_KT`/`FMHA_DIAG` in `os.environ`, then re-imports). `ck_check.py` and the
  benches already fork per shape — respect that.

---

## Rule 4 — Small shapes need isolated, repeated runs on a free GPU
sq1024 / 2048 **swing under parallel-GPU load** (a real session saw sq2048 read 29.2 then
12.9 TF — a 2.3× swing — across parallel runs; isolated repeats showed no real
difference).

- **Repeat 3× on a free GPU** for small/fast shapes and take the consistent number.
- Parallel sweeps are OK **only** for the large/slow shapes (sq16384 / 32768).
- **Pick a FREE GPU at run time — never hardcode one:**

```bash
rocm-smi --showpids                       # no KFD pids on a GPU = free
rocm-smi --showmeminfo vram               # idle ≈ 298 MB used
HIP_VISIBLE_DEVICES=<that free id> python3 ...
```

---

## Rule 5 — Correctness gate BEFORE speed
A faster-but-failing variant is a **lossy SKU, not a win** (e.g. `FMHA_FEXP=1` was faster
but failed the error gate). Gate fp8 **err < 6e-2**, one shape per process:

```bash
HIP_VISIBLE_DEVICES=<free> python3 tests/kernels/ck_check.py <module> 1 <sq> <sq> 1 8 1 16
#                                              args: module b sq sk nk gqa causal page_size [pscale]
```

If it isn't `OK`, the lever is wrong — fix or discard. **Do not time speed yet.**

---

## Rule 6 — Reference points (device-fair, this session)
Use these as the bar; re-measure on *your* machine before trusting them blindly.

| seqlen | CK-Tile fp8 (target) | FlyDSL dispatch-best | % of CK |
|---|---|---|---|
| sq1024  | 30  | 26  | 87% |
| sq2048  | 62  | 55  | 89% |
| sq16384 | 141 | 129 | 91% |
| sq32768 | 145 | 142 | 98% |

TFLOPS, bs=1 nq8 nk1 causal. FlyDSL dispatch-best = **log2dom for sq≤2048, hk5 for
sq>2048**. **Biggest gap: sq1024 (relative) and ~12 TF at sq16384 (absolute).**

---

## Rule 7 — The general experiment loop
For every lever:

1. **Probe the real bottleneck metric** first (ISA/PMC), so you optimize the *measured*
   bottleneck, not the plausible one.
2. **Change ONE lever** (new variant file or env knob; never stack two unverified).
3. **Correctness gate** (Rule 5) — `OK` before anything else.
4. **Confirm structurally** — did the binding ISA/PMC count actually move? A change that
   doesn't move VGPR/spills, the instruction histogram, or the binding counter did
   nothing, regardless of the stopwatch.
5. **Confirm device-fair** (Rule 1) — `bench_fmha_fair.py`, small shapes isolated+repeated
   (Rule 4).
6. **Keep only if it beats the device-fair PREVIOUS BEST.** Record negatives too, so no
   one re-runs a ruled-out experiment.

> **Never claim a speedup/regression from a single `do_bench` run or from reasoning
> alone.** Two reasons in a row without a measurement = stop and run the cheapest
> falsifier.

---

## Anti-artifact checklist (run through this before recording a number)
```
- [ ] Device-fair (bench_fmha_fair.py graph-replay), not raw do_bench?
- [ ] WITHIN-file toggle comparison (interleaved), not a cross-file delta < ~20%?
- [ ] Each kernel on its NATIVE harness (FlyDSL=fair; CK=--ck; .co C++=asm-ABI only)?
- [ ] Env knob set BEFORE import, one process per (module, seqlen)?
- [ ] Free GPU (showpids/showmeminfo), small shapes repeated 3×?
- [ ] ck_check OK (err < 6e-2) on the touched shape?
- [ ] Binding ISA/PMC counter actually moved (not just the clock)?
- [ ] Beats the device-fair previous best, not a depressed do_bench baseline?
- [ ] Implausibly high TF? -> suspect a foreign-ABI mislaunch, re-bench.
```

---

## Related
- **flydsl-fmha-prefill-opt** — the kernel knowledge: lineage, structural wins, the full
  measured dead-end catalog, harness reference. (*What* to optimize.)
- **fmha-prefill-autoresearch** — the autonomous edit→measure→keep/discard loop that
  *applies* this discipline lever-by-lever. (*How* to drive the loop.)
- KB skills: `kernel-trace-analysis`, `cdna-kernel-opt`, `lds-optimization`,
  `gemm-optimization` — generic CDNA profiling/method this specializes.

## One-sentence takeaway
> Believe a kernel speedup only when the **device-fair graph-replay bench** confirms it on
> the kernel's **native harness**, with **one process per (module, seqlen)**, **small
> shapes isolated+repeated on a free GPU**, **correctness gated first**, and the
> **binding ISA/PMC counter actually moved** — because the +3.7/6.2% do_bench "win" that
> birthed this skill was really a device-fair regression.
