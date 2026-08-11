# FlyDSL Findings — MI308X / gfx942

Append-only. Newest entries at the top of each section. One–two lines each, dated,
with measured evidence. Read before starting; update at the end of every task.

Hardware: 8× MI308X (gfx942/CDNA3), 80 CU, 512 VGPR/SIMD (combined arch+accum),
64 KB LDS, ~5.3 TB/s HBM, ~1.3 PFLOPS fp8. FlyDSL = wheel 0.2.2 at `/opt/venv`
(Python 3.14). No C++/backend rebuild available from the wheel.

---

## 1k–8k CK-gap study: profiling + CK-Tile source read — 2026-07-15

Focused study of why hk6 trails CK-Tile at seqlen 1024–8192 (bs1 nq8 nk1 causal fp8).
Device-fair hk6+dispatch vs CK: 28/30 (93%), 59/62 (95%), ~120/132 @8k (91%); the
RELATIVE gap is WORST at 3k–8k, not at the tiny shapes.

**hk6 profile (rocprofv3 PMC + ATT, this session):**
- VALU:MFMA ratio ≈ **12–14:1 at every shape** — softmax VALU is the dominant instruction
  class; the two MFMA bursts + the softmax between them run SERIALLY (no independent MFMA to
  hide the VALU). Confirmed #1 ATT stall @sq4096 = L547 GEMM2 PV-MFMA (11%, stalling on V LDS
  reads), #2 = L569 per-tile `s_barrier` (11%). s_waitcnt+barrier = ~31% of all stall.
- Residual **LDS bank conflicts ≈ 0.40/LDS-op** despite +8B padding; LDS-wait/busy climbs
  0.02→0.11 from sq1024→8192.
- ISA: 20× ds_bpermute (P-transpose) + fp8 repack (cvt_pk_fp8/perm/bfe) is a real gfx942 cost.

**CK-Tile production path** (aiter `BlockFmhaBatchPrefillPipelineQRKSVSAsync`, hd128 fp8):
128×128×32 tile, **4 warps / 256 thr, kBlockPerCu=2 (`__launch_bounds__(256,2)`)**, K via
**async buffer_load→LDS in a 3-deep shared K/V ring** with a static LdsSeq rotation, **vmcnt
staircasing** (partial `s_waitcnt vmcnt(N)`, full drain only at GEMM0 tail), softmax bubble
filled with **V load/store (memory, not a 2nd MFMA)**, causal tile-range skip + IsEdgeTile,
**vectorized-V (NO P-transpose)** via a transposed-C warp distribution. Full report in the
subagent transcript.

**Verdict — the 1k–8k gap is dominated by WHEEL-LOCKED levers (matches prior conclusion):**
1. 4-wave + 2-CTA/CU occupancy — UNREACHABLE (hk6 VGPR-bound at 3 waves; maxnreg broken). #1 cause.
2. K async global→LDS DMA ring — DMA broken in wheel + incompatible with our padding.
3. kN0=128 outer tile (4× fewer barriers/softmax-phases than KT=32) — LDS-BLOCKED: KT=128 ×
   NBUF=2 = exactly 64KB even at ZERO padding, so it can't fit with double-buffering.
CK levers hk6 ALREADY has: column-V, exp2, causal tile-skip+loop-split (L361–380), diagonal-pair,
O-rescale-skip, prefetch-relocate, svalu fold.
Portable + not-yet-in-hk6 shortlist: vmcnt staircasing (BUT vmcnt is LLVM-injected, not
Python-controllable — see codegen note), GEMM2 LDS-read pipeline (open idea), shared K/V LDS
ring (big rewrite, mostly pays via the async DMA we can't do). The big levers (4-wave occupancy,
async DMA ring, kN0=128) are wheel/LDS-locked.

**IMPLEMENTED — GEMM2 V-LDS-read hoist (`FMHA_G2HOIST`, hk6):** hoist the GEMM2 V LDS reads to
overlap the softmax exp2/rescale VALU (V packs depend only on vbuf, not P), so the PV MFMA (top
ATT stall @sq4k) stops waiting on the V-read lgkmcnt. Measured device-fair within-file (toggle
on same module): **sq2048 59->60, sq4096 88->90 (+2.3%), sq8192 119->121 (+1.7%), sq16384
143->144** — the FIRST new win in the mid zone. BUT **sq32768 158->143 (-9.5%)** and neutral at
sq1024: the +9 VGPR (158->167, still 3 waves) tips the sq32768 DIAG1 diagonal-pair budget. So
wired into `get_kernel` for **2048 <= sq <= 16384 only** (off at 32768/1024). Correctness OK
(err 0.043-0.055). Only in `ck_hk6.py`; `layoutmax_hk6.py` not yet ported. Small but real — the
mid-zone gap to CK is otherwise the wheel-locked occupancy/DMA ceiling.

---

## Compiler codegen quality (FlyDSL → MLIR → LLVM)

- **2026-07-08** **Scale-MFMA is CDNA4/gfx950-ONLY.** The only scaled-MFMA intrinsics in the
  wheel are `mfma_scale_f32_16x16x128_f8f6f4` / `32x32x64_f8f6f4` (`expr/rocdl/cdna4.py`,
  "CDNA4 scaled MFMA atom", f8f6f4 shapes). gfx942's fp8 MFMA `mfma_f32_32x32x16_fp8_fp8`
  has **no scale operand**. ⇒ the "fold scaleQ·scaleK into the MFMA's `scale·(A@B)+C`" trick
  is **not available on MI308X**; the only gfx942-achievable part (fold the descale into the
  `exp2` bias, Schraudolph) is already done by svalu/hk6 (`exp2(fma(qs,u,-safe_m_p))`). The
  per-kv K-descale mul (`sv*kdv`) is irreducible on gfx942 without the scale-MFMA.
- **2026-07-08** `//` and `%` by a compile-time **power of two are ALREADY lowered to
  shift/mask** — an explicit `>>`/`&` rewrite is a no-op (`v_rcp_iflag_f32`/`v_mul_hi_u32`
  count unchanged; only address-CSE removed ~33 static instrs). Don't bother hand-lowering
  pow2 div/mod.
- **2026-07-08** `fx.math.fma` → a single `v_fma_f32` (verified; the score-VALU fold relies
  on it). Folding `mul + sub` feeding `exp2` into one `fma` really removes the mul.
- **2026-07-08** `fx.rocdl.ballot(i64, pred)` → SGPR result = usable as a **wave-uniform**
  branch condition with no VGPR cost (basis of the O-rescale-skip win).
- **2026-07-08** `_wait_lds()` already emits **lgkmcnt-only** (`s_waitcnt 0xC07F`); the
  `vmcnt(N)` seen at the P-transpose is injected downstream by LLVM `SIInsertWaitcnts` and
  **cannot be decoupled from Python** (re-emitting lgkmcnt-only is a guaranteed no-op).
  `_wait_vmem()` = `s_waitcnt 0x3F70` (vmcnt(0)).
- **2026-07-08** `maxnreg` / waves-per-eu occupancy cap is **BROKEN in the wheel** — lowered
  to `--amdgpu-num-vgpr`/`--amdgpu-waves-per-eu` which don't exist in this LLVM and are
  silently dropped; VGPR never moves. Occupancy can only be changed by shedding real VGPR.
- **2026-07-08** `fx.select` / `fx.take` on a **Layout or Tensor** operand HARD-ABORT the
  process (SIGABRT, `IntTupleUtils.h:729 intTupleSelect expects a non-leaf tuple`) — a
  user-reachable crash, not a clean error. Only pure **IntTuple** operands are safe. In
  kernels, extract a layout mode with `fx.slice(L, (None, i))`, not `fx.select`. (Note:
  `efficient-gemm.py` uses `fx.select(frag, [0,2,1])` on tensors — works only if the
  installed binary matches that source tree; the 0.2.x wheel aborts.)
- **2026-07-08** MFMA is **vgpr-form** here (accumulators in the arch VGPR file, `accum=0`);
  occupancy = `512 / vgpr_count`. `code.json` alone can't show accum/LDS/SGPR — read
  `out_kernel_trace.csv`.

## Grid decode via idx2crd — 2026-07-13 (a layout-algebra WIN)

The block-id -> (qhead, first_idx, batch) grid decode IS a coordinate<->index mapping = a Layout,
so `idx2crd(blk, make_layout(shape, strides))` replaces the hand div/mod. Verified: `idx2crd(lblk,
(nq,num_first,B):(1,nq,nq*num_first))` == the manual `(lblk%nq, (lblk//nq)%num_first, ...)` for all
values (probe); batch extent B = grid_dim.x/(nq*num_first). Both decode branches converted in
`fmha_prefill_fp8_layoutmax_hk6`, correct (err 0.039-0.047), **zero perf change** (idx2crd lowers to
the same div/mod). This is the one place BEYOND the MMA where the algebra genuinely fits — it's pure
coordinate mapping, exactly what layouts are.
- **XCD remap composition FAILED in this wheel:** `crd2idx(idx2crd(blk, L_in), L_out)` (the natural
  layout-composition form of the chiplet permutation) raised `ValueError: expected ArithValue, got
  IntTuple` — crd2idx does not accept an idx2crd IntTuple result directly. Reverted that core to
  manual index math (it's an optional perf lever). Codegen finding: idx2crd->crd2idx composition
  isn't chainable via the Python API here.

## How far layout algebra actually goes in FMHA — 2026-07-08

The "layoutmax" lineage's "maximally layout-algebraic" claim is overstated: the algebra expresses
ONLY the two matmuls (`make_mma_atom`→`make_tiled_mma`→`thr_slice`→`make_fragment_A/B/C`→`fx.gemm`),
and even there the operand DATA is hand-loaded and `.store()`-d into fragments (the algebra gives
fragment SHAPES + drives the matmul). ~82 addressing sites vs ~5 gemm calls. The DATA MOVEMENT is
irreducibly manual: paged KV gather (page-table indirection = data-dependent, not affine), custom
padded/col-V LDS, `ds_bpermute` P-transpose, transposed epilogue scatter — none fit the affine
tiled-copy/`partition_S/D` recipe (which needs dense statically-strided tiles, e.g.
`examples/efficient-gemm.py`).
- **Coord-tensor predication PROBE (works in isolation):** `thr_mma.partition_C(make_view(
  make_coord(0,0), make_identity_layout((32,32))))` gives each lane its C-fragment `(kv,q)` coords;
  `coord[i]` mode 0 == the hand-derived `half*4 + (i//4)*8 + (i%4)` EXACTLY (verified for lanes
  0/1/32/33). So the algebra CAN reproduce the MFMA lane index math.
- **BUT integrating it into the causal mask FAILED** (err 2.4-2.98): `coord[i]` from
  `partition_C(identity)` does not line up element-for-element with `make_fragment_C(g_c).load()[i]`
  (g_c is a strided (32,32):(32,1) view; the coord tensor uses basis strides) — a fragment-ordering
  mismatch. Reverted. Lesson: partition_C coords match the atom's C layout, but you must build the
  score fragment from the SAME coord/identity donor (not a separate strided view) for the element
  order to agree. Left as an open follow-up; the working kernel keeps the hand-indexed mask.

## Layout-algebra artifact — 2026-07-08

`fmha_prefill_fp8_layoutmax_hk6.py` = the fastest kernel (hk6) expressed MAXIMALLY through the
layout algebra: both GEMMs via `make_mma_atom`→`make_tiled_mma`→`thr_slice`→`make_fragment_A/B/C`
→`fx.gemm` (no raw mfma / i64 packing), = `fmha_prefill_fp8_layoutmax` (layout-algebraic hk5) +
hk6's 3 levers (svalu exp2-FMA fold, prefetch relocate, O-rescale-skip ballot) transplanted onto
the softmax/loop (which is byte-identical between hk5/layoutmax/hk6, so the levers port verbatim).
Result: **28/60/120/143/160 TF** @ sq 1024/2048/8192/16384/32768 — MATCHES hk6 (28/59/120/142/160),
marginally better at 2048/16384. Correctness identical (err 0.039-0.047). Proof the layout-algebra
MMA path costs ~0 vs hand-mfma here; the softmax levers are orthogonal to the matmul expression.
What stays manual (no layout-algebra expression): paged KV gather, fp8 cvt_pk, ds_bpermute
P-transpose, per-kv descale, transposed epilogue scatter.

## Performance levers — WINS (device-fair, graph-replay)

- **2026-07-08 Per-seqlen (KT, DIAG) dispatch** (small-shape win, env-knob only — no kernel
  edit). Bigger KT = fewer kv tiles = less per-tile LDS-wait/barrier/store, which dominates
  at short seq (the occupancy cost that killed KT>32 at large shapes barely matters when
  few waves run). Measured optimum for hk6: sq1024 KT64/DIAG0 **21→28** (+33%), sq2048
  KT64/DIAG1 **54→59** (+9%), sq16384 KT32/DIAG0 **140→143** (beats CK 141), sq32768
  KT32/DIAG1 160 (default). Wired as `hk6.get_kernel(sq)`. DIAG (diagonal-pair causal
  load-balancer) helps everywhere EXCEPT sq≤1024 (grid-halving loss). All gated OK.
  KT=128 fails to build (LDS/assert) — KT=64 is the max useful.
- **2026-07-08 O-rescale skip** (BIG). Wave-uniform ballot on `m_new > m_run` skips the
  64-mul online `o_acc *= corr` on tiles where the causal max didn't grow (`corr==1` →
  exact). hk5→hk6 large-shape jump. Best single lever found. Data-dependent branch but
  stable across runs; helps most where the row-max stabilizes (long causal prefill).
- **2026-07-08 Prefetch relocate**. Issue the next-tile K/V prefetch (`load_kv_regs`) AFTER
  GEMM1 (overlapping softmax) instead of at loop top → P-transpose stops stalling on
  prefetch drain. ATT VMEM-wait 24.3M→13.7M cyc; +5% @ sq16384. VGPR-neutral, no carried
  state (return the regs, don't `yield` them).
- **2026-07-08 Score-VALU fold**. Apply per-lane `qs` once via `exp2(fma(qs,u,-safe_m_p))`
  instead of a per-element descale mul; −30 `v_mul_f32`/tile, VGPR-neutral, ~+1%.
- These stack into **hk6** (`kernels/fmha_prefill_fp8_ck_hk6.py`): 21/54/140/160 TF @
  sq 1024/2048/16384/32768 vs hk5 19/48/124/143 and CK 30/62/141/146 (beats CK@32768,
  ties@16384). Commit `810b5c12`.

## Dead-ends (measured + ruled out — do NOT retry)

- **2026-07-08 half-O two-pass** (occupancy attempt): halving the carried O accumulator only
  moved peak VGPR 165→159 (RA peak is the **softmax temporaries**, not O), so no 4th wave,
  and it pays 2× GEMM1/softmax → catastrophic regress (73/84 vs 124/143). To reach 4 waves
  you must shrink softmax temps AND O together — and even then 2× work likely won't pay.
- **2026-07-08 softmax-temp streaming** alone: VGPR unchanged (167), flat perf. Compiler
  keeps values live; single-lever streaming doesn't reach the 128-VGPR/4-wave threshold.
- **2026-07-08 pow2 address rewrite** (`pow2addr`): no-op — compiler already does it (above).
- **2026-07-08 barrier hoist** (`store_kv_to_lds` before compute): regressed 124→118.
- **2026-07-08 s_setprio pipeline** (`pipelite`, finer prio bracketing): regressed 124→120;
  s_setprio can't create an independent in-wave MFMA.
- **2026-07-08 full cross-tile pipeline** (`pipedscp`, triple-buffer): +73 VGPR → 2 waves →
  regress at large (112/126). Wins at small only because there wave parallelism is unused.
- Prior (pre-session, from autoresearch skill): XOR swizzle (+27 VGPR), split-K (wash),
  8-wave (regress), KT>32 (occupancy regress), async K DMA (broken err 2.48), wider O store.

## SMALL-shape profile (hk6, sq2048 / sq1024) — 2026-07-08

Different wall than large shapes. Stall breakdown (sq2048 / sq1024):
- **LDS/SMEM-wait 30% / 27%** — DOMINANT. Top line hk6 L547 (GEMM2 MFMA, stall_rate 88%)
  stalling on the V LDS reads (L538) + P from ds_bpermute (the "LDS read right before PV
  MFMA" pattern). L524 P-transpose `_wait_lds` adds more.
- **VMEM-store 15% / 23%** — the O epilogue `buffer_store` (L615), stall_rate 98%. Few kv
  tiles at short seq ⇒ the one-time output store is a large exposed fraction.
- **barrier 9-12%** (L569 per-tile), **other/VALU 17-19%**, MFMA 8-11%.
Takeaway: small-shape gap to CK (21/54 vs 30/62) is per-tile LDS/store/barrier overhead,
NOT softmax VALU. Levers must cut GEMM2 LDS-wait and/or the store/barrier exposure.

## SMALL-shape parallelism analysis — 2026-07-08 (why "way above CK" is hard)

- **Block-starvation confirmed:** sq1024 AND sq2048 launch only **64 workgroups on 80 CUs**
  (bs1, nq8: 8 heads × 8 q-tiles), each occupied CU running 1 wg = **1 wave/SIMD** (VGPR
  allows 3). GPU grossly under-utilized. At 28 TF (sq1024) we are ~47× off fp8 peak ⇒
  pure latency/overhead, NOT compute — so headroom exists in principle.
- **BUT the fixes all regress** (measured): the per-block fixed latency chain (prologue
  load → LDS stage → kv loop → O store), not parallelism, is the floor.
  - **NWAVES down** (smaller q-tile = more blocks): sq1024 28→21→12 (NW 4→2→1),
    sq2048 59→33. More/smaller blocks lose — per-block overhead doesn't shrink. NW=4 best.
  - **split-KV / Flash-Decoding**: the `splitkv` branch measured 20/43 vs 26/55 —
    regress. The combine kernel's global partial-O+LSE round-trip exceeds the parallelism
    gain when the main kernel is only ~0.08 ms.
- **GQA K/V redundancy is real but blocked by the occupancy tension:** `kvhead=qhead//gqa`
  ⇒ with nk=1/gqa=8 all 8 query-head blocks load+LDS-stage the SAME K/V (8× redundant —
  a real cause of the LDS-wait 30%). Grouping heads to reuse K/V = 8× FEWER blocks (64→8),
  which the NWAVES result says regresses. CK resolves this with head-grouping + careful
  register/LDS distribution — a major kernel rewrite, and the realistic ceiling is ~parity
  (the regime is overhead-bound near a shared floor), not "way above".

## Small-case ATT attribution (sq512/256) — 2026-07-08 (head-grouping REFUTED)

Profiled hk6 at even-smaller cases before attempting a head-grouped rewrite. Verdict:
head-grouping (share K/V across the 8 GQA query heads) will NOT pay.
- As seq shrinks, the **O output store (L615) dominates and GROWS**: VMEM-store =
  15% / 23% / **28%** at sq2048 / sq1024 / sq512 (98% stall-rate, #1 at sq512). It is the
  last op with nothing to hide behind + no sibling wave (starvation) → a hard floor.
- The store, plus per-head GEMM1/softmax/GEMM2 and the per-head **Q** load, are all
  **per-head — NOT reduced by K/V head-grouping**. The head-groupable part (redundant K/V
  global load + LDS stage) is only ~2–13% and SHRINKS as shapes shrink.
- Efficiency craters with block count: sq512 10 TF (32 blocks), sq256 3 TF (16 blocks);
  kernel time floors ~0.04–0.05 ms regardless = fixed per-block latency floor.
⇒ "Way above CK" at small shapes is not reachable by head-grouping (or the earlier
split-KV / smaller-block levers). The regime is O-store + overhead bound at a floor CK
also hits; realistic target is parity (already at 93–95%). Not built — profiling first
saved the rewrite.

## Small-shape structural rewrites — both MEASURED DEAD (2026-07-08)

Tried both non-tuning directions (worktree fan-out) after profiling; both refuted:
- **Persistent grid** (`persist`, cap the launch + grid-stride loop over work items to
  overlap item i's O-store with item i+1's GEMM): control (cap off) = hk6 (28/60); forcing
  multi-item wgs by capping BELOW the item count monotonically REGRESSES (sq1024 28→14→11→6
  @ cap 24/16/8; sq2048 60→32→16). Root cause: item count (32/64) is already < 80 CUs, so
  capping only REMOVES CU coverage — no overlap gain possible when the GPU isn't even full.
  Correct + VGPR-neutral, just no headroom to exploit.
- **Head-grouping** (`hgroup`, HG query heads/wg share K/V LDS, kv-outer/head-inner loop):
  HG=2 correct but **VGPR 166→285 → 1 wave/SIMD** (doubled O-accumulator) AND grid halves;
  59→32 TF. The per-head O-accumulator (64 VGPR) dominates, so carrying 2 heads craters
  occupancy — the ~2-13% K/V-reuse saving is dwarfed. HG≥4 would spill (o_acc alone 4×64=256).
Net: small shapes are firmly at the O-store + occupancy floor; no schedule beats it here.

## Open ideas (untried — candidate levers)

- **~~Scale-fold trick~~ — RULED OUT on gfx942** (scale-MFMA is gfx950-only; see codegen
  note). Its gfx942-achievable part is already in hk6. Revisit only on MI350/gfx950.
- **GEMM2 LDS-read software-pipeline (small-shape #1).** Prefetch the V LDS reads for dt+1
  while MFMA-ing dt so the PV MFMA (L547) stops stalling on lgkmcnt (30% of small-shape
  stall). VGPR-aware (V packs are small).
- Reduce exposed O-store: overlap the epilogue `buffer_store` (L615) with tail compute, or
  a small-seq dispatch variant with a lighter prologue (few-tile regime).

## sq8192 ATT classification — 2026-07-08 (mid-size = LARGE-shape wall)

Profiled hk6 @ sq8192 (KT32/DIAG0). Stall: **other/VALU 50%**, LDS-wait 15%, MFMA 13%,
barrier 10%, **VMEM-store only 2%**. This is the LARGE-shape wall (softmax VALU), NOT the
small-shape store/occupancy floor (store is negligible at 8k — enough tiles to hide it;
512 blocks = well-occupied). So 8k's sub-CK result (120 vs 132, 91%) is NOT a new
bottleneck: the large-shape levers (VALU fold, prefetch relocate, O-rescale skip) are
already applied, but the O-rescale-skip gain scales with tile count (more stabilized-max
tiles at longer seq), so it pays off more at 16k/32k (>CK) than at 8k. KT sweep confirms
KT32/DIAG0 optimal (KT64 → 89/80, occupancy regress). No new transferable lever — 8k is
bound by the same softmax-VALU ceiling (exp2 HW-fixed, scale-fold gfx950-only, 3-wave cap).

## Runbook cross-reference (mlse OPTIMIZATION_RUNBOOK.md) — 2026-07-08

Checked the general kernel-opt runbook for overlap. It CONFIRMS our bottleneck
classification and dead-ends; almost every attention lever it lists we've already
measured:
- §4.3/§13.3 Attention: split-K/V (measured wash/regress), persistent CTA (regress),
  register pressure from accumulators + softmax state (= our RA-peak finding), store path
  for O (= our #1 small-shape stall), coalesced V/O stores (ours already dwordx4 = max on
  CDNA3), warp-level softmax (ours already is — shuffle_xor within the 64-lane wave).
- §3.4 Sync-bound signals (many barriers, small per-barrier work, "fuse iterations between
  barriers") = our small-shape profile; KT=64 already maximizes barrier-fusion (KT=128
  won't build).
- ★ §3.5 **Launch-bound** signals ("tiny kernel latency", "graph replay improves
  throughput", "kernel time close to launch latency", "fusion or persistent kernels
  help") = EXACTLY the bs=1 small-shape regime. KEY REFRAME: our device-fair (graph-replay)
  bench PAYS launch once at capture, so it CANNOT see launch overhead — that's why
  persistent measured as no-help. In REAL bs=1 serving the small shapes are launch-bound,
  and the lever is **fusion / a persistent megakernel across calls** (a serving/integration
  change, invisible to device-fair timing), not a kernel-internal device-compute change.
No NEW device-fair kernel lever surfaced; the one non-overlapping idea is launch-bound
fusion for wall-clock.

## Environment / tooling notes

- **2026-07-08** CK-Tile (aiter) runs under the py3.14 `/opt/venv` after `pip install
  pybind11 einops` and deleting stale `aiter/aiter/jit/*.so` (JIT rebuilds, ~15 s). Time it
  with **`AITER_LOG_MORE=1`** (clean `cuda.Event`) — the default profiler path
  double-counts via ROCTracer at large shapes (reported 71/72 vs true 141/146).
- ATT decoder: install `rocprof-trace-decoder` 0.1.6 `.so` into `/opt/rocm/lib` (rocprofv3
  1.1.0 has ATT but not the decoder). Capture needs `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` for
  source mapping.
- GPUs: use any FREE GPU in parallel; **avoid GPU 2** (colleague). Prefer 0,1,3–7.

## Measured baselines (device-fair, bs=1 nq8 nk1 causal), TFLOPS

Full head-to-head (device-fair graph-replay for FlyDSL/CK; asm via C++ .co harness):

| seqlen | prev dispatch (log2dom/hk5) | **hk6 + get_kernel(sq)** | CK-Tile fp8 | asm/PyISA* |
|--------|-----------------------------|--------------------------|-------------|-----------|
| 1024   | 26                          | **28**                   | 30          | 33 |
| 2048   | 55                          | **59**                   | 62          | 80 |
| 8192   | 109                         | **120**                  | 132         | 190 |
| 16384  | 129                         | **142** (> CK)           | 141         | 239 |
| 32768  | 143                         | **160** (> CK)           | 145         | 292 |

hk6+dispatch beats the previous dispatch at EVERY seqlen (+8/+7/+10/+10/+12%), **beats
CK-Tile at 16384/32768**, but trails CK at 1024/2048/8192 (93% / 95% / **91%**). sq8192 is
the relative low point vs CK (mid-size: past the small-shape floor, not yet where our
large-shape VALU/O-skip wins dominate). DIAG0 best at 8192 (120 vs DIAG1 109), so
get_kernel's sq<=16384→DIAG0 rule is correct there. *asm numbers use a different
(non-device-fair-comparable) data contract via the asm C++ harness — asm baseline only, NOT
a cross-stack comparison (per bench-discipline Rule 2). Remaining small-shape gap to CK is
per-tile LDS-wait/O-store (structural, few tiles) — see the small-shape analysis above.
