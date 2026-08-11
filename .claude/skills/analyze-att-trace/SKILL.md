---
name: analyze-att-trace
description: Analyze an AMD GPU ATT (Advanced Thread Trace) decode folder to find kernel bottlenecks and write an annotations.json that WaveScope overlays on the trace for visual verification. Use when the user points you at a rocprofv3 --att decode directory (contains code.json, filenames.json, occupancy.json, se*_wv*.json) and wants bottlenecks identified and marked up in the viewer.
---

# Analyze an ATT trace and annotate its bottlenecks

You are analyzing a **rocprofv3 `--att` decode folder** and producing `annotations.json` — a set of
bottleneck findings that the WaveScope viewer overlays on the trace so a developer can **visually
verify** each one. Each annotation carries your explanation *and* an anchor pointing at where in the
trace it lives (instructions, a source line, or a wave + time window).

Your output is a single file, `annotations.json`, written into the trace folder (next to `code.json`).
WaveScope loads it automatically (and validates it — malformed entries are dropped, so follow the
schema exactly).

## 1. Read and understand the trace folder

A decode folder (`ui_output_agent_*_dispatch_*/`) contains:

- **`code.json`** — `{ "code": [ row, ... ] }`. Each row is one static ISA instruction. Columns
  (by position): `[0] ISA text`, `[1] _`, `[2] LineNumber`, `[3] Source` (`"<path>:<line>"`, empty
  if no DWARF), `[4] Codeobj`, `[5] Vaddr`, `[6] Hit` (execution count), `[7] Latency` (avg cycles),
  `[8] Stall` (avg cycles waiting), `[9] Idle`. **The row's array index is its instruction `idx`** —
  this is the stable id you anchor to (`anchor.instIdxs`).
- **`filenames.json`** — maps shader-engine / SM / slot / wave → the per-wave JSON file names.
- **`se*_sm*_sl*_wv*.json`** — one per wave: `{ "wave": { begin, end, instructions, timeline, ... } }`.
  - `wave.instructions`: array of `[time, type, stall, latency, instIdx]` — the *dynamic* execution
    (a static instruction can appear many times, e.g. a loop body). `time` is a cycle offset;
    `instIdx` links back to `code.json`.
  - `wave.timeline`: a `[state, duration]` stream accumulated from `wave.begin`. `state % 5` →
    `0 EMPTY, 1 IDLE, 2 EXEC, 3 WAIT, 4 STALL`. This is the authoritative per-wave state breakdown —
    a wave dominated by WAIT/STALL is stalling; use it to quantify.
  - `wave.begin` / `wave.end`: the wave's cycle span. Use `[t0, t1]` inside this range for a
    `wave`+`time` anchor.
- **`occupancy.json`** — `{ dispatch_id: [[time, se, simd, sl, wave_slot, active], ...] }`; `active`
  is the live-wave count. Low `active` over time = an occupancy bottleneck.

Instruction categories (from the ISA mnemonic): `s_load/s_store` = SMEM; `buffer_/global_/flat_load`
= VMEM load; `*_store` = STORE; `ds_*` = LDS; `v_mfma*/v_wmma*` = matrix (MFMA); `s_waitcnt` =
the wait that blocks on outstanding memory; `v_exp/log/rcp/rsq` = transcendentals (softmax);
`v_permlane*/readlane` = cross-lane.

## 2. How to reason about bottlenecks

Look for, and quantify with the data:

- **Memory-latency waits** — `s_waitcnt` instructions with high `Stall` (col 8) × `Hit` (col 6).
  The loads they wait on are the deps; anchor to the waitcnt + the loads. (WaveScope's own rules do
  this too — your value is explaining *which* access pattern and *why*.)
- **MFMA feed stalls** — matrix instructions preceded by long waits/idle; the matrix engine starved
  for operands. Look for `v_mfma*` with high stall or big idle gaps before them.
- **Narrow / strided loads** — many single-dword `buffer_load_dword` where a wider load would do.
- **Low occupancy** — `occupancy.json` `active` staying well below the peak; often VGPR/LDS pressure.
- **Cross-lane / softmax stalls** — clusters of transcendentals or `v_permlane*` dominating a stall
  window.
- **Wave imbalance / leading idle** — big IDLE spans at wave start (from `wave.timeline`).

Ground every claim in numbers you can point at (cycle counts, hit counts, % of a wave in STALL).
Prefer a few high-confidence, well-anchored annotations over many vague ones.

## 3. Write `annotations.json`

Write an object `{ "annotations": [ ... ] }` (a bare array is also accepted). Each annotation:

```jsonc
{
  "id": "mem-wait-1",              // stable string id (unique in the file)
  "severity": "critical",         // critical | high | medium | low | info
  "title": "s_waitcnt stalls 51k cy on 3 buffer loads",  // one line
  "note": "The waitcnt at idx 559 blocks the MFMA feed... (your explanation, why it's a bottleneck)",
  "category": "mem-wait",          // optional free-form tag
  "confidence": 0.9,               // optional, 0..1
  "suggestion": "Prefetch K one tile ahead so the load overlaps the prior MFMA.", // optional
  "metric": 51584,                 // optional number, used to sort within a severity
  "anchor": {                      // at least one kind; null/omitted = whole-kernel (list-only)
    "instIdxs": [559, 562, 563, 564],                 // instruction idxs (code.json row index)
    "source": { "file": "attn.py", "line": 459 },     // or a source line (endLine optional for a range)
    "wave": { "se": 0, "sm": 2, "slot": 0 }, "time": [12000, 53000]  // or a wave + cycle window
  }
}
```

Anchor guidance:
- Prefer **`instIdxs`** — it lights the exact blocks in both the ISA list and the timeline. Use the
  `idx` (array position) from `code.json`.
- Use **`source`** when the point is "this line/loop is the problem"; WaveScope resolves it to the
  line's instructions and badges the source column. `file` may be a basename — it matches by basename
  when the full path differs (e.g. a container path). `endLine` makes it a range.
- Use **`wave` + `time`** to box a specific stall window on one wave's timeline lane. Both parts are
  required; pick the wave from `filenames.json` and the `[t0,t1]` from within its `begin..end`.
- Omit `anchor` (or set it `null`) for a global finding like "occupancy is low across the dispatch" —
  it shows in the Annotations list with no overlay.

Rules the validator enforces (so your file loads cleanly):
- `title` is **required** (entries without one are dropped).
- Out-of-range `instIdxs` are filtered; a partial `wave`/`time` anchor (only one of the two) is
  ignored. `confidence` is clamped to 0..1; unknown `severity` becomes `info`. Duplicate `id`s are
  de-duplicated.

## 4. Deliver

Write the file to `<trace-dir>/annotations.json`. In WaveScope it appears as an **Annotations** tab
(next to Bottlenecks); each entry jumps to its location on click, numbered flags appear on the
timeline, wave/time anchors draw a region box, and `n`/`p` walk through them in severity order.

If driving the WaveScope VS Code extension live (no reload), the host can push the same payload via
the `wavescope.setAnnotations` command / exported `setAnnotations(data)` API instead of (or in
addition to) the file.

## Worked example

For a trace where the softmax MFMA feed is starved by K-tile loads, a good annotation:

```json
{
  "annotations": [
    {
      "id": "kfeed-stall",
      "severity": "critical",
      "title": "MFMA feed stalls 51k cy waiting on K buffer loads",
      "note": "s_waitcnt vmcnt(7) at idx 559 gates the first v_mfma of each K-tile. It waits on the three buffer_load_dwordx4 at idx 562-564, which are issued only ~100 cy earlier, so the matrix engine idles ~51k cy across the wave (34% of its STALL time per wave.timeline).",
      "category": "mfma-feed",
      "confidence": 0.9,
      "suggestion": "Software-pipeline the K load one tile ahead so it overlaps the previous tile's MFMAs.",
      "metric": 51584,
      "anchor": { "instIdxs": [559, 562, 563, 564], "source": { "file": "attn.py", "line": 459 } }
    }
  ]
}
```
