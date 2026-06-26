# research_mfma — how hard is it to change the MFMA instruction?

Grew out of `learn_fmha/lesson_01_single_mfma.py` and `lesson_02_gemm_tiles.py`.
The question:

> *I have a working `mfma_f32_16x16x16` GEMM K-loop. How hard is it to swap in a
> different MFMA instruction, and does it matter for performance? And what does
> the HipKittens "XCD swizzle" buy on top?*

A **deliberately minimal** register-resident GEMM — the same no-LDS K-loop as
lesson_02, on a real grid — where the *only* things that change between variants
are:

1. **the MFMA instruction** — `mfma_f32_16x16x16bf16_1k` vs `mfma_f32_32x32x8bf16_1k`
2. **the block→tile mapping** — plain row-major vs a HipKittens-style XCD swizzle

No LDS, no prefetch, no double-buffering, no scheduling barriers. Every lane
copies its A/B fragment straight from global on every K-step. That makes the
kernel badly HBM-bound *on purpose* — it's the clean control that isolates the
two knobs above.

`C[M,N] = A[M,K] @ B[N,K]^T`, bf16 in, f32 accumulator. One wavefront (64 lanes,
wave layout `(1,1,1)`) computes one MFMA-shaped output tile.

## Written in the idiomatic layout-algebra style

This is the same style as `examples/03-tiledMma.py` / `examples/04-preshuffle_gemm.py`,
**not** the hand-rolled `lesson_01` style:

- `fx.rocdl.make_buffer_tensor` + `fx.flat_divide(...)[None,None,pid,None]` for tiling
- `fx.make_tiled_mma(fx.make_mma_atom(MFMA(...)), wave_layout)` — derives the
  A/B/C **fragment layouts** for you (no hand-written lane↔element math)
- `fx.make_tiled_copy_{A,B,C}` + `fx.copy` to move global↔register
- `frag.load()` / `frag.store()` to touch the register fragment as a vector

→ **no `.ir_value()`, no `buffer_ops`, no manual offset arithmetic.** Compare the
git history of `gemm_mfma.py` to see how much index bookkeeping that removed.

## Files

| file | what |
|------|------|
| `minimal_tiled_gemm.py` | the idiomatic single-tile K-loop, ~60 lines — the cleanest illustration; flip `SHAPE` to swap instruction |
| `gemm_mfma.py` | the 4 grid variants + XCD swizzle + correctness check + benchmark table |

## Run

```bash
HIP_VISIBLE_DEVICES=2 python3 research_mfma/minimal_tiled_gemm.py
HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_mfma.py --M 4096 --N 4096 --K 4096
HIP_VISIBLE_DEVICES=2 python3 research_mfma/gemm_mfma.py --xcds 4 --group-m 8
```

## The honest answer: "how hard is it?"

In idiomatic FlyDSL, swapping the instruction is **almost free**: change
`MFMA(m,n,k,bf16)` and `make_tiled_mma` rebuilds every fragment layout to match
the new shape. You do *not* hand-derive "which lane holds which element" — that
was the hard part in lesson_01, and the layout algebra now does it.

**But there is a real DSL ceiling, and finding it is the lesson** (this is the
HipKittens *"DSL reality check — probe before you build"* applied literally):

- `fx.gemm` lowers through the CDNA3 MMA-atom dispatch
  (`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`). On **gfx942 bf16** that table only
  really reaches `16x16x16`: `32x32x8` has **no dispatch entry**, and `32x32x4`
  isn't even LLVM-selectable on this arch (`Cannot select intrinsic
  llvm.amdgcn.mfma.f32.32x32x4bf16`).
- So the *instruction comparison itself* lives **below** `fx.gemm`. The trick
  that keeps it idiomatic: the **fragment layouts `make_tiled_mma` derives are
  already exactly what the hardware op wants**, so we keep the tiled-copy data
  movement and only drop to the raw intrinsic for the one instruction, calling
  it on the fragment vectors:

  ```python
  a = frag_A.load().bitcast(fx.Int16)
  b = frag_B.load().bitcast(fx.Int16)
  frag_C.store(fx.Vector(opcode(f32xacc, [a, b, frag_C.load()])))
  ```

  For `16x16x16` you *could* instead write
  `fx.gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C)`; we use the raw-opcode
  form for both shapes so the K-loop body is identical and the comparison is
  apples-to-apples.

Two small width gotchas the layout algebra still makes you get right:
- A/B copy width = `(m*k/64)*16` bits (the bf16-per-lane of the fragment).
- C copy must be **`BufferCopy32b`**: the C fragment's f32s are *strided across
  rows*, so a wider coalesced store writes them to the wrong addresses (a 128-bit
  C copy silently produces garbage — found the hard way).

## Results

Measured on **MI308X (gfx942, 80 CU)**. bf16 in / f32 acc. `do_bench` median.
`GB/s(min)` is the *lower-bound* I/O (A,B read once); real HBM traffic is this ×
the reread factor, which is why the kernel is HBM-bound.

**M=N=K=4096** (`--xcds 4 --group-m 8`):

| variant | tile | µs | TFLOPS | vs baseline |
|---------|------|----|--------|-------------|
| `16x16x16`       | 16×16 | 20273 |  6.78 | 1.00× |
| `32x32x8`        | 32×32 | 12500 | 10.99 | **1.62×** |
| `16x16x16 + xcd` | 16×16 | 19966 |  6.88 | 1.04× |
| `32x32x8 + xcd`  | 32×32 | 11259 | 12.21 | **1.80×** |

Stable across sizes (2048/4096/8192 all give baseline ≈6.1–6.8 TFLOPS,
`32x32x8` ≈1.5–1.8×).

### Reading the numbers

- **Single-digit TFLOPS vs ~1300 peak is the point, not a bug.** With no LDS and
  no reuse, each output tile re-reads its A rows / B columns from global every
  K-step, so the kernel is pinned at HBM bandwidth. MFMA *compute* throughput is
  irrelevant — what matters is **how much global traffic each instruction
  amortizes**.

- **The MFMA shape is the dominant lever (+62–91%).** `32x32x8` spreads each
  loaded A/B fragment over a 32×32 = 1024-element output tile instead of 16×16 =
  256 (≈4× more MACs per loaded byte → ≈half the HBM traffic per flop). Swapping
  the instruction changed the *arithmetic intensity*, which is the whole game in
  a bandwidth-bound kernel.

- **The XCD swizzle is a real but second-order win (a few %), biggest on the
  already-better `32x32x8`.** Isolating the pieces at 4096:

  | swizzle | `16x16x16` | `32x32x8` |
  |---------|-----------|-----------|
  | none (row-major)                | 6.56 | 11.64 |
  | windowed only (xcds=1, group=8) | 6.68 | 12.01 |
  | + XCD remap (xcds=4)            | 6.84 | 12.50 |
  | + XCD remap (xcds=8)            | 6.84 | 12.29 |

  Most of the benefit is the **windowed traversal** keeping a reused B column hot
  in L2; the **XCD remap** adds a little by landing consecutive blocks on the
  same XCD's private L2. `NUM_XCDS=4` edges out 8, matching the in-repo
  `efficient-gemm.py` comment that the 308 behaves as 4 XCDs. This is exactly
  HipKittens fact #5: on a low-/single-XCD part like MI308X the chiplet swizzle
  is **second-order — the structural lever (output-tile/instruction) dominates.**

## Caveats

- This is **not** a competitive GEMM — it's a teaching control. A real kernel
  stages A/B through LDS so each tile is read from global once, worth far more
  than either knob here (see `examples/efficient-gemm.py`).
- The two variants don't do identical work per output, so this measures *kernel
  throughput*, not raw instruction throughput. That's the intended question:
  "if I swap the instruction in this fixed skeleton, what happens?"
- bf16 only. `gfx942` also has `mfma_f32_16x16x32_fp8_fp8` / `mfma_i32_16x16x32_i8`;
  adding those needs quantization + a different reference, left as an extension.

## Possible next steps

- Stage A/B through LDS (lesson_10) and watch both knobs shrink to noise as the
  kernel stops being HBM-bound.
- Add the fp8 `16x16x32` instruction as a 5th variant.
- Sweep `--group-m` / `--xcds` to map the L2-reuse sweet spot.
