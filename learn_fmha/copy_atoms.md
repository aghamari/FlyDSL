# Copy Atoms in FlyDSL — variants, parameters, and real usage

A companion reference for the `learn_fmha` lessons. A **copy atom** is the *how* of
data movement: it describes the hardware instruction that moves a small fixed block
of bytes between two memory spaces (global ↔ register ↔ LDS), independent of *which*
element each thread owns (that's the layout's job — see `divide_playground.py`).

You always use a copy atom in two steps:

```python
atom = fx.make_copy_atom(<copy_op>, <elem_type>)   # build the atom (the instruction)
fx.copy_atom_call(atom, src, dst, pred=None)        # execute: move src -> dst
```

`src`/`dst` are **tensors** (a global slice, an rmem/smem fragment), not scalars. To
read/write the actual numbers in a register tensor you still use
`fx.memref_load_vec` / `fx.memref_store_vec` (see lesson 00).

---

## 1. `make_copy_atom(copy_op, elem_type)` — the two widths

This is the single most important thing to understand. A copy atom has **two**
independent widths:

| Parameter | Set by | Meaning |
|-----------|--------|---------|
| **transaction width** | the `copy_op` (e.g. `BufferCopy128b`) | total bits moved per atom call |
| **element width** (`val_bits`) | the `elem_type` arg | bits per logical element |

The number of elements moved per call is `transaction_width / element_width`.

`elem_type` accepts a Numeric type (`fx.Float32`), an `ir.Type`, or a raw `int`
(interpreted directly as `val_bits`). Source: `python/flydsl/expr/primitive.py:890`.

```python
# 1 f32 per call (lesson 00): 32-bit transaction / 32-bit element = 1 elem
fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

# 4 f32 per call (lesson 09 "wide loads"): 128 / 32 = 4 elems, one dwordx4
fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# 8 bf16 per call (qk_norm_rope_quant.py): 128 / 16 = 8 elems
fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)

# 4 bf16 per call (the rope half): 64 / 16 = 4 elems
fx.make_copy_atom(fx.rocdl.BufferCopy(64), 16)
```

> **Max transaction width is 128 bits** on CDNA3 (one `dwordx4`). That's why
> `ELEMS_PER_THREAD = 4` for f32 in `lesson_09_wide_loads.py` — 4×f32 = 128 bits =
> exactly one wide load.

---

## 2. The atom families

There are two axes: **who issues the access** (Universal vs Buffer) and **what it
does** (plain copy, LDS copy, atomic).

### 2a. `UniversalCopy{8,16,32,64,128}b` — portable load/store

Plain pointer-based load/store. Works on any address space, no bounds checking.
Use when the access is always in-bounds (e.g. LDS↔register, or a register stage you
control). Defined in `python/flydsl/expr/primitive.py:191`.

```python
cp = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
cp = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)   # 4 f32
```

### 2b. `BufferCopy{8,16,32,64,128}b` — CDNA3 buffer load/store (OOB-checked)

AMD **buffer-resource** load/store. The descriptor carries a `num_records` byte
range, so out-of-bounds lanes are hardware-masked (reads return 0, writes dropped) —
this is the idiomatic way to handle ragged/boundary tiles on AMD. Requires the source
to be wrapped with `fx.rocdl.make_buffer_tensor` first (lesson 00, step 1).
Defined in `python/flydsl/expr/rocdl/universal.py:19`.

```python
cp = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
cp = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
cp = fx.make_copy_atom(fx.rocdl.BufferCopy(ATOM_BITS), elem_bits)   # parametric
```

Atom state (settable via `.set_value`/`fx.copy(..., **kwargs)`): `soffset` (i32
scalar byte offset, default 0).

**Universal vs Buffer rule of thumb:** global memory that might be partial → Buffer
(free OOB masking). LDS or known-in-bounds register staging → Universal.

### 2c. `BufferCopyLDS{32,64,128}b` — direct global → LDS DMA

Streams global memory straight into LDS **without** staging through registers (the
CDNA `buffer_load ... lds` path). Only supports `BufferDesc -> Shared` direction.
Defined in `python/flydsl/expr/rocdl/universal.py:36`. In practice the FMHA kernels
issue it through the lower-level helper `fx.rocdl.buffer_load_to_lds(...)` (see
`kernels/fmha_prefill_fp8_*.py`, the K-prefetch path) plus a `s_waitcnt vmcnt(0)` to
wait on the DMA. Atom state: `soffset`, `imm_offset` (both i32, default 0).

### 2d. Atomic atoms — `BufferAtomic*` and `UniversalAtomic*`

#### What "atomic" means

A normal copy *store* is a plain write: `dst = src`. An **atomic** copy turns that
store into a **read-modify-write (RMW)** that the hardware guarantees happens as one
indivisible step:

```
read old = dst
new = old <op> src      # op = Add, Max, Min, ...
write dst = new
```

The "atomic" guarantee is that no other thread (anywhere on the GPU) can observe or
interleave with the middle of that sequence. If 64 lanes all do an atomic-add to the
same address, you are guaranteed to get the sum of all 64 contributions — none get
lost. With a plain store you'd get a **race**: every lane reads the same old value,
adds its bit, and writes back, so all but one update is silently overwritten ("lost
update"). Atomics are the fix for that race.

> Trade-off: atomics serialize concurrent updates to the *same* address, so they are
> slower than a plain store when there is contention. Use them only when multiple
> threads/blocks genuinely target the same location. Updates to *distinct* addresses
> run in parallel and are cheap.

#### When you actually need them

- **Split-K / reduction GEMM**: several blocks each compute a partial sum for the same
  output tile and `AtomicAdd` their partials into one global accumulator.
- **Histograms / counters / scatter**: many lanes bump the same bin (`AtomicAdd`, or
  `Inc`).
- **Global reductions** (a single max/sum across the whole grid): `AtomicMax` /
  `AtomicAdd` into one scalar.
- **MoE / gather-scatter** where output rows from different experts land in the same
  destination.

If each thread owns a unique output element (the lesson-00 vector-add case), you do
**not** need an atomic — a plain `BufferCopy*`/`UniversalCopy*` store is correct and
faster.

#### What each op does

`new = old <op> src`, applied atomically:

| Op | Effect (`new = ...`) | Typical use |
|----|----------------------|-------------|
| `Add` | `old + src` | sums, split-K accumulate, counters |
| `Max` | `max(old, src)` | global maximum (e.g. running softmax max) |
| `Min` | `min(old, src)` | global minimum |
| `And` | `old & src` | bit-mask clearing (integer) |
| `Or`  | `old \| src` | bit-mask setting / flags (integer) |
| `Inc` | `old + 1` (wrapping) | allocate-next-slot counters |
| `Dec` | `old - 1` (wrapping) | reference counts |

`And`/`Or`/`Inc`/`Dec` are integer ops; `Add`/`Max`/`Min` work on float or int.

#### Universal vs Buffer atomics

Same split as the plain copies:

- **`UniversalAtomic`** — pointer-based, any address space, no bounds check. Build it
  from an `AtomicOp` + value type (`primitive.py:198`).
- **`BufferAtomic`** — CDNA3 buffer-resource atomic, so the descriptor's `num_records`
  range gives free OOB masking for ragged/boundary tiles (`rocdl/universal.py:53`).

```python
# Universal (any addr space) — primitive.py:198
add = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx.Float32), fx.Float32)
mx  = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Max, fx.Float32), fx.Float32)

# src is a register tensor holding this lane's contribution; dst is the SHARED target.
# Semantically: tOut[0] = tOut[0] + rA   (atomically, across all racing lanes)
fx.copy_atom_call(add, rA, fx.slice(tOut, (None, fx.Int32(0))))

# Buffer variants (OOB-checked) — rocdl/universal.py:53
fx.rocdl.BufferAtomicAdd(fx.Float32)
fx.rocdl.BufferAtomicMax(fx.Float32)
fx.rocdl.BufferAtomicMin(fx.Float32)
fx.rocdl.BufferAtomicPkAdd(fx.Float16)   # packed 2-wide add: two f16 lanes in one op
```

`BufferAtomicPkAdd` packs **two** values into a single atomic (a `v2` vector type), so
you add two adjacent f16/bf16 elements in one instruction — handy for half-precision
accumulation where a scalar atomic would be wasteful.

#### Things to watch out for

- **Return value**: these atoms are used for their *effect* on `dst`; treat the
  pre-update value as unavailable (FlyDSL exposes the op as a copy, not a fetch).
- **No ordering between different addresses**: atomicity is per-address only. If you
  need all blocks' adds to be *visible* before a later read, you still need the usual
  cross-block synchronization (separate kernel launch, or a global barrier) — an atomic
  alone doesn't order unrelated memory.
- **Init the accumulator**: an `AtomicAdd` reduction assumes the destination starts at
  the identity (0 for Add, `-inf` for Max, etc.). Zero it (e.g. in a prior kernel or
  `torch.zeros`) before launching.
- **fp atomics are non-deterministic in order**: floating-point `Add` results can vary
  run-to-run in the low bits because the summation order is not fixed.

Available `AtomicOp`s wired up as `UniversalAtomic*` helpers: `Add`, `Max`, `Min`,
`And`, `Or`, `Inc`, `Dec` (`primitive.py:199`). Buffer helpers: `Add`, `Max`, `Min`,
packed-`Add` (`rocdl/universal.py:63`). See `tests/unit/test_universal_atomic.py` for
runnable examples.

---

## 3. `copy_atom_call` parameters

```python
fx.copy_atom_call(copy_atom, src, dst, *, pred=None)
```

- **`copy_atom`** — the atom from `make_copy_atom`.
- **`src` / `dst`** — tensors. Direction is inferred from their address spaces
  (global buffer slice ↔ rmem/smem fragment). For atomics, `dst` is the accumulator.
- **`pred`** — optional per-element predicate tensor to mask lanes/elements. `None`
  means "copy everything". Buffer atoms often don't need `pred` because the descriptor
  already masks OOB; `pred` is for logical masking you compute yourself.

There is also `fx.copy(copy_atom, src, dst, *, pred=None, **kwargs)`, which is the
same call but lets you set atom state inline (e.g. `soffset=...`) via
`copy_atom.set_value(kwargs)`. Source: `primitive.py:970`.

---

## 4. Tiled copies — one atom, a whole tile, all threads

For real kernels you rarely call the atom element-by-element. A **`TiledCopy`** binds
a copy atom to a **thread-value (TV) layout** so one logical copy spreads a tile across
all lanes with the right per-thread fragment.

```python
# Generic: atom + TV layout + tiler  (flydsl-tile-programming skill)
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
layout_tv = fx.raked_product(thr_layout, val_layout)   # which lane owns which elems
tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, fx.make_tile(4, 8))

thr_copy = tiled_copy.get_slice(tid)     # this thread's view (a ThrCopy)
src = thr_copy.partition_S(bA)           # this lane's source fragment
dst = thr_copy.partition_D(bB)           # this lane's dest fragment
frag = fx.make_fragment_like(src)
fx.copy(copy_atom, src, frag)            # global -> registers
fx.copy(copy_atom, frag, dst)            # registers -> global
```

`make_tiled_copy(copy_atom, layout_thr_val, tile_mn)` — `primitive.py:926`. The
per-thread `ThrCopy` exposes `partition_S` (source) and `partition_D` (dest).

### MMA-matched tiled copies

When loading operands for an `fx.gemm`, the copy's per-thread layout must match what
the matrix instruction expects. FlyDSL derives it for you from a `tiled_mma`:

```python
tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)   # operand A layout
tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)   # operand B layout
tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)   # accumulator/output
```

Defined in `python/flydsl/expr/derived.py:139`. See `kernels/preshuffle_gemm_v2.py`
(lines ~124–171) and the `flydsl-tile-programming` skill for full GEMM wiring.

---

## 5. Where each variant shows up in the repo

| Variant | Real usage |
|---------|-----------|
| `BufferCopy32b` (1 elem) | `lesson_00_hello_flydsl.py`, `lesson_13_fast_exp2.py`, scale loads in `rmsnorm_kernel.py` |
| `BufferCopy128b` (wide) | `lesson_09_wide_loads.py`, `softmax_kernel.py:118`, `rmsnorm_kernel.py:196`, `preshuffle_gemm_v2.py` |
| `BufferCopy(bits)` parametric | `topk_gating_softmax_kernel.py:272`, `qk_norm_rope_quant.py:278` |
| `UniversalCopy128b` | `efficient-gemm.py:160` (LDS↔reg), `preshuffle_gemm_v2.py:455` (g2s) |
| `buffer_load_to_lds` (g2s DMA) | `fmha_prefill_fp8_ck_hk5.py:291` and siblings (K prefetch) |
| `UniversalAtomic` Add/Max | `tests/unit/test_universal_atomic.py` |
| `make_tiled_copy_A/B/C` | `preshuffle_gemm_v2.py:132`, GEMM examples |

> Note: the FMHA prefill kernels (`kernels/fmha_prefill_fp8_*.py`) mostly drop to the
> lower-level `buffer_ops.buffer_load(... vec_width=...)` / `buffer_load_to_lds`
> intrinsics rather than `make_copy_atom`, because they hand-schedule the loads. The
> copy-atom API is the higher-level, layout-driven front-end for the same hardware ops.

---

## 6. Quick decision guide

1. **Which space?** global that may be OOB → `BufferCopy*`. LDS / known-in-bounds →
   `UniversalCopy*`. global → LDS bulk prefetch → `BufferCopyLDS*` /
   `buffer_load_to_lds`.
2. **Accumulating into the destination?** → an atomic atom (`*Atomic*`).
3. **How wide?** pick the largest `copy_op` your per-thread tile allows; element count
   = `op_bits / elem_bits`, capped at 128-bit transactions. Wider = fewer
   instructions + better coalescing, but only helps if the kernel is memory-bound
   (lesson 09's punchline).
4. **One element or a tile across threads?** single element → `copy_atom_call`. A whole
   tile spread over the block → `make_tiled_copy` (+ `make_tiled_copy_A/B/C` when
   feeding an MMA).
5. **Partial/masked tile?** rely on the buffer descriptor's OOB masking, or pass an
   explicit `pred=` tensor.

---

## 7. Minimal templates

```python
# (a) single-element global<->reg, OOB-safe (lesson 00 pattern)
atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
fx.copy_atom_call(atom, fx.slice(tX, (None, tid)), r)     # load
# ... fx.memref_load_vec(r) / compute / fx.memref_store_vec(...) ...
fx.copy_atom_call(atom, r, fx.slice(tY, (None, tid)))     # store

# (b) wide vectorized load (lesson 09 pattern)
atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)   # 4 f32 = 128 bits
fx.copy_atom_call(atom, tX, reg)
fx.copy_atom_call(atom, reg, tY)

# (c) atomic accumulate into global
add = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx.Float32), fx.Float32)
fx.copy_atom_call(add, r, fx.slice(tOut, (None, tid)))
```
