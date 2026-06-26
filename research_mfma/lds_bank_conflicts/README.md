# LDS Bank-Conflict Scenarios (FlyDSL)

A FlyDSL port of CK-Tile's `tutorial_14_bank_conflict_scenarios`, focused on the
LDS-layout cases. Every file does the **same job** — transpose `X[M,K] -> Y[K,M]`
through LDS — and changes **only the LDS layout**. That layout choice is what
creates or avoids bank conflicts on the transposed read.

## The kernel (shared, `lds_common.py`)

```
load 64x64 tile from global X  (row-major, coalesced)
store tile into LDS at lds_layout(m, k)
block barrier
read it back transposed: value at logical (m=yc, k=yr) = lds_layout(yc, yr)
store to global Y  (coalesced)
```

Correctness holds for **any** layout, because a layout is a bijection used
identically on write and read. Only the bank behaviour differs.

## LDS banks (gfx942 / CDNA3)

32 banks × 4 bytes. For f32, `bank(addr_elems) = addr_elems % 32`. A wave64 access
spans 64 addresses over 32 banks, so the conflict-free floor is **2-way**. The
transpose read has consecutive lanes reading consecutive `m` at a fixed `k`.

## The four cases — expressed with FlyDSL layout algebra

The whole point: the LDS address is **not** hand-computed (`m*stride+col`, manual `^`).
The LDS is a layout-bearing tensor (`make_view(get_dyn_shared(f32), layout)`) and
plain `s[m,k]` indexing applies the layout — the idiomatic analog of CK's
`make_naive_tensor_descriptor` (strides) and `make_xor_transform` (swizzle).

| # | File | LDS layout (FlyDSL) | read bank pattern |
|---|------|---------------------|-------------------|
| 1 | `01_row_major.py` | `make_ordered_layout((TM,TK),(1,0))` | `m*64+k` → bank `k%32` for **all m** → all-same-bank |
| 2 | `02_column_major.py` | `make_ordered_layout((TM,TK),(0,1))` | `k*64+m`, stride 1 → contiguous (conflict moves to write) |
| 3 | `03_padded.py` | `make_layout((TM,TK),(TK+1,1))` | `m*65+k` → bank `(m+k)%32` → spreads all 32 banks |
| 4 | `04_xor.py` | `make_composed_layout(SwizzleType.get(5,0,6), row-major)` | bank `k^(m&31)` → spreads all 32 banks, zero extra LDS |

`SwizzleType.get(B, M, S)` = CK's `make_xor_transform`: XOR `B` bits at position `M`
with the bits at `M+S`. `Swizzle(5,0,6)` over a 64-wide f32 row extracts `m`
(bits [6,11)) and XORs it into the bank bits [0,5).

## Measured (MI308X / gfx942, X[4096,4096] f32, tile 64×64)

| scenario | time | bandwidth | vs baseline |
|----------|------|-----------|-------------|
| 1. row-major (baseline) | 162 µs | 829 GB/s | 1.0× |
| 2. column-major | 161 µs | 834 GB/s | 1.0× |
| 3. padded (+1 elem) | 84 µs | 1592 GB/s | **1.92×** |
| 4. xor swizzle | 87 µs | 1549 GB/s | **1.87×** |

Takeaways:
- **Row vs column-major are the same speed** — column-major makes the *read*
  conflict-free but moves the identical conflict to the *write*; net cost unchanged.
- **Padding and XOR both ~1.7–1.8× faster** by spreading the strided access across
  all 32 banks.
- **XOR ≈ padding in speed but uses zero extra LDS** — padding costs +1.5% LDS, which
  matters when LDS capacity bounds occupancy. This is why production GEMM/attention
  kernels prefer XOR swizzles.

## Run

```bash
cd FlyDSL
for f in 01_row_major 02_column_major 03_padded 04_xor; do
  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_bank_conflicts/$f.py
done
```

## Note on CK scenario 5 (xor + padding)

CK also combines XOR with padding. Naively composing a *bit*-swizzle over a
*non-power-of-two* padded stride (65) is not a clean bijection (the swizzle would
XOR bits that no longer isolate `m`), so it needs a coordinate-level swizzle (permute
`k` within the row, then apply the padded row stride) rather than `make_composed_layout`
over the final offset. Left out here to keep each layout a single clean primitive.

## FlyDSL LDS idioms used (no manual address arithmetic)

- LDS tensor: `make_view(get_dyn_shared(fx.Float32), lds_layout)`; dynamic smem is
  reserved at launch via `.launch(..., smem=bytes)`.
- LDS layout (the bank-conflict knob): `make_ordered_layout` / `make_layout` /
  `make_composed_layout` + `SwizzleType` — order, padding and swizzle live here.
- Global tiles: `flat_divide(make_buffer_tensor(X), (TM,TK))[None,None,bm,bk]`.
- Element access: plain indexing `s[m,k]` / `gX[m,k]` (the layout does the addressing).
- Linear id -> coordinate: `idx2crd` (instead of `//` and `%`).
- Sync: `fx.gpu.barrier()`.
