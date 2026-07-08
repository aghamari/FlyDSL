<!-- SPDX-License-Identifier: Apache-2.0 -->
# Lesson 04 — Dimensions & shapes walkthrough

A shape-by-shape reading of `lesson_04_softmax.py`: what each value's dimensions are, in
pseudocode. The kernel launches **one wavefront** (`block = 64 threads`, `grid = 1`) and
processes **one 16×16 score tile**.

## Constants

```
BQ     = 16     # queries in the tile      (Q rows)
BKV    = 16     # keys/kv in the tile       (K rows)
HD     = 64     # head dim                  (contraction length)
MFMA_K = 16     # K handled per MFMA step   -> HD // MFMA_K = 4 K-steps
WAVE   = 64     # lanes in a wavefront (AMD gfx942)
```

## Inputs / output (global memory)

```
Q : bf16[BQ , HD]  = bf16[16, 64]     # queries, row-major
K : bf16[BKV, HD]  = bf16[16, 64]     # keys,    row-major
P : f32 [BKV, BQ]  = f32 [16, 16]     # output softmax weights, [kv, q]
```

The math we implement:

```
S[kv, q] = (K @ Qᵀ)[kv, q] * sm_scale        # scores, shape [16, 16]
P[kv, q] = softmax over kv of S[:, q]        # normalise each query column down kv
```

---

## Step 1 — wrap as buffer tensors (no shape change)

```
K = make_buffer_tensor(K)   # bf16[16,64]  + AMD buffer descriptor
Q = make_buffer_tensor(Q)   # bf16[16,64]
P = make_buffer_tensor(P)   # f32 [16,16]
```

## Step 2 — carve MFMA-shaped tiles with flat_divide + slice

`flat_divide(T, (tileM, tileN))` splits both axes and flattens to
`(TileM, TileN, RestM, RestN)`; `slice(..., (None,None,0,None))` keeps the tile modes and
picks block 0 of the row axis.

```
# K is [kv=16, hd=64], tiled by (16, 16):
#   rows 16/16 = 1 tile ;  cols 64/16 = 4 tiles
flat_divide(K,(16,16))            -> (16, 16, 1, 4)      # (tileKv, tileK, restKv=1, restK=4)
gK = slice(..., (None,None,0,None)) -> (16, 16, 4)       # (kv, k, nK) ; nK = 4 K-steps
gQ = slice(...)                     -> (16, 16, 4)       # (q,  k, nK)
gP = slice(flat_divide(P,(16,16)), (None,None,0,0)) -> (16, 16)   # (kv, q)
```

So:

```
gK : (kv=16, k=16, nK=4)      # the K tile, split into 4 K-steps
gQ : (q=16 , k=16, nK=4)      # the Q tile, split into 4 K-steps
gP : (kv=16, q=16)            # the output score tile
```

## Step 3 — the MMA and its per-lane fragments

```
mma_atom  = MFMA(16, 16, 16, bf16)          # one hardware matrix op: 16x16x16
tiled_mma = make_tiled_mma(mma_atom, (1,1,1))  # 1 atom, 1 wave (no replication)
thr_mma   = tiled_mma.thr_slice(tid)         # this lane's view
```

For a `16×16×16` bf16 MFMA on a 64-lane wave, each lane owns:

```
frag_K (A operand)  : 4 x bf16     # per lane, for ONE K-step
frag_Q (B operand)  : 4 x bf16     # per lane
frag_S (C accum)    : 4 x f32      # per lane  -> the 4 scores this lane will hold
```

(16·16 = 256 A-elements / 64 lanes = 4 per lane; same for B; C is 16·16 = 256 / 64 = 4 f32
per lane.)

### What "the lane view" is

`thr_mma = tiled_mma.thr_slice(tid)` is **this lane's slice** of the tiled MMA: for lane
`tid` it knows exactly which elements of A, B, C that lane owns. `make_fragment_A/B/C`
allocate the small register fragments using that map, and `partition_S` / `copy` fill them
from global memory at the right addresses. `make_tiled_mma` *derives* this map from the
hardware MFMA definition — the rest of this section is the same map written **by hand**, so
you can see what the abstraction computes for you.

### From tile layout → lane layout (16×16×16 bf16, 64 lanes), by hand

Number the 64 lanes as a `[k_outer, mn]` grid:

```
lane    = tid
k_outer = lane // 16     # 0..3   (which quarter of the 16-wide K / which 4 rows of C)
mn      = lane % 16      # 0..15  (the M/N index this lane serves)
e       = 0..3           # the 4 values this lane holds
```

The CDNA3 fragment layout (verified empirically in Lesson 01, not guessed):

```
A operand (= K, rows indexed by kv):
    frag_K[e] = K[ kv = mn , hd = k_outer*4 + e ]     # 4 contiguous K per lane
B operand (= Q, rows indexed by q):
    frag_Q[e] = Q[ q  = mn , hd = k_outer*4 + e ]
C accumulator (= S, laid out [kv, q]):
    frag_S[e] = S[ kv = k_outer*4 + e , q = mn ]      # 4 contiguous kv per lane
```

So if you wrote the loads/stores **by hand** (this is exactly what Lessons 01/03 do), it is
plain index math — no `partition_S`, no `make_fragment`:

```
# load A fragment for K-step kt  (global K index = kt*16 + k_outer*4 + e):
for e in 0..3:
    frag_K[e] = K[ mn , kt*16 + k_outer*4 + e ]       # addr = mn*HD + kt*16 + k_outer*4 + e

# store C fragment:
for e in 0..3:
    S[ k_outer*4 + e , mn ] = frag_S[e]               # addr = (k_outer*4 + e)*BQ + mn
```

`make_tiled_mma` + `make_fragment_C` + `partition_S` produce **exactly these addresses**:
`partition_S(gK)` + `retile(frag_K)` compute `mn*HD + kt*16 + k_outer*4 + e` for you, and
`tcP.partition_S(gP)` computes `(k_outer*4 + e)*BQ + mn` on the store. The "lane view" is the
object holding this per-lane `(k_outer, mn) → elements` mapping.

### The same thing as a layout (TV form)

Read the C mapping as a **thread-value layout** `(thread, value) → (row, col)`:

```
thread = lane = k_outer*16 + mn      # 64 threads
value  = e                           # 4 values per thread
row (kv) = k_outer*4 + e   =  (lane // 16)*4 + e
col (q)  = mn              =   lane % 16
```

That is a `(64, 4)`-shaped TV layout over the `16×16` tile — precisely the
`tv_layout_C_tiled` the tiled MMA stores and passes to `make_tiled_copy_C`. Writing it out
by hand is what Lessons 01/03 keep explicit; `make_tiled_mma` is exactly this decomposition
packaged so you never spell out the `(k_outer, mn, e)` math yourself.

## Step 4 — the K-loop (accumulate over HD)

```
for kt in 0..nK-1:                 # nK = HD/MFMA_K = 4 steps
    copy K-step kt  ->  frag_K     # 4 bf16 per lane
    copy Q-step kt  ->  frag_Q     # 4 bf16 per lane
    frag_S += mma(frag_K, frag_Q)  # 16x16x16, accumulate into 4 f32 per lane
```

After 4 steps `frag_S` (4 f32 per lane) holds the full scores. The lane↔element mapping is
the CDNA3 C-fragment layout:

```
lane = (k_outer = lane // 16, mn = lane % 16)
frag_S[e]  ==  S[kv = k_outer*4 + e, q = mn]   for e = 0..3
```

So each lane holds **4 kv values for one fixed query column `q = mn`**; the other 12 kv for
that query live in lanes `mn+16, mn+32, mn+48`.

## Step 5 — softmax over kv (a reduction, kept direct)

```
sv = frag_S * sm_scale             # 4 f32 per lane (the scaled scores)

# max over kv:
m = max(sv[0..3])                  # intra-lane: this lane's own 4 kv
m = max(m, shuffle_xor(m, 16))     # cross-lane: merge k_outer groups (XOR 16)
m = max(m, shuffle_xor(m, 32))     #             (XOR 32) -> all 16 kv covered

p[e] = exp2((sv[e] - m) * log2e)   # 4 values per lane
l = sum(p[0..3])                   # intra-lane sum
l = l + shuffle_xor(l, 16)         # cross-lane
l = l + shuffle_xor(l, 32)
inv_l = 1 / l

pw[e] = p[e] * inv_l               # normalised weights, 4 per lane
```

The `shuffle_xor(mask, 64)` lets a lane read `lane XOR mask`; `XOR 16` and `XOR 32` flip the
two bits of `k_outer = lane//16`, so after both every lane in column `mn` shares the reduced
value.

## Step 6 — store P via the tiled copy

```
frag_S = pw                        # write the 4 weights back into the C fragment
cp_c   = BufferCopy32b, f32        # 32-bit = 1 f32 per element
tcP    = tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
copy(frag_S -> gP)                 # lands at P[kv = k_outer*4+e, q = mn]
```

---

## The `BufferCopy` width calculation

```python
cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((BKV * MFMA_K // 64) * 16), fx.BFloat16)
```

`BufferCopy(bits)` picks the width, in **bits**, of one thread's buffer load. The expression
computes the number of bits **one lane** must move to fill its slice of a `BKV × MFMA_K`
operand tile:

```
BKV * MFMA_K            = 16 * 16 = 256     # elements in one A/B operand tile (per K-step)
(BKV * MFMA_K) // 64    = 256 // 64 = 4     # elements PER LANE   (256 elems / 64 lanes)
(... ) * 16             = 4 * 16 = 64       # bits  PER LANE      (4 elems * 16 bits/bf16)
```

So `BufferCopy(64)` = a **64-bit** load = **4 bf16 elements per lane**, which is exactly the
size of the `frag_K` / `frag_Q` fragment. Breaking down the three factors:

| factor | meaning |
|---|---|
| `BKV * MFMA_K` | elements in one operand tile (`16×16 = 256`) |
| `// 64` | divide across the 64 lanes of the wave → **4 elements/lane** |
| `* 16` | multiply by **16 bits** (one bf16) → **64 bits/lane** |

General form: `bits_per_lane = (tile_elements / wave_size) * bits_per_element`. If the dtype
were f16 it would still be 16 bits; for f32 you'd use `* 32`. Keep it ≤ 128 (the max copy
atom width), which holds here (64).

The store side uses `BufferCopy32b` = **32 bits** = **1 f32** per element, matching the f32
`P` output written one value at a time from the C fragment.

## Dimension cheat-sheet

| value    | dims (per tile)     | per-lane        | dtype |
|----------|---------------------|-----------------|-------|
| `Q`,`K`  | `[16, 64]`          | —               | bf16  |
| `P`      | `[16, 16]`          | —               | f32   |
| `gK`,`gQ`| `[16, 16, 4]`       | —               | bf16  |
| `gP`     | `[16, 16]`          | —               | f32   |
| `frag_K` | A operand, 1 K-step | `4 × bf16`      | bf16  |
| `frag_Q` | B operand, 1 K-step | `4 × bf16`      | bf16  |
| `frag_S` | C accumulator       | `4 × f32`       | f32   |
| `sv`,`p`,`pw` | scores/weights | `4 × f32`       | f32   |
```
