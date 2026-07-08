<!-- SPDX-License-Identifier: Apache-2.0 -->
# FlyDSL GEMM instructions, mapped to two CK examples

Each FlyDSL instruction (as used in `research_mfma/minimal_tiled_gemm.py`) explained by what it
*does*, and its counterpart in two CK reference kernels:

- **RAW** = `01_simple_hgemm.cpp` — a hand-written HIP MFMA GEMM: explicit per-lane loads, the raw
  `__builtin_amdgcn_mfma_*`. This mirrors FlyDSL's *by-hand* style (lessons 01/03).
- **CK-TILE** = `tile_sweeping_with_y_repetition.cpp` — CK Tile with `tile_distribution_encoding`,
  `WarpGemm`, and tile windows. This mirrors FlyDSL's *layout-algebra* style.

Both CK kernels use the **16×16×16 f16 MFMA** and a 64-lane wave, so the pieces line up 1:1.

---

## The FlyDSL kernel we're mapping (anchor)

```python
A = fx.rocdl.make_buffer_tensor(A)                                   # (1)
gA = fx.flat_divide(A, (m, k))[None, None, 0, None]                  # (2)

mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(m, n, k, fx.BFloat16))    # (3)
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1,1,1),...)) # (4)
thr_mma   = tiled_mma.thr_slice(tid)                                 # (5)
frag_A    = thr_mma.make_fragment_A(...)                            # (5)

cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy(64), fx.BFloat16)      # (6)
tcA   = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)        # (7)

for kt in fx.range_constexpr(K // k):                                # (9)
    fx.copy(cp_ab, tcA.partition_S(...)[..., kt], tcA.retile(frag_A))# (8)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)                # (10)

fx.copy(cp_c, tcC.retile(frag_C), tcC.partition_S(gC))               # (11)
```

---

## Instruction-by-instruction

### (1) `fx.rocdl.make_buffer_tensor(T)` — wrap global memory
Gives the tensor an AMD buffer descriptor + a layout (shape/stride) so later ops can compute
addresses.
- **RAW:** the bare pointer + `ld` (`lda`, `ldb`) passed to the load helpers; `col_major`/`row_major`
  lambdas do `coord.first + coord.second*ld`.
- **CK-TILE:** `make_naive_tensor_view<global>(a, make_tuple(M,K), make_tuple(1,lda), ...)` — the
  tensor view with shape + strides.

### (2) `fx.flat_divide/zipped_divide(...) + fx.slice(..., (None,bid))` — pick this block's tile
Splits the tensor into `(tile, rest)` and selects which tile this block/wave owns.
- **RAW:** `waveGridX = (blockIdx.x*blockDim.x + threadIdx.x)/WAVE_SIZE`, then
  `cRow = waveGridX*BLOCK_M`, `cCol = waveGridY*BLOCK_N` — the by-hand "which tile" math, and the
  pointer offset `a + (cRow + i*lda)`.
- **CK-TILE:** `block_m/block_n = get_block_id() ... ; m_block_base = block_m*kMPerBlock`, then
  `make_tile_window(tensor, (kMPerBlock,kWarpK), {m_block_base,0}, distribution)`.

### (3) `fx.make_mma_atom(fx.rocdl.MFMA(m,n,k,ty))` — name the MFMA instruction
One hardware MFMA + its fixed per-operand TV layouts.
- **RAW:** the `__device__ mfma_f32_16x16x16f16(...)` wrapper around
  `__builtin_amdgcn_mfma_f32_16x16x16f16`.
- **CK-TILE:** `using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16;` — the `WarpGemm` type **is** the
  atom (its `WarpGemmAttributeMfma` carries the A/B/C distribution encodings).

### (4) `fx.make_tiled_mma(atom, make_layout((Mrep,Nrep,Krep),...))` — replicate the atom over waves
The atom spread across the block's waves; derives the tiled A/B/C layouts.
- **RAW:** implicit — one wave = one block tile (`T_BLOCK_X = 1*WAVE_SIZE`), so `Mrep=Nrep=1`.
- **CK-TILE:** `MWarp = 2, NWarp = 2` (a 2×2 wave grid, `kBlockSize = MWarp*NWarp*64 = 256`). That
  `(MWarp, NWarp)` **is** the `atom_layout` you'd pass to `make_tiled_mma((2,2,1),...)`.
  *Extra:* CK also adds `MIterPerWarp/NIterPerWarp` (Y-repetition) so each warp sweeps 2×2 tiles —
  in FlyDSL that's either a larger atom-layout replication or an outer unrolled loop over the
  fragment's repeat dimension (minimal_tiled_gemm uses 1, i.e. no repetition).

### (5) `thr_mma = tiled_mma.thr_slice(tid)` + `make_fragment_A/B/C(...)` — this lane's registers
Per-lane operand view; allocate the register fragments with the correct lane↔element map.
- **RAW:** `AFragT = VecT<float16_t, BLOCK_M*BLOCK_K/64>` (= 4), `auto fragA = AFragT{}` — the 4-reg
  vector per lane. The "Register Mapping" comment tables (`Reg0[0:15]=K0`, …) **are** the TV layout.
- **CK-TILE:** `make_static_distributed_tensor<ADataType>(make_static_tile_distribution(a_warp_dstr_encode))`
  and the `a_warp_dstr_encode = tile_distribution_encoding<...>` — CK's spelling of the same
  per-warp TV layout FlyDSL stores in `mma_atom.layout_A_tv`.

### (6) `fx.make_copy_atom(fx.rocdl.BufferCopy(bits), ty)` — name one copy instruction
The data-movement analog of the MMA atom: how many bits one lane moves (`64 = 4×f16`).
- **RAW:** the load helpers build the vector element-by-element (`input[startOffset + i*kOffset]`);
  for C it becomes a single `global_load_dwordx4`. No named atom — it's the load pattern itself.
- **CK-TILE:** implicit inside `load_tile` / the tensor view's vector width (`number<4>{}` alignment
  on the B view).

### (7) `fx.make_tiled_copy_A/B/C(cp, tiled_mma).get_slice(tid)` — copy matched to the MMA
Builds a copy whose thread-value layout is taken from the tiled MMA, so loads land in the lanes the
MFMA expects; `.get_slice(tid)` is this lane's view (`partition_S`, `retile`).
- **RAW:** the hand-written `startCoord2D = (threadIdx.x%Dim, (threadIdx.x/Dim)*VW)` in
  `load_A_16x16_col_major` — literally computing where this lane reads. That coord math == what
  `partition_S` generates from the TV layout.
- **CK-TILE:** attaching `a_block_distribution` to `make_tile_window(...)`; the block distribution is
  `make_embed_tile_distribution_encoding(a_block_outer_dstr_encode, a_warp_dstr_encode)` — the
  block-over-warp compose that `make_tiled_copy_A` does.

### (8) `fx.copy(cp, src_partition, retile(frag))` — global → register load
Moves this lane's slice from global into its fragment.
- **RAW:** `fragA = load_A_16x16_col_major(a + (cRow + i*lda), lda);` (and `load_B...`).
- **CK-TILE:** `const auto a_block_tile = load_tile(a_block_window);` then
  `get_y_sliced_thread_data(...)` to pull one warp-tile out.

### (9) `for kt in fx.range_constexpr(K // k)` — the K accumulation loop
Loop the contraction in MFMA-K chunks, accumulating into the C fragment.
- **RAW:** `for(int i = 0; i < k; i += BLOCK_K)`.
- **CK-TILE:** `for(k_iter ...) { ... move_tile_window(a_block_window, {0, kWarpK}); }`.

### (10) `fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)` — the MFMA
Issue the matrix multiply-accumulate on the fragments.
- **RAW:** `fragAcc = mfma_f32_16x16x16f16(fragA, fragB, fragAcc);`.
- **CK-TILE:** `WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);` (inside the
  `static_for` over `mIter/nIter` — the Y-repetition tile sweep).

### (11) store: `fx.copy(cp_c, retile(frag_C), tcC.partition_S(gC))` — register → global
Write the C fragment back to global (optionally after epilogue scaling).
- **RAW:** `store_C_16x16_col_major(d + (cRow + cCol*ldd), fragC, ldd);` (after
  `fragC[i] = alpha*fragAcc[i] + beta*fragC[i]`).
- **CK-TILE:** `store_tile(d_block_window, c_block_tile);` (after `tile_elementwise_inout` scales by
  alpha/beta).

---

## Summary table

| FlyDSL | does | `01_simple_hgemm` (RAW) | `tile_sweeping...` (CK-TILE) |
|---|---|---|---|
| `make_buffer_tensor` | wrap global + layout | bare ptr + `ld` + `col/row_major` | `make_naive_tensor_view` |
| `flat_divide`+`slice` | pick block's tile | `cRow = waveGridX*BLOCK_M` | `make_tile_window(...,{m_block_base,0})` |
| `make_mma_atom(MFMA)` | name the MFMA | `mfma_f32_16x16x16f16` builtin | `WarpGemmMfmaF16F16F32M16N16K16` |
| `make_tiled_mma(atom, layout)` | atom × wave grid | 1 wave (implicit) | `MWarp,NWarp = 2,2` (+ `*IterPerWarp`) |
| `make_fragment_A/B/C` | per-lane registers | `AFragT/BFragT/AccumFragT` | `make_static_distributed_tensor` |
| *(the TV layout)* | lane↔element map | the "Register Mapping" comments | `*_warp_dstr_encode` |
| `make_copy_atom(BufferCopy)` | one copy instr | load helper pattern | vector width in `load_tile` |
| `make_tiled_copy_A/B/C` + `partition_S` | copy matched to MMA | `startCoord2D` lane math | block distribution + window |
| `fx.copy` | global→reg | `load_A/B_16x16_*` | `load_tile` / `get_y_sliced_thread_data` |
| K-loop | accumulate over K | `for i in 0..k step BLOCK_K` | `for k_iter` + `move_tile_window` |
| `fx.gemm` | the MFMA | `mfma_f32_16x16x16f16(...)` | `WarpGemm{}(c,a,b)` |
| store copy | reg→global | `store_C_16x16_col_major` | `store_tile` |

## The one distinction between the two CK styles
- **RAW (`01_simple_hgemm`)** hand-codes the lane↔element math (`threadIdx.x % Dim`, the register
  tables). FlyDSL's *by-hand* lessons (01/03) look like this; the offsets are written out.
- **CK-TILE (`tile_sweeping`)** declares `tile_distribution_encoding`s and lets `WarpGemm` +
  `make_tile_window` place data. FlyDSL's `make_tiled_mma` + `make_tiled_copy_*` is the same idea:
  the TV layout (FlyDSL `layout_A_tv` ↔ CK `a_warp_dstr_encode`) drives everything, so you never
  write `threadIdx.x % 16` yourself.

---

## Inside `make_tiled_mma`: what its objects actually are

**Is a `TiledMma` the same as a `tile_distribution_encoding`?** No — a `TiledMma` is a *bundle*.
Its `tv_layout_{A,B,C}_tiled` **members** are the direct analogs of CK's `*_dstr_encode`
(tile_distribution_encoding); the object as a whole also carries the atom, the wave replication,
the thread layout, and the methods (`thr_slice`, `make_fragment_*`, `partition_*`). In CK that
same content is spread across `WarpGemm` (the atom) + three `*_block_dstr_encode`s + the
`make_tile_window` / `get_slice` helpers.

Walking the pieces (`tiled_mma = make_tiled_mma(mma_atom, atom_layout)`):

- **`mma_atom`** — the underlying `MmaAtom`: one MFMA plus its fixed per-operand hardware TV
  layouts (`layout_A_tv`, `layout_B_tv`, `layout_C_tv`, `shape_mnk`, `thr_layout`). "Which
  instruction + how one wave holds its operands."
- **`atom_layout`** — the `(Mrep, Nrep, Krep)` layout you passed: how many copies of the atom run
  and how they're arranged over the block's waves. Its *size* = number of waves.
- **`permutation_mnk`** — optional reordering of the M/N/K tiling (usually none/identity); lets you
  permute how atoms map onto the tile.
- **`tile_size_mnk`** — the `M×N×K` this tiled MMA covers = atom shape × `atom_layout`. For a
  `16×16×16` atom with `(2,2,1)` → `32×32×16`.
- **`thr_layout_vmnk`** — the thread layout over `(V, M, N, K)`: maps a thread id to `(v = lane
  within the atom, m-atom, n-atom, k-atom)`. It's "which wave/lane owns which atom position."
- **`tv_layout_A_tiled` / `_B_tiled` / `_C_tiled`** — the **tiled** thread-value layouts: the atom's
  TV layout scaled up by `atom_layout`, mapping `(thread, value) → position` in the whole tiled
  A/B/C tile. **These are the tile_distribution_encoding equivalents.**
- **`thr_slice(tid)` / `get_slice(tid)` → `ThrMma`** — this lane's view of the tiled MMA (the object
  that actually hands you fragments and partitions).
- **`ThrMma.make_fragment_A/B/C(block_tile)`** — allocate this lane's register fragment, sized and
  laid out per the tiled TV layout.
- **`ThrMma.partition_A/B/C(tensor)`** — given a block-level tensor, return the sub-view of elements
  *this lane* operates on (the operand partition; the copy analog is `ThrCopy.partition_S/D`).

### Table

| FlyDSL (`tiled_mma` / `thr_mma`) | what it is | `01_simple_hgemm` (RAW) | `tile_sweeping...` (CK-TILE) |
|---|---|---|---|
| `mma_atom` (`MmaAtom`) | the MFMA + its per-wave operand TV layouts | `mfma_f32_16x16x16f16` builtin + the "Register Mapping" comment tables | `using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16` (wraps `WarpGemmAttributeMfma`) |
| `atom_layout` | `(Mrep,Nrep,Krep)` replication over waves | 1 wave, implicit (`T_BLOCK_X = 64`) | `MWarp=2, NWarp=2` (wave grid) |
| `permutation_mnk` | optional M/N/K reorder | — | encoding ordering (none here) |
| `tile_size_mnk` | M×N×K the tiled MMA covers (atom × atom_layout) | `BLOCK_M×BLOCK_N×BLOCK_K` | `kMPerBlock/kNPerBlock` = warp×wave |
| `thr_layout_vmnk` | thread → `(lane v, m-atom, n-atom, k-atom)` | `waveGridX/Y` + `threadIdx.x % Dim` | `get_warp_id()`→`(iMWarp,iNWarp)` + lane |
| `tv_layout_A_tiled` / `_B` / `_C` | tiled `(thread,value)→pos` layout **(= the encodings)** | register-mapping tables extended across waves | `a_block_dstr_encode` = embed(`a_block_outer_dstr_encode`, `a_warp_dstr_encode`) |
| `thr_slice(tid)` / `get_slice(tid)` → `ThrMma` | this lane's view of the tiled MMA | implicit (each thread runs with its `threadIdx`) | per-thread slice of the distribution / `get_warp_id()` |
| `make_fragment_A/B/C(tile)` | allocate this lane's register fragment | `AFragT{}` / `BFragT{}` / `AccumFragT{}` (VecT<…,4>) | `make_static_distributed_tensor<T>(make_static_tile_distribution(warp_dstr_encode))` |
| `partition_A/B/C(tensor)` | this lane's slice of a block tensor (compute operand) | `startCoord2D`/`startOffset` in `load_A_16x16_*` | `make_tile_window(tensor, …, distribution)` + `get_y_sliced_thread_data` |

### The mental grouping
```
TiledMma  ≈  WarpGemm (mma_atom)                      # which instruction + per-wave layouts
           + atom_layout / thr_layout_vmnk            # how waves tile the block  (CK: MWarp/NWarp)
           + tv_layout_{A,B,C}_tiled                  # the 3 distribution encodings (CK: *_dstr_encode)
           + thr_slice / make_fragment_* / partition_*# the per-lane helpers (CK: make_tile_window / get_slice)
```
So it isn't one `tile_distribution_encoding` — it's the WarpGemm **plus** the three encodings
**plus** the helpers, packaged as a single object whose `.tv_layout_*_tiled` are those encodings.
