<!-- SPDX-License-Identifier: Apache-2.0 -->
# ALGORITHM — fp8 paged causal FMHA prefill (gfx942)

The precise operation, data layout, and per-threadgroup steps for the FlyDSL FMHA prefill
kernel. The optimization history lives in [README.md](./README.md); this file is the
ground-truth math + memory contract that every level must preserve (the parity gate
`ck_check.py` enforces `max|O - ref| < 6e-2`).

## Operation

Flash-attention prefill, one query block per workgroup, online (streaming) softmax:

```
S   = (Q · Kᵀ) · sm_scale            # [sq, sk], sm_scale = 1/sqrt(HD)
S'  = causal_mask(S)                  # S'[i,j] = -inf  where  j > i + (sk - sq)
P   = softmax_rows(S')                # row-wise, numerically-stable streaming
O   = P · V                           # [sq, HD]
```

* **dtype:** Q, K, V are fp8 **e4m3 FNUZ** (1 byte). Accumulation is f32. Output O is **bf16**.
* **scales (dynamic quant):** per-token-per-head for Q (`Qd[b,h,sq]`) and K (`Kd[b,hk,sk]`),
  per-head for V (`Vd[b,hk]`). Q/K descales multiply the GEMM1 score; V descale folds into
  the epilogue normalization. A per-`(batch, qhead)` `p_scale` (`Ps`) rescales P into the
  e4m3 range before the fp8 cast and cancels exactly in O/L (`log2(p_scale)` is added to the
  exp bias).
* **shape (customer):** `HD=128`, GQA `nq=8 / nk=1` (q-head `h` uses kv-head `h // gqa`),
  causal, `page_size=16`, bs=1, seqlens 1024 / 2048 / 16384 / 32768.
* **atom:** both GEMMs use `mfma_f32_32x32x16_fp8_fp8` (K=16 fp8 contraction per instruction,
  32×32 f32 accumulator tile, one wave = 64 lanes).

## Data layout

### Inputs / output
| tensor | shape | layout |
|--------|-------|--------|
| `Q`  | `[b, sq, nq, HD]` fp8 | token-major; q-token stride `nq*HD` |
| `O`  | `[b, sq, nq, HD]` bf16 | same layout as Q |
| `Qd` | `[b, nq, sq]` f32 | per-token-head Q descale |
| `Kd` | `[b, nk, sk]` f32 | per-token-head K descale |
| `Vd` | `[b, nk]` f32 | per-head V descale |
| `Ps` | `[b*nq]` f32 | per-(batch, q-head) p_scale (1.0 = disabled) |

### Paged KV (`vec_k_col_v`)
Paging is a flat physical-page table `LTD[total_slots]` (int32 page id per slot) plus a
per-batch `kv_indptr` `LTP[b+1]`. For a logical kv index `kv` in batch `b`:

```
slot  = LTP[b] + kv // page_size
phys  = LTD[slot]            # physical page id
intra = kv % page_size       # row within the page
```

* **K pool** — `[pages, nk, hd/16, page_size, 16]` (the `vec_k_col_v` K layout). The 128
  features of a key are split into `hd/16 = 8` feature-groups of 16 fp8; within a group the
  16 features are contiguous, and the `page_size` rows of a group are contiguous. Per-kv
  base byte: `phys*k_page_stride + kvhead*HD*page_size + intra*16`; feature-group `cg` is at
  `+ cg*(page_size*16)`.
* **V pool — two layouts (this is lever 7):**
  * **row-major** (baseline, `V_COL=0`): `[pages, page_size, nk, hd]`; per-kv base
    `phys*v_page_stride + intra*(nk*HD) + kvhead*HD + d`. GEMM2's contraction dim (kv) is
    NOT contiguous ⇒ needs a transpose into LDS (16× `ds_write_b8` scatter).
  * **column-major** (`V_COL=1`, CK true `vec_k_col_v`): `[pages, nk, hd, page_size]`; kv is
    contiguous per `(head, d)` ⇒ the V→LDS copy is one 128-bit store/slot, **no transpose**.
  The host packs the matching pool via `pack_paged_cache(v_col=K.V_COL)`; a kernel that sets
  `VCOL` MUST export `V_COL` so the harness packs column-major V.

## Tiling & threadgroup mapping

* One workgroup owns one `(batch, q_head, q-tile)`. The grid is
  `grid = b * nq * ceil(sq / BM)`; `BM` is exported by each kernel.
* **Baseline:** `BM=32`, 1 wave (64 threads). **CK line:** `TILE_BM = NWAVES*32` (=128 at
  the default 4 waves / 256 threads); each wave owns 32 q-rows.
* **Diagonal-pair (lever 6, `DIAG`):** a CTA also processes the causal mirror tile
  `num_q_tiles-1-t`, so `BM` exported to the grid divisor becomes `2*TILE_BM` and the grid
  halves. Balances light early vs heavy late causal tiles.
* **Outer KV tile (`KT`, CK `kN0`):** the kv axis is streamed in `KT`-key tiles
  (`NSUB = KT/BN` MFMA subtiles per cooperative load + barrier). `KT=32` large-seq optimum;
  `KT=64` helps small seq, regresses large (lever 13 dispatches per shape).
* **XCD remap (lever 11, `XCD`):** logical block ids are permuted so `XCD_C=4` consecutive
  blocks (ordered q-head-fast ⇒ all GQA heads of a tile share the same K/V range) land on one
  XCD's L2.

## Per-threadgroup steps (one q-tile)

Lane layout: `q_local = lane % 32` (the 32 q-rows of a wave), `half = lane // 32` (splits the
MFMA's 64-lane operand). Online-softmax state per lane: running max `m_run`, running
denominator `l_run`, and the output accumulator `o_acc[DT]` (`DT = HD/32 = 4` f32×16 tiles).

**Prologue (once):**
1. Decode `(batch, qhead, qtile)` from `block_idx`; `kvhead = qhead // gqa`.
2. Load this batch's page range start `page0 = LTP[batch]`.
3. Load Q rows for the tile into registers as `KSTEPS = HD/16 = 8` fp8 i64 operands
   (`q_i64[ks]`), reused across the whole kv loop (CK `kQLoadOnce`).
4. Load scalar descales: `q_descale` (per token-head), `v_descale` (per head),
   `p_scale` → `log2_pscale = log2(p_scale)`.
5. Init `m_run = -inf`, `l_run = 0`, `o_acc = 0`.

**KV loop (streaming, over `n_kv = ceil(sk / BN)` subtiles; CK splits masked vs unmasked):**
For each kv subtile:
1. **Page-resolve** the 32 keys of the subtile (`slot/phys/intra` as above).
2. **GEMM1 `S = K @ Qᵀ`** → S as `[kv, q]`: `KSTEPS` MFMAs of
   `mfma_f32_32x32x16_fp8_fp8` accumulating over the 128 features (16 per step), giving each
   lane 16 f32 score slots.
3. **Descale + causal mask:** `s = S * (q_descale * k_descale * sm_scale)`; mask
   `kv <= qrow + (sk - sq)` for causal (interior/unmasked tiles skip this — lever 5);
   masked slots → `-inf`.
4. **Online softmax (register-resident):**
   * row max `m_loc` over the 16 slots, then across `half` via `shuffle_xor(32)`;
     `m_new = max(m_run, m_loc)` (lever 12 folds this into `v_max3`).
   * correction `corr = exp2((m_run - m_new) * LOG2E)` (guarded against all-masked `-inf`).
   * `p = exp2((s - m_new) * LOG2E + log2_pscale)` via fast `rocdl.exp2` (lever 4; lever 10
     folds `LOG2E` into the descale so the score domain is already log2).
   * `l_loc = Σ p` (cross-`half`); `l_run = l_run * corr + l_loc`.
5. **Transpose P for GEMM2:**
   * **baseline:** pack `p` to fp8 (`cvt_pk_fp8_f32`) and store q-major into the LDS P
     scratch `p_lds[q_local*BN + kv_in_tile]`, `gpu.barrier()`, reload as the MFMA B operand.
   * **8wave+ (lever 3):** P stays in registers, transposed via `ds_bpermute` — no LDS P
     scratch.
6. **Rescale running O:** `o_acc[dt] *= corr`.
7. **GEMM2 `O += Vᵀ @ P`** → O as `[d, q]`: for each of `DT=4` d-tiles, 2 MFMA k-steps of 16
   over `BN=32` kv. A operand = V (column-V ⇒ contiguous, no transpose; lever 7); B operand =
   the transposed P. `gpu.barrier()`.

**Epilogue (once):**
1. `inv_l = 1 / l_run` (guarded: `l_run≈0` ⇒ emit 0, not NaN — covers query rows with no
   valid keys when `sk < sq`).
2. `O[d,q] *= v_descale * inv_l`; cast to bf16.
3. Store to `O[batch, qrow, qhead, d]` (bounds-guarded for the padded tail rows).

## Numerical / correctness contract (what the parity gate guards)

* Streaming softmax must rescale `l_run` and `o_acc` by `corr` every subtile; the all-masked
  `m_new == -inf` guard (use exponent 0 ⇒ `p=0`, `corr=0`) prevents NaN on fully-masked rows.
* `p_scale` must be divided back out in the epilogue exactly as `log2(p_scale)` was added to
  the exp bias, or O drifts.
* The causal offset is `kv <= qrow + (sk - sq)` (right-aligned causal for `sk ≥ sq`).
* The V descale is **per-head**, folded once in the epilogue — applying it per-element changes
  rounding and fails parity.
* `V_COL` export MUST match the kernel's V addressing; a column-V kernel fed a row-major pool
  (or vice-versa) reads transposed V and fails.

> The exact GEMM1/GEMM2 MFMA operand packing, the `ds_bpermute` transpose indices, the
> LDS padding strides (lever 8), and the kdlds / log2dom VALU folds (levers 9–12) are in the
> kernel sources — start from `kernels/fmha_prefill_fp8.py` (clearest, register-P-via-LDS
> baseline) and `kernels/fmha_prefill_fp8_ck_log2dom.py` (the measured-peak variant).
