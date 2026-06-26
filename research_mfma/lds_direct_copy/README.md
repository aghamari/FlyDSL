# Direct global → LDS copy paths (FlyDSL)

The `lds_bank_conflicts/` examples move data **global → register → LDS** (`s[i,j] = gX[i,j]`).
This folder is about the *other* family: getting data **straight into LDS** without a VGPR
roundtrip (CK's `global_to_lds` / async paths), plus the hardware LDS-transpose.

## Validated here (gfx942 / MI308X)

### `01_buffer_load_to_lds.py` — direct DMA via the raw intrinsic ✅ PASS
`fx.rocdl.buffer_load_to_lds(rsrc, lds_ptr, voffset, size_bytes)` emits
`buffer_load_dword ... lds`: the load lands directly in LDS, no register.

Key facts (validated by `tests/kernels/lds_dma_probe.py`):
- The per-lane LDS destination pointer is **ignored**; hardware writes `LDS[M0 + lane*4]`
  where `M0` is the *uniform* base per wave. Multi-wave blocks must pass each wave's own
  `M0 = base + wave*64*4`; the per-lane spread comes from `voffset` (per-lane byte offset
  into the global buffer).
- Must follow with `s_waitcnt vmcnt(0)` (`0x3F70`) before the `barrier()` so the async DMA
  has landed.

## Available but NOT included as runnable files

I left these out because they either need CDNA4 (won't run on this gfx942 box) or need
exact atom/pointer operand shapes that crash the backend if slightly off. Each already has
a working reference in the repo — use those as the source of truth.

### `BufferCopyLDS` copy atom (CDNA3, gfx942) — layout-API flavor of the same DMA
Same `buffer_load ... lds` hardware op as 01, but issued via `fx.copy(atom, src, dst)`:
```python
dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
# src = slice of a divided global buffer tensor; dst = make_view over an LDS pointer
fx.copy(dma_atom, src, dst, soffset=...)
```
The `dst` LDS `make_view` must have the **exact** element type / layout the 128-bit atom
expects, or LLVM fails with "Do not know how to expand this operator's operand". See the
working setup in `kernels/fp8_gemm_utils.py` (`G2SLoader`) and
`kernels/flash_attn_gfx950.py` (`_buffer_load_lds_128`).

### TDM async tensor copy global → LDS (gfx950)
Descriptor-based bulk DMA, CK's `async_load_tile_packed_lds`:
```python
desc = fx.rocdl.make_tensor_descriptor_2d(global_ptr, lds_memref, ...)
fx.rocdl.tensor_load_to_lds(desc.dgroup0, desc.dgroup1, ...)
fx.rocdl.s_wait_asynccnt(0)
```
Includes gather variants (`make_tensor_gather_descriptor`). See `python/flydsl/expr/rocdl/tdm_ops.py`
and `learn_fmha/lesson_21_neg_async_dma.py`.

### `ds_read_tr` — hardware LDS read-transpose (gfx950 / gfx1250)
The GPU transposes *on the LDS read*, so you skip the swizzle entirely — the direct
hardware answer to `lds_bank_conflicts/`:
```python
tr_atom = fx.make_copy_atom(fx.rocdl.LDSReadTrans16_64b(), ...)   # ds_read_tr16_b64
fx.copy(tr_atom, lds_src, reg_dst)
# or fx.rocdl.lds_transpose_load(result_type, lds_memref, elem_offset, elem_bytes)  # gfx1250
```
See `python/flydsl/expr/rocdl/cdna4.py` (`LDSReadTrans*`) and `__init__.py` (`lds_transpose_load`).
**Not runnable on gfx942** (CDNA4+).

### `cluster_load_async_to_lds` — multicast async load (gfx1250)
`fx.rocdl.cluster_load_async_to_lds(global_ptr, lds_ptr, size_bytes, ...)`.

## Summary

| approach | FlyDSL API | hardware | status |
|---|---|---|---|
| direct DMA (raw) | `buffer_load_to_lds` | gfx942 | ✅ `01_*.py` |
| direct DMA (atom) | `BufferCopyLDS128b` + `fx.copy` | gfx942 | ref: `fp8_gemm_utils.py` |
| TDM async | `make_tensor_descriptor_2d` + `tensor_load_to_lds` | gfx950 | ref: `tdm_ops.py` |
| LDS read-transpose | `LDSReadTrans*` / `lds_transpose_load` | gfx950/1250 | ref: `cdna4.py` |
| multicast async | `cluster_load_async_to_lds` | gfx1250 | ref: `rocdl/__init__.py` |

## Run

```bash
cd FlyDSL
HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_direct_copy/01_buffer_load_to_lds.py
```
