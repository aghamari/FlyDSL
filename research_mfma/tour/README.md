<!-- SPDX-License-Identifier: Apache-2.0 -->
# A tour of the FlyDSL tiled-MMA / tiled-copy instructions

Tiny probe scripts (like `probe_tmma.py`) that build one object each and **print what it is**,
so you can see the layouts instead of guessing. They do no real compute — the prints fire at
trace time. Run any of them on a GPU:

```
HIP_VISIBLE_DEVICES=2 python3 research_mfma/tour/0X_name.py
```

| file | instruction | what it prints / the takeaway |
|---|---|---|
| `01_mma_atom.py` | `make_mma_atom(MFMA(...))` | one instruction + its fixed per-wave operand layouts `layout_{A,B,C}_tv` (the lane↔element map) |
| `02_tiled_mma.py` | `make_tiled_mma(atom, atom_layout)` | atom replicated over waves: `tile_size_mnk`, `thr_layout_vmnk` (thread→wave), `tv_layout_{A,C}_tiled` — compare `(1,1,1)` vs `(2,2,1)` |
| `03_copy_atom.py` | `make_copy_atom` + `make_tiled_copy_A` | one copy instruction (bits/lane) + the A‑matched `TiledCopy` (its `tile_mn`, `layout_tv_tiled`, src/dst layouts derived from the MMA) |
| `04_partition_fragment_retile.py` | `partition_S` / `make_fragment_A` / `retile` | the three per‑lane objects side by side, in a real kernel |

## What tour 04 shows (the confusing trio)

```
thr_gA = tcA.partition_S(gA)     -> Tensor<bf16, buffer_desc, ((4,1),1,1,?):((1,0),0,0,16)>   # SOURCE view (global)
frag_A = thr_mma.make_fragment_A -> Tensor<bf16, register,    (4,1,1):(1,0,0)>                 # REGISTERS (MMA layout)
retile = tcA.retile(frag_A)      -> Tensor<bf16, register,    ((4,1),1,1):((1,0),0,0)>         # frag re-viewed in COPY layout
```

- `partition_S` = *where this lane reads* (a global buffer view; the trailing `?` is the K-step dim, stride 16).
- `make_fragment_A` = *this lane's registers* in the **MMA** layout `(4,1,1)`.
- `retile` = the **same registers** re-grouped as `((4,1),1,1)` to match the **copy**, so `fx.copy` can fill them.

## The pipeline these build up to

```
make_buffer_tensor -> flat_divide/slice        # pick this block's tile
make_mma_atom -> make_tiled_mma                 # (01, 02) the compute plan + layouts
  .thr_slice(tid) -> make_fragment_A/B/C        # (04) this lane's registers
make_copy_atom -> make_tiled_copy_A/B/C         # (03) the copy plan, matched to the MMA
  .get_slice(tid) -> partition_S + retile       # (04) this lane's source view + fragment re-view
fx.copy (load) ; fx.gemm (MFMA) ; fx.copy (store)
```

See `../flydsl_gemm_vs_ck.md` for how each of these maps to the raw-HIP and CK-Tile equivalents,
and `../minimal_tiled_gemm.py` / `../gemm_2x2wave.py` for the full runnable kernels.
