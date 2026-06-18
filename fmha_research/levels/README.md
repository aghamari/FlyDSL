<!-- SPDX-License-Identifier: Apache-2.0 -->
# `levels/` — per-level kernel sources

In the `fused_mega_moe` analog this folder holds curated `level_NN_<name>.py` **snapshots**
for the structural levels (because that kernel's early atoms can't be flag-toggled back —
e.g. the K=32 path hangs comgr). 

**For the FMHA prefill kernel there are no snapshot files**, and that is intentional: every
one of the 14 levels is reachable from an **existing production module in `../../kernels/`
plus `FMHA_*` env overrides**. The lineage was built as a chain of whole files
(`fmha_prefill_fp8` → `_8wave` → `_v7` → `_ck` → `_ck_hk5` → `_combined` → `_ck_log2dom`),
and the late-stage levers are also exposed as default-on env flags
(`FMHA_DIAG/VCOL/KPAD/VPAD/XCD/KT`). So the cleanest A/B is "previous module" vs "this module"
(or a flag toggle on one module), not a duplicated snapshot.

Duplicating a kernel here would risk drifting from the production source, so we **reference the
module by name** in `reproduce_levels.LEVELS` instead. The level → module/env mapping:

| # | lever | family | module (`kernels/…`) | env override / A-B |
|---|-------|--------|----------------------|--------------------|
| 0 | baseline naive fp8 | baseline | `fmha_prefill_fp8` | — |
| 1 | multiwave BM128 4-wave | throughput | `fmha_prefill_fp8_8wave` | `FMHA_NWAVES=4` (vs 2/8) |
| 2 | cooperative K/V→LDS | structural | `fmha_prefill_fp8_8wave` | embedded (no flag) |
| 3 | register-P ds_bpermute | throughput | `fmha_prefill_fp8_8wave` | embedded (no flag) |
| 4 | fast exp2 | throughput | `fmha_prefill_fp8_8wave` | embedded (no flag) |
| 5 | causal masked/unmasked split | structural | `fmha_prefill_fp8_8wave` | embedded (no flag) |
| 6 | diagonal-pair tiling | structural | `fmha_prefill_fp8_v7` | vs `_8wave`, or `FMHA_DIAG=0/1` |
| 7 | column-V (delete transpose) | structural | `fmha_prefill_fp8_ck` | vs `_v7`, or `FMHA_VCOL=0/1` |
| 8 | LDS row padding | throughput | `fmha_prefill_fp8_ck_hk5` | vs `_ck`, or `FMHA_KPAD/VPAD=0/8` |
| 9 | kdlds (K-descale→LDS) | throughput | `fmha_prefill_fp8_combined` | vs `_ck_hk5` |
| 10 | LOG2E-descale + exp-bias hoist | throughput | `fmha_prefill_fp8_ck_log2dom` | vs `_combined` |
| 11 | XCD chiplet remap | dispatch | `fmha_prefill_fp8_ck_log2dom` | `FMHA_XCD=0/1` (C=4) |
| 12 | softmax VALU fold | throughput | `fmha_prefill_fp8_ck_log2dom` | baked in (no flag) |
| 13 | per-seqlen (KT,DIAG) dispatch | dispatch | `fmha_prefill_fp8_ck_log2dom` | per-seqlen `FMHA_KT`/`FMHA_DIAG` |

If a future lever is ever NOT reachable from an existing module + flag (e.g. an atom that
can't be toggled back), drop a `level_NN_<name>.py` snapshot here and load it with
`_build_by_path.load_snapshot(path, unique_smem_sym)` so it won't collide with the production
kernel's module-global `SmemAllocator` symbol.
