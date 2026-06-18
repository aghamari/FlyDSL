# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill — K=128 scaled atom (mfma_scale_f32_16x16x128_f8f6f4).

================================================================================
STATUS: BLOCKED on target hardware (gfx942 / CDNA3). ``run_attn`` raises.
================================================================================

The plan was to use the K=128 scaled MFMA `mfma_scale_f32_16x16x128_f8f6f4` for
GEMM1 (S = K @ Q^T, full HD=128 contraction in ONE MFMA), unscaled (scales pinned),
keeping hk5's post-MFMA per-token/head descale. This is the "hero-atom" analog of
the fused_mega_moe K=128 win.

FEASIBILITY (the gate) — RESULT: the atom is NOT available on our target HW.
--------------------------------------------------------------------------------
Target HW (per project): **MI308X, gfx942, CDNA3**, fp8 = **e4m3 FNUZ**
(hk5 builds with `SmemAllocator(arch="gfx942")`). The K=128 scaled atom is a
**gfx950 / CDNA4-ONLY** instruction. Evidence in this very tree:

  * docs/kernel_authoring_guide.md:285  ->  "# GFX950 scaled MFMA (MXFP4/FP6/FP8)"
    immediately above the `rocdl.mfma_scale_f32_16x16x128_f8f6f4(...)` example.
  * The dispatch is wired ONLY in lib/Dialect/FlyROCDL/**CDNA4**/MmaAtom.cpp
    (line 157: verify accepts only 16x16x128 / 32x32x64 *Scale*; line 246 emits
    `ROCDL::mfma_scale_f32_16x16x128_f8f6f4`). The **CDNA3** dispatch
    (lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:167-198) tops out at fp8/bf8
    **16x16x32 / 32x32x16** (K<=32). There is NO K=64/K=128, NO *scale* atom on CDNA3.
  * kernels/fp8_gemm_utils.py:209  `Mfma16x16x128` builds it via
    `fx.rocdl.**cdna4**.MFMA_Scale(16,16,128, fx.Float8E4M3FN)` — the CDNA4 namespace.
  * Element type mismatch: the scaled atom only accepts OCP types
    {Float8E4M3FN, E5M2, F6x2, F4} (CDNA4/MmaAtom.cpp:134-151) — NOT the
    **E4M3 FNUZ** that gfx942 (and this kernel) uses.
  * Every consumer is a gfx1250/gfx950 kernel (moe_gemm_2stage_mxscale_gfx1250,
    gemm_fp8fp4_gfx1250, ...). No gfx942 kernel uses it.

The Python wrapper `fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4` IS importable (it is a
wheel-wide ROCDL binding), so this would *look* available — but lowering it for
gfx942 maps to LLVM intrinsic `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`, which
has NO gfx942 instruction-selection pattern (gfx950 / gfx12 only). It would fail at
codegen, or — worse — silently target the wrong format. It cannot run on MI308X.

CORRECTION OF THE PRIOR RECOMMENDATION
--------------------------------------------------------------------------------
The mfma16 (F3) verdict from the previous step suggested this K=128 scaled atom as
"the genuine hero atom for gfx942". That was WRONG: it is a gfx950/CDNA4 instruction.
The fused_mega_moe K=128 win was on gfx950-class HW, not gfx942. On gfx942 the ONLY
fp8 MFMA atoms are 32x32x16 (which hk5 already uses — the larger-MAC one) and
16x16x32 (smaller; the analyzed non-win). There is NO larger-K fp8 atom to climb to
on this hardware. hk5 already uses the widest-MAC fp8 atom gfx942 has.

WHAT THIS WOULD BE ON gfx950 (kept so a CDNA4 port can pick it up)
--------------------------------------------------------------------------------
Atom 16x16x128 f8f6f4 (gfx950): a,b = vector<8xi32> (32 fp8/lane), c = vector<4xf32>
(16x16 output, 4/lane). Scales scaleA/scaleB are E8M0 bytes selected by opselA/opselB
(NOT 1.0 at byte 0: E8M0 byte 0x7F == 2^0 == 1.0; byte 0 == 2^-127 ~= 0). To run
UNSCALED you must pass a scale byte that decodes to 1.0 (0x7F via the opsel-selected
lane), NOT 0 — verify the exact convention against gemm_fp8fp4_gfx1250 before trusting
"scales=0". GEMM1: one MFMA collapses hk5's KSTEPS=8; the 32x32 S subtile becomes a
2x2 grid of 16x16 blocks (4 MFMAs/subtile). Then the same 16x16-output ripples as the
mfma16 scaffold apply (4-way lane//16 input split, f32x4 accumulators, two-butterfly
[shuffle_xor 32 then 16] softmax reduction, P-transpose ds_bpermute re-derivation,
GEMM2 P-handoff, epilogue d-indexing). On gfx942 NONE of this is reachable.

PATH FORWARD on gfx942 (the actual target)
--------------------------------------------------------------------------------
The MFMA axis is exhausted on gfx942: hk5 uses the best available fp8 atom. The
residual ~1.2-1.4x gap to CK-Tile is the VALU/softmax-scheduling + VGPR-occupancy
ceiling already cataloged, which the 0.2.0 wheel does not expose levers for. To use
the K=128 scaled atom at all requires gfx950/MI350 hardware + an FN (not FNUZ) fp8
re-quant + the CDNA4 codegen path.
================================================================================
"""

import os

import flydsl.expr as fx  # noqa: F401  (kept so the import surface matches hk5)

HD = 128

# Geometry that the gfx950 K=128 GEMM1 path WOULD use (kept so BM export is meaningful
# and a CDNA4 port matches hk5's grid). KSTEPS collapses 8 -> 1 with the K=128 atom.
KSTEPS = 1  # full HD=128 contraction in one 16x16x128 MFMA (gfx950 only)
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32
TILE_BM = NWAVES * WAVE_ROWS  # 128 — same q-tile as hk5 so BM export is unchanged
BN = 32

DIAG = int(os.environ.get("FMHA_DIAG", "1")) != 0
BM = (2 * TILE_BM) if DIAG else TILE_BM

KT = int(os.environ.get("FMHA_KT", "32"))
NBUF = int(os.environ.get("FMHA_NBUF", "2"))

VCOL = int(os.environ.get("FMHA_VCOL", "1")) != 0
V_COL = VCOL

GLOBAL_SYM_NAME = "fmha_prefill_fp8_mfma128_smem"

_BLOCKED_MSG = (
    "fmha_prefill_fp8_mfma128 is BLOCKED on the target HW. The K=128 scaled atom "
    "mfma_scale_f32_16x16x128_f8f6f4 is a gfx950/CDNA4 instruction (docs/kernel_authoring_guide.md "
    "labels it 'GFX950 scaled MFMA'; wired only in CDNA4/MmaAtom.cpp via fx.rocdl.cdna4.MFMA_Scale; "
    "accepts OCP fp8 E4M3FN, not the E4M3FNUZ gfx942 uses). Our target is MI308X / gfx942 / CDNA3, "
    "whose fp8 MFMA atoms top out at 32x32x16 (already used by hk5) and 16x16x32. There is no larger-K "
    "fp8 atom on gfx942; this instruction cannot be selected for gfx942. Run on gfx950/MI350 with an "
    "FN-format re-quant to use it. See module docstring for the full evidence and the gfx950 spec."
)


def run_attn(*args, **kwargs):  # noqa: D401 — signature-compatible stub
    """Drop-in stub matching hk5's ``run_attn`` call site; raises (BLOCKED on gfx942).

    Real signature (preserved for a future gfx950 port):
      run_attn(Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size,
               k_page_stride, v_page_stride, sm_scale, causal, grid_blocks, stream=...)
    """
    raise NotImplementedError(_BLOCKED_MSG)
