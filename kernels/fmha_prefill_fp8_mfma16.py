# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill — LEVER F3 scaffold: K=32 fp8 MFMA atom (mfma_f32_16x16x32_fp8_fp8).

================================================================================
STATUS: SCAFFOLD ONLY — NOT IMPLEMENTED. ``run_attn`` raises NotImplementedError.
================================================================================

Lever F3 (from the optimization brief): hk5 uses ``fx.rocdl.mfma_f32_32x32x16_fp8_fp8``
(M=N=32, K=16) in BOTH GEMM1 (K@Q^T) and GEMM2 (V^T@P), so the head-dim K-loop runs
KSTEPS = HD/16 = 8 trips. The proposal is to swap to ``mfma_f32_16x16x32_fp8_fp8``
(M=N=16, K=32) to "halve the K-loop trips (8 -> 4) and per-trip overhead".

--------------------------------------------------------------------------------
FEASIBILITY (the gate): the K=32 fp8 atom EXISTS and is reachable.
--------------------------------------------------------------------------------
  * Python wrapper: flydsl/expr/rocdl.py:212  `def mfma_f32_16x16x32_fp8_fp8(...)`
    bound UNGUARDED at flydsl/expr/rocdl.py:135 (`_ods_... = mfma_f32_16x16x32_fp8_fp8`),
    unlike the f16/bf16 16x16x32 siblings which `globals().get(...,None)` -> raise
    AttributeError("gfx950+"). So this fp8 atom is live on gfx942.
  * MLIR op: _mlir/dialects/_rocdl_ops_gen.py:19506
    `class mfma_f32_16x16x32_fp8_fp8` (OPERATION_NAME "rocdl.mfma.f32.16x16x32.fp8.fp8").
  * CDNA3 dispatch table: lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:190.
  * REAL PRODUCTION USE: kernels/mla_fwd_decode_m16x8_fp8_fp8.py (MFMA_K=32,
    MFMA_M=MFMA_N=16, MFMA_ELEM_PER_THR=8) — the lane layout below is taken from it.

So this is NOT the "atom missing" branch. It is feasible.

--------------------------------------------------------------------------------
WHY THIS IS A SCAFFOLD AND NOT A FINISHED KERNEL (the decisive finding)
--------------------------------------------------------------------------------
F3 is a STRUCTURAL NON-WIN for THIS kernel. The atom swap cannot beat hk5; it is
expected to regress. Three independent reasons, all derivable without a GPU:

1) INSTRUCTION COUNT DOUBLES (geometry-independent).
   MAC/instruction:  32x32x16 = 32*32*16 = 16384 ;  16x16x32 = 16*16*32 = 8192.
   The 32x32x16 atom packs 2x the MACs per instruction. Covering the fixed
   S=[KV x Q] @ K=HD problem therefore needs EXACTLY 2x the MFMA instructions with
   the smaller atom, for ANY tiling:
       #MFMA(32x32x16) = (KV*Q/1024) * (HD/16)
       #MFMA(16x16x32) = (KV*Q/256)  * (HD/32) = 2 * #MFMA(32x32x16).
   The "halve K-trips 8->4" is real but fully cancelled: each trip now issues a
   2x2 grid of 16x16 output blocks (4 MFMAs) instead of 1, so net MFMA = 2x.
   On CDNA3 both atoms run at the SAME MAC/cycle (16x16x32 ~16 cyc, 32x32x16 ~32 cyc),
   so total matrix cycles are equal but instruction-issue overhead doubles. Net: worse.

2) WRONG BOTTLENECK. hk5 is VALU/softmax- and VGPR-occupancy-bound (VALU:MFMA ~19:1;
   3 waves/SIMD pinned by 166 VGPR), NOT MFMA-K-trip bound. Shaving K-loop trips
   does not touch the binding constraint. (Same lesson as the measured hk8/hk10/async
   dead-ends: optimize the measured bottleneck, not the plausible one.)

3) SOFTMAX CROSS-LANE REDUCTION GETS WORSE. With 32x32x16, a q column's 32 kv live in
   2 lanes (lane, lane^32) -> ONE butterfly `shuffle_xor(32)`. With 16x16x32 a q
   column's kv are spread across the 4 lanes that share lane%16 (lane//16 in 0..3),
   so the reduction needs TWO butterfly steps `shuffle_xor(32)` then `shuffle_xor(16)`
   (confirmed: mla_fwd_decode `_warp_reduce_max_16` reduces over strides [32,16]).
   That ADDS VALU to the exact bottleneck we are trying to relieve.

The genuine "hero atom" analog (the fused_mega_moe K=128 win) on gfx942 is the
LARGER-MAC atom `mfma_scale_f32_16x16x128_f8f6f4` (16x16, K=128 -> 32768 MAC/inst =
2x the 32x32x16 atom -> HALF the instructions). That is the lever with real ceiling,
*if* its microscaled-fp8 semantics can be made to carry per-token/head descale.
16x16x32 is a SMALLER atom than what hk5 already uses; it is the wrong direction.

This file is therefore left as a precise, ready-to-fill scaffold. The complete
layout rewrite is specified below so a GPU-equipped session can build+verify it if
the bench is still wanted, but it is NOT recommended to spend a bench slot here
before the K=128 scaled atom.

================================================================================
COMPLETE REWRITE SPEC (exact layout deltas vs hk5) — for whoever fills this in
================================================================================
Atom layout (16x16x32 fp8, 64-lane wave), grounded in mla_fwd_decode + lesson_01:
  INPUT A,B  ([MN=16, K=32], 8 fp8/lane = i64):
     lane l -> mn = l % 16 ; k = (l // 16)*8 + e , e=0..7   (4 quarters of the wave
     hold K-slices 0-7 / 8-15 / 16-23 / 24-31 of one mn row).
  OUTPUT C  ([16,16], 4 f32/lane = f32x4):
     lane l -> row = (l // 16)*4 + r , col = l % 16 , r=0..3.
  (Contrast hk5's 32x32x16: A,B mn=l%32, k=(l//32)*8+e ; C 16 f32/lane,
   col=l%32, row = 8*group + 4*half + e for acc index i = 4*group+e.)

Chosen geometry to keep the grid identical to hk5 (BM export unchanged, DIAG intact):
  NWAVES=4, WAVE_ROWS=32 (TILE_BM=128), BN=32 kv/subtile — but each 32x32 S subtile is
  now a 2(kvb) x 2(qb) grid of 16x16 blocks. KSTEPS16 = HD/32 = 4.
  (Alternative: WAVE_ROWS=16/TILE_BM=64 native 16x16 — simpler per-MFMA but halves the
   q-tile and changes BM; still 2x MFMA total. Documented, not preferred.)

Per (kvb in 0,1, qb in 0,1) accumulator acc[kvb][qb] : f32x4, chained over ks=0..3:
  GEMM1 mapping for acc[kvb][qb][r] (r=0..3):
     kv = sub*BN + kvb*16 + (lane//16)*4 + r
     q  = qb*16 + (lane%16)
  K fragment (A): k_lds_elem = kbuf + (sub*BN + kvb*16 + lane%16)*_K_LDSW
                              + ks*32 + (lane//16)*8   ; load vec(8,i8) -> i64.
  Q fragment (B): q row = wave_q0 + qb*16 + (lane%16) ; head = ks*32 + (lane//16)*8
                  -> buffer_load vec_width=2 i32 (8 fp8) -> i64. NOTE Q must now be
                  reloaded/addressed per (qb) AND the head slice is (lane//16)*8 (4-way),
                  not half*8 (2-way) — this changes q_i64 packing from KSTEPS(8) entries
                  to KSTEPS16(4) x QB(2) entries with a 4-way lane//16 head split.

  DESCALE/MASK: per acc element use (kv,q) above. kd load granularity changes: each lane
  now owns kv = kvb*16 + (lane//16)*4 + {0..3} (4 contiguous kv per kvb) -> load kd as
  vec_width=4 at kv base (kvb*16 + (lane//16)*4). q_descale indexed by the per-(qb) q row
  (TWO q rows per lane now, vs one in hk5). p_scale/sm_scale fold unchanged.

  SOFTMAX: reduce kv per q-column with TWO butterflies: shuffle_xor(32) then
  shuffle_xor(16) (combines the 4 lanes sharing lane%16). Do max and sum this way.
  Each lane handles 2 independent q-columns (qb=0,1) -> keep 2 separate m_run/l_run or
  vectorize. o_acc rescale applies per (qb).

  P-TRANSPOSE (GEMM2 input): this is the HIGHEST-RISK part. GEMM2 computes O[d,q] = V^T@P
  with the SAME 16x16x32 atom, so P must be presented as B with [k=kv, n=q] in the 16x16x32
  INPUT layout: lane -> n=q=l%16, k=kv=(l//16)*8+e. After softmax, P currently lives in the
  C-output layout (lane row=kv group, col=q). The ds_bpermute byte-swap that hk5 uses
  (q_byte=q_local*4, q32_byte=(q_local+32)*4 for the 32x32 map) MUST be re-derived for the
  16x16 map: the source lane that holds (kv,q) in C-layout vs the dest lane that needs it in
  B-input-layout differ; compute byte offset = dest_lane*4 where dest_lane encodes
  (q = n = target l%16, kv-slice = target l//16). cvt_pk_fp8 packs 4 f32->i32 as before, but
  the grouping of which 4 P-values share an i32 changes with the new layout.

  GEMM2: V fragment (A, [d=16, kv=32]) lane -> d=l%16 within a 16-d block (dt now in
  HD/16=8 d-blocks instead of DT=HD/32=4), k=kv=(l//16)*8+e. o_acc per d-block is f32x4
  (was f32x16 per 32-d block). EPILOGUE store: O[d,q] with d=(dblk*16)+(lane//16)*4+r,
  q=lane%16 — the 4-wide d-run is now spaced by the lane//16 group; store as bf16
  vec4 per (dblk) like hk5 but with the new d index.

VERIFY ON GPU (per lesson_01 discipline): write known values, read back which lane got
what, assert against torch — for BOTH the GEMM1 C-layout and the P-transpose, BEFORE
trusting throughput. err < 6e-2 (fp8) is the correctness gate.
================================================================================
"""

import os

import flydsl.expr as fx  # noqa: F401  (kept so the import surface matches hk5)

HD = 128

# --- Geometry (documented; see spec above). Kept so a future impl matches hk5's grid. ---
KSTEPS = HD // 32  # 4 GEMM1 K-steps (halved from hk5's 8) — the lever's nominal win.
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32  # q rows per wave (unchanged; processed as 2 q-blocks of 16)
TILE_BM = NWAVES * WAVE_ROWS  # 128 — same q-tile as hk5 so BM export is unchanged
BN = 32  # kv per subtile (processed as 2 kv-blocks of 16)

# Diagonal-pair tiling knob preserved for parity with hk5 (BM grid divisor).
DIAG = int(os.environ.get("FMHA_DIAG", "1")) != 0
BM = (2 * TILE_BM) if DIAG else TILE_BM

KT = int(os.environ.get("FMHA_KT", "32"))
NBUF = int(os.environ.get("FMHA_NBUF", "2"))

# Column-major V (vec_k_col_v): preserved so the ck_check / bench harness packs the
# matching V pool (it reads this export).
VCOL = int(os.environ.get("FMHA_VCOL", "1")) != 0
V_COL = VCOL

# Unique global symbol name, per the brief (would be used by SmemAllocator once built).
GLOBAL_SYM_NAME = "fmha_prefill_fp8_mfma16_smem"

_NOT_IMPLEMENTED_MSG = (
    "fmha_prefill_fp8_mfma16 is a SCAFFOLD ONLY (lever F3, mfma_f32_16x16x32_fp8_fp8). "
    "The K=32 fp8 atom is feasible (exists + reachable) but the swap is a STRUCTURAL "
    "NON-WIN for this kernel: 16x16x32 packs half the MACs/instruction of the 32x32x16 "
    "atom hk5 already uses, so it doubles MFMA instruction count at equal CDNA3 MAC/cycle, "
    "leaves the VALU/softmax+VGPR bottleneck untouched, and worsens the softmax cross-lane "
    "reduction (2 butterflies vs 1). See the module docstring for the full layout rewrite "
    "spec and verification plan. Do not bench before filling this in and GPU-verifying the "
    "lane layouts; prefer the K=128 scaled atom (mfma_scale_f32_16x16x128_f8f6f4) instead."
)


def run_attn(*args, **kwargs):  # noqa: D401 — signature-compatible stub
    """Drop-in stub matching hk5's ``run_attn`` call site; raises until implemented.

    Real signature (preserved for whoever implements this):
      run_attn(Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size,
               k_page_stride, v_page_stride, sm_scale, causal, grid_blocks, stream=...)
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
