# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 — hk5 + fast-exp2 + correct max-freeze.

COMBINED VARIANT of ``fmha_prefill_fp8_ck_hk5.py`` (canonical) that activates TWO levers at once,
because they collapse DISJOINT parts of the interior-tile softmax chain:

  * FMHA_FEXP in {0,1,2}: Schraudolph fast-exp2 (shortens the exp BODY). Identical to hk_fexp.
      0 = exact rocdl.exp2; 1 = pure affine; 2 = affine + quadratic mantissa correction (DEFAULT).
  * FMHA_FREEZE in {0,1}: CORRECT max-freeze WITH rollback (removes the rowmax HEAD + the
      o_acc/l_run rescale TAIL on the unmasked interior tiles). NOT the constant-seed probe in
      hk_freeze — this one seeds the pivot from the real tile-0 rowmax and rolls back the rare
      fp8-pack overflow exactly, reconstructed from P (no QK re-run). Ported from the MI350 PyISA
      kernel (_softmax_exp_laccum_frozen_tilemax / _rollback_recover / _softmax_rescale_R).

Max-freeze design (per q-tile, causal-forward tile order is already kv-ascending in hk5):
  1. SEED: the FIRST unmasked interior tile (kt_iv==0) is PEELED and runs EXACT -> sets FA_max
     (= m_run) to the true tile-0 rowmax. No tile reorder (causal alignment + prefetch intact).
  2. FROZEN tiles (1..n_unmask-1): pivot frozen at FA_max, corr==1, so o_acc/l_run are NOT
     rescaled (the win). P = 2^(scale*(S - FA_max)); softmax is pivot-invariant so as long as no
     P exceeds the fp8 (e4m3 FNUZ, max 240) pack ceiling the result is numerically == hk5.
  3. OVERFLOW GATE (after the exp loop, before the fp8 _cvt4 pack): wave-uniform ballot over the
     per-lane max of the PRE-cvt Schraudolph fma FLOAT (FEXP>0; always finite/monotonic, so a
     saturated u32-cvt NaN can never be dropped by maxNum and miss the overflow) or post-exp P
     (FEXP=0). Common case: no lane over cap -> skip recover entirely (a scalar s_cbranch).
  4. RARE ROLLBACK (reconstruct from P, NO QK re-run): clamp P as U32 bits with v_min_u32(2^120)
     (not v_min_f32 -> a saturated cvt is a NaN bit pattern), mx = cross-lane row-max P, mx=max(mx,1),
     delta=1/mx; GATE delta=1 (and the FA_max raise=0) where mx<=cap so non-overflow rows stay
     bit-exact; P*=delta, o_acc*=delta, l*=delta, FA_max += log2(mx). The masked/diagonal loop
     (do_mask=True) ALWAYS stays fully exact (no freeze).

Both knobs are constexpr (read at import). Per Rule 3 set them BEFORE import, one process per cfg.

Tunables (env): FMHA_FEXP (0/1/2, default 2), FMHA_FREEZE (0/1, default 0), plus all hk5 knobs:
FMHA_NWAVES (4), FMHA_KT (32), FMHA_VCOL (1), FMHA_DIAG (1), FMHA_XCD (1; FMHA_XCD_C 4),
FMHA_KPAD/VPAD (8), FMHA_NBUF (2), FMHA_BUFK (0 -- broken in this wheel).
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

HD = 128
KSTEPS = HD // 16  # GEMM1 K-steps over head_dim
DT = HD // 32  # GEMM2 d-tiles
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32  # q rows owned by each wave
TILE_BM = NWAVES * WAVE_ROWS  # q rows per q-tile (128 @ NWAVES=4)
BN = 32  # kv per MFMA subtile (online-softmax granularity)
# Diagonal-pair tiling (CK's causal load-balancer): each CTA does q-tile t AND its causal mirror
# num_q_tiles-1-t, so early (light) and late (heavy) causal tiles share a workgroup. +8-24% at
# sq>=2048 (matches v7/v8). BM (grid divisor exported to tests/bench) = 2*TILE_BM when on.
DIAG = int(os.environ.get("FMHA_DIAG", "1")) != 0
BM = (2 * TILE_BM) if DIAG else TILE_BM

# CK kN0: outer KV tile per cooperative load / barrier. NSUB MFMA subtiles per tile.
# MEASURED: KT>32 regresses (occupancy); KT=32 == baseline. Default KT=32.
KT = int(os.environ.get("FMHA_KT", "32"))
assert KT % BN == 0, "FMHA_KT must be a multiple of 32"
NSUB = KT // BN
NBUF = int(os.environ.get("FMHA_NBUF", "2"))  # LDS alloc depth (loop is 2-deep ping-pong)
# CK async global->LDS for K via buffer_load_to_lds. DISABLED: broken in flydsl 0.2.0 (wrong
# results; the pre-existing v12 that uses it also fails correctness while its VGPR parent v7 passes).
BUFK = int(os.environ.get("FMHA_BUFK", "0")) != 0

# COLUMN-MAJOR V (CK's true vec_k_col_v): V pool [pages, nk, hd, page_size] has kv (the GEMM2
# contraction dim) CONTIGUOUS per (head, d). The cooperative load then copies V->LDS straight
# (one 128-bit store per slot) instead of the 16x ds_write_b8 scatter-transpose the row-major
# path needs. Requires the col-V pool (pack_paged_cache(v_col=True)); harness checks the V_COL export.
VCOL = int(os.environ.get("FMHA_VCOL", "1")) != 0
V_COL = VCOL  # consumed by ck_check.py / bench_fmha_compare.py to pack the matching V pool

# XCD/chiplet block-ID remap (HipKittens Algorithm 1, phase-1 grouping). MI308X = 4 XCDs; HW routes
# physical block b -> XCD (b % NXCD). Invert that round-robin so XCD_C consecutive logical blocks
# land on the SAME XCD's private L2. Default ON with C=4. FMHA_XCD=0 restores the flat grid.
NXCD = int(os.environ.get("FMHA_NXCD", "4"))
XCD_REMAP = int(os.environ.get("FMHA_XCD", "1")) != 0
XCD_C = int(os.environ.get("FMHA_XCD_C", "4"))  # blocks grouped onto one XCD (knob C)

NSLOT = KT * 8  # KT kv x 8 feature-groups of 16 fp8 = 16B slots per tile (same count K and V)
KVG = KT // 16  # column-V: kv-groups-of-16 per d (HD*KVG == NSLOT)
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

# --- Schraudolph fast-exp2 (lever 1) -------------------------------------------------------
# Replace the per-element softmax exp2 with the affine bit-trick 2^x ~= bitcast(u32(round(
# x*AC + bias))). See hk_fexp for the full derivation. FMHA_FEXP: 0=exact rocdl.exp2,
# 1=pure affine, 2=affine + quadratic mantissa correction (DEFAULT).
_FEXP = int(os.environ.get("FMHA_FEXP", "2"))
_EXP2_AC = 8388608.0  # 2^23 (kExp2Scale)
_EXP2_BIAS = 1064866805.0  # kExp2Bias = 127*2^23 - 486411 (RMS-optimal Schraudolph offset)
_EXP2_C0 = 1.041030
_EXP2_A1 = -0.23832 / _EXP2_AC
_EXP2_A2 = 0.23832 / (_EXP2_AC * _EXP2_AC)
_EXP2_MANT = 0x7FFFFF

# --- Correct max-freeze + rollback (lever 2) ----------------------------------------------
# FMHA_FREEZE=1: peel the first unmasked interior tile (exact, seeds FA_max = true tile-0 rowmax),
# then FREEZE FA_max on tiles 1..n_unmask-1 (corr==1 -> NO o_acc/l_run rescale), detecting + rolling
# back the rare fp8-pack overflow from P alone (no QK re-run). Masked diagonal loop stays exact.
_FREEZE = int(os.environ.get("FMHA_FREEZE", "0")) != 0
_FP8_PACK_MAX = 240.0  # e4m3 FNUZ max finite (this kernel's fp8 P-pack ceiling)
_CAP_P = 224.0  # per-row rollback trigger + common-path P ceiling (margin under 240 for the r<=1.041 mode-2 gain)
# Pre-cvt Schraudolph fma-FLOAT threshold whose u32-cvt bitcasts to 224.0 (= int(0x43600000)). The
# gate (FEXP>0) fires on this finite/monotonic fma float, never on the post-cvt P (a far overflow
# saturates the u32-cvt to a NaN bit pattern that maxNum would silently drop, missing the overflow).
_CAP_FMA = 1130364928.0
_GATE_CAP = _CAP_FMA if _FEXP != 0 else _CAP_P
# 2^120 as u32 bits: the rare-path P clamp (v_min_u32, NOT v_min_f32) maps every overflow/inf/NaN
# bit pattern down to 2^120 and leaves in-range P bit-exact (float bits are monotonic for P>=0).
_P_CLAMP_BITS = 0x7B800000

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_ck_hk_fzx_smem")
# HK5 LDS bank-conflict fix via row PADDING (K_PAD=V_PAD=8 swept optimum). Padding affects only the
# LDS buffer strides; global strides keep HD/KT. See hk5 for the full bank-period analysis.
_K_PAD = int(os.environ.get("FMHA_KPAD", "8"))  # bytes of pad per K LDS row
_V_PAD = int(os.environ.get("FMHA_VPAD", "8"))  # bytes of pad per V LDS row
_K_LDSW = HD + _K_PAD  # K LDS row width (bytes/elements, fp8=1B)
_V_LDSW = KT + _V_PAD  # V LDS row width
_K_BYTES = KT * _K_LDSW  # K tile [KT kv x (HD+pad)]
_V_BYTES = HD * _V_LDSW  # V tile [HD d x (KT+pad)]
_K_OFF = 0
_V_OFF = _K_OFF + NBUF * _K_BYTES
_alloc.ptr = _V_OFF + NBUF * _V_BYTES


def const_expr(x):
    return fx.const_expr(x)


def _raw(v):
    # raw ir.Value out of a flydsl wrapper (Float32 / Vector / Int*) or a bare ir.Value.
    return v.ir_value() if hasattr(v, "ir_value") else fx.arith.unwrap(v)


def _scf_if_carry(cond_raw, else_raw, then_fn):
    # Manual scf.if with carried results (the flydsl high-level if/else only threads scalar named
    # vars, not the o_acc/p_all lists we need). else_raw = the common-path raw ir.Values (already
    # built in the parent block); then_fn() builds + returns the rollback raw values, same order/types.
    result_types = [v.type for v in else_raw]
    if_op = scf.IfOp(cond_raw, result_types, has_else=True, loc=ir.Location.unknown())
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        scf.YieldOp(then_fn())
    if len(if_op.regions[1].blocks) == 0:
        if_op.regions[1].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
        scf.YieldOp(list(else_raw))
    return list(if_op.results)


# s_waitcnt lgkmcnt(0): wait for all LDS/scalar ops (vmcnt/expcnt = don't-care).
_LGKMCNT0 = 0xC07F


def _wait_lds():
    fx.rocdl.s_waitcnt(_LGKMCNT0)


# s_waitcnt vmcnt(0): wait for outstanding VMEM (incl. buffer_load_to_lds DMA).
_VMCNT0 = 0x3F70


def _wait_vmem():
    fx.rocdl.s_waitcnt(_VMCNT0)


@flyc.kernel(known_block_size=[NTHREADS, 1, 1])
def attn_kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    Qd: fx.Tensor,
    Kd: fx.Tensor,
    Vd: fx.Tensor,
    LTD: fx.Tensor,  # int32 [total_pages] physical page id per slot
    LTP: fx.Tensor,  # int32 [batch+1] kv_indptr
    Ps: fx.Tensor,  # f32 [batch*nq] per-(batch,qhead) p_scale (1.0 = disabled)
    O: fx.Tensor,
    sq: fx.Int32,
    sk: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    page_size: fx.Constexpr[int],
    k_page_stride: fx.Int32,
    v_page_stride: fx.Int32,
    sm_scale: fx.Constexpr[float],
    causal: fx.Constexpr[int],
):
    tid = fx.Int32(fx.thread_idx.x)
    wave_id = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)
    blk = fx.Int32(fx.block_idx.x)
    gqa = nq // nk
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)

    # Grid mapping. Diagonal-pair: ceil(num_q_tiles/2) CTAs per (batch,qhead); each does a tile
    # and its causal mirror. Non-diag: one CTA per q-tile (== 8wave baseline).
    num_q_tiles = (sq_i + fx.Int32(TILE_BM - 1)) // fx.Int32(TILE_BM)
    if const_expr(DIAG):
        num_first = (num_q_tiles + fx.Int32(1)) // fx.Int32(2)
    else:
        num_first = num_q_tiles
    if const_expr(XCD_REMAP):
        # Invert HW round-robin (blk -> XCD blk%NXCD) so XCD_C consecutive logical ids land on one XCD.
        gdim = fx.Int32(fx.grid_dim.x)
        chunk_span = fx.Int32(NXCD * XCD_C)
        n_full = (gdim // chunk_span) * chunk_span
        xcd = blk % fx.Int32(NXCD)
        local = blk // fx.Int32(NXCD)
        chunk_idx = local // fx.Int32(XCD_C)
        pos = local % fx.Int32(XCD_C)
        lblk_r = chunk_idx * chunk_span + xcd * fx.Int32(XCD_C) + pos
        lblk = (blk < n_full).select(lblk_r, blk)
        qhead = fx.Int32(lblk % fx.Int32(nq))
        tmp = lblk // fx.Int32(nq)
        first_idx = fx.Int32(tmp % num_first)
        batch = fx.Int32(tmp // num_first)
    else:
        first_idx = blk % num_first
        tmp = blk // num_first
        qhead = tmp % fx.Int32(nq)
        batch = tmp // fx.Int32(nq)
    kvhead = qhead // fx.Int32(gqa)

    q_local = lane % fx.Int32(32)
    half = lane // fx.Int32(32)

    rq = fx.buffer_ops.create_buffer_resource(Q)
    rk = fx.buffer_ops.create_buffer_resource(K)
    rv = fx.buffer_ops.create_buffer_resource(V)
    rqd = fx.buffer_ops.create_buffer_resource(Qd)
    rkd = fx.buffer_ops.create_buffer_resource(Kd)
    rvd = fx.buffer_ops.create_buffer_resource(Vd)
    rltd = fx.buffer_ops.create_buffer_resource(LTD)
    rltp = fx.buffer_ops.create_buffer_resource(LTP)
    ro = fx.buffer_ops.create_buffer_resource(O)

    q_tok_stride = fx.Int32(nq * HD)
    v_tok_stride = fx.Int32(nk * HD)  # within a V page (row-major [ps, nk, hd])

    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)  # vec_k: [pages, nk, hd/16, ps, 16]
    v_head_off = kvhead * fx.Int32(HD)  # row-major V: [pages, ps, nk, hd]
    v_head_off_col = kvhead * fx.Int32(HD * page_size)  # column V: [pages, nk, hd, ps]
    v_descale = fx.buffer_ops.buffer_load(rvd, batch * fx.Int32(nk) + kvhead, vec_width=1, dtype=fx.Float32)

    rps = fx.buffer_ops.create_buffer_resource(Ps)
    p_scale = fx.buffer_ops.buffer_load(rps, batch * fx.Int32(nq) + qhead, vec_width=1, dtype=fx.Float32)
    _ps_raw = p_scale.ir_value() if hasattr(p_scale, "ir_value") else p_scale
    log2_pscale = fx.Float32(fx.math.log2(_ps_raw))

    f32x16 = fx.typing.T.vec(16, fx.typing.T.f32)
    neg_inf = fx.Float32(-3.0e38)
    width64 = fx.Int32(64)
    off32 = fx.Int32(32)
    f32t = fx.typing.T.f32
    i32t = fx.typing.T.i32
    i64t = fx.typing.T.i64
    _ar = fx.arith.unwrap

    def _fmax(a, b):
        return fx.Float32(arith.maxnumf(_ar(a), _ar(b)))

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NBUF * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NBUF * _V_BYTES,)).get()
    if const_expr(BUFK):
        k_lds_base = memref.extract_aligned_pointer_as_index(k_lds)
        k_lds_ptr_base = fx.buffer_ops.create_llvm_ptr(arith.index_cast(fx.typing.T.i64, k_lds_base), address_space=3)

    pass_valid = []
    pass_kv = []  # K slot//8 : kv row 0..KT-1
    pass_cg = []  # K slot%8  : feature group 0..7 (16 feats each)
    pass_dv = []  # col-V slot//KVG : head-dim 0..HD-1
    pass_kvg = []  # col-V slot%KVG : kv-group-of-16 index 0..KVG-1
    for p in fx.range_constexpr(NPASS):
        slot = fx.Int32(p * NTHREADS) + tid
        pass_valid.append(slot < fx.Int32(NSLOT))
        slot_s = (slot < fx.Int32(NSLOT)).select(slot, fx.Int32(0))
        pass_kv.append(slot_s // fx.Int32(8))
        pass_cg.append(slot_s % fx.Int32(8))
        pass_dv.append(slot_s // fx.Int32(KVG))
        pass_kvg.append(slot_s % fx.Int32(KVG))

    ps_i = fx.Int32(page_size)
    kv_local = lane % fx.Int32(32)
    sk_m1 = sk_i - fx.Int32(1)
    kd_row_base = (batch * fx.Int32(nk) + kvhead) * sk_i
    is_h0 = half == fx.Int32(0)
    q_byte = q_local * fx.Int32(4)
    q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)
    n_kt_full = (sk_i + fx.Int32(KT - 1)) // fx.Int32(KT)

    def load_kv_regs(kv0_):
        kc = []
        vc_words = []
        for p in fx.range_constexpr(NPASS):
            kvrow = kv0_ + pass_kv[p]
            kvrow_safe = (kvrow < sk_i).select(kvrow, fx.Int32(0))
            kslot = page0 + kvrow_safe // ps_i
            kphys = fx.buffer_ops.buffer_load(rltd, kslot, vec_width=1, dtype=fx.Int32)
            kintra = kvrow_safe % ps_i
            kc_off = kphys * k_page_stride + k_head_off + pass_cg[p] * (ps_i * fx.Int32(16)) + kintra * fx.Int32(16)
            if const_expr(BUFK):
                kc.append(kc_off)
            else:
                kc.append(fx.buffer_ops.buffer_load(rk, kc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32))
            if const_expr(VCOL):
                kvg0 = kv0_ + pass_kvg[p] * fx.Int32(16)  # group base (multiple of 16)
                kvg0_safe = (kvg0 < sk_i).select(kvg0, fx.Int32(0))
                vslot = page0 + kvg0_safe // ps_i
                vphys = fx.buffer_ops.buffer_load(rltd, vslot, vec_width=1, dtype=fx.Int32)
                vtok = kvg0_safe % ps_i
                vc_vidx = vphys * v_page_stride + v_head_off_col + pass_dv[p] * ps_i + vtok
            else:
                vc_vidx = kphys * v_page_stride + kintra * v_tok_stride + v_head_off + pass_cg[p] * fx.Int32(16)
            vc_words.append(fx.buffer_ops.buffer_load(rv, vc_vidx // fx.Int32(4), vec_width=4, dtype=fx.Int32))
        return kc, vc_words

    def store_kv_to_lds(kc, vc_words, kbuf_off, vbuf_off):
        for p in fx.range_constexpr(NPASS):
            guard = pass_valid[p] if const_expr(NPASS > 1) else None
            kv_row = pass_kv[p]
            cg = pass_cg[p]
            k_dst = kbuf_off + kv_row * fx.Int32(_K_LDSW) + cg * fx.Int32(16)

            def _do_store():
                if const_expr(BUFK):
                    for d in fx.range_constexpr(4):
                        k_dword_off = fx.Int32(d * 4)
                        k_lds_ptr = fx.buffer_ops.get_element_ptr(k_lds_ptr_base, byte_offset=fx.Index(k_dst + k_dword_off))
                        fx.rocdl.buffer_load_to_lds(rk, k_lds_ptr, kc[p] + k_dword_off, size_bytes=4)
                else:
                    fx.Vector(kc[p]).bitcast(fx.Int8).store(k_lds, [fx.Index(k_dst)])
                if const_expr(VCOL):
                    v_dst = vbuf_off + pass_dv[p] * fx.Int32(_V_LDSW) + pass_kvg[p] * fx.Int32(16)
                    fx.Vector(vc_words[p]).bitcast(fx.Int8).store(vt_lds, [fx.Index(v_dst)])
                else:
                    v_d0 = cg * fx.Int32(16)
                    vc_i8 = fx.Vector(vc_words[p]).bitcast(fx.Int8)
                    for e in fx.range_constexpr(16):
                        fx.Vector.from_elements([fx.Vector(vc_i8)[e]], fx.Int8).store(
                            vt_lds, [fx.Index(vbuf_off + (v_d0 + fx.Int32(e)) * fx.Int32(_V_LDSW) + kv_row)]
                        )

            if guard is not None:
                if guard:
                    _do_store()
            else:
                _do_store()

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    def process_qtile(qtile):
        wave_q0 = qtile * fx.Int32(TILE_BM) + wave_id * fx.Int32(WAVE_ROWS)
        qrow = wave_q0 + q_local
        qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
        q_base = batch * (sq_i * q_tok_stride) + qrow_safe * q_tok_stride + qhead * fx.Int32(HD)

        q_i64 = []
        for ks in fx.range_constexpr(KSTEPS):
            off = q_base + fx.Int32(ks * 16) + half * fx.Int32(8)
            w = fx.buffer_ops.buffer_load(rq, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
            q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])

        qd_idx = (batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe
        q_descale = fx.buffer_ops.buffer_load(rqd, qd_idx, vec_width=1, dtype=fx.Float32)

        if const_expr(causal != 0):
            cb = qrow + (sk_i - sq_i)
            eff_bound = (cb < sk_m1).select(cb, sk_m1)
        else:
            eff_bound = sk_m1

        if const_expr(causal == 0):
            n_kt_rt = n_kt_full
        else:
            q_max = qtile * fx.Int32(TILE_BM) + fx.Int32(TILE_BM - 1)
            kv_max = q_max + (sk_i - sq_i)
            n_kt_caus = (kv_max + fx.Int32(KT)) // fx.Int32(KT)
            n_kt_rt = (n_kt_caus < n_kt_full).select(n_kt_caus, n_kt_full)

        if const_expr(causal == 0):
            lim = sk_i
        else:
            bnd = wave_q0 + (sk_i - sq_i) + fx.Int32(1)
            lim = (bnd < sk_i).select(bnd, sk_i)
        lim = (lim > fx.Int32(0)).select(lim, fx.Int32(0))
        n_unmask = lim // fx.Int32(KT)
        n_unmask = (n_unmask < n_kt_rt).select(n_unmask, n_kt_rt)

        # Process the whole KT tile (NSUB 32-kv subtiles) with the softmax done ONCE over all of it.
        def compute_kt_tile(kv0_outer, kbuf, vbuf, m_run, l_run, o_acc, do_mask, freeze=False):
            # --- GEMM1 for all subtiles: S[kv,q] = K @ Q^T ---
            sv = []
            for sub in fx.range_constexpr(NSUB):
                k_packs = []
                for ks in fx.range_constexpr(KSTEPS):
                    k_lds_elem = kbuf + (fx.Int32(sub * BN) + kv_local) * fx.Int32(_K_LDSW) + fx.Int32(ks * 16) + half * fx.Int32(8)
                    kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
                    k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
                acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
                for ks in fx.range_constexpr(KSTEPS):
                    a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
                    b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
                    acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
                sv.append(fx.Vector(acc_raw))

            # --- descale + causal mask for all subtiles -> s_vals[sub][i] ---
            qs = q_descale * fx.Float32(sm_scale * LOG2E)  # per-lane const (folds sm AND log2e into descale)
            s_all = []  # flat list over subtiles
            for sub in fx.range_constexpr(NSUB):
                kv0 = kv0_outer + fx.Int32(sub * BN)
                kdv = []
                for g in fx.range_constexpr(4):
                    kv_g0 = kv0 + fx.Int32(g * 8) + half * fx.Int32(4)
                    if const_expr(do_mask):
                        kv_g0 = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
                    kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0, vec_width=4, dtype=fx.Float32)))
                s_sub = []
                for i in fx.range_constexpr(16):
                    s = sv[sub][i] * (qs * kdv[i // 4][i % 4])
                    if const_expr(do_mask):
                        kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
                        s = (kv <= eff_bound).select(s, neg_inf)
                    s_sub.append(fx.Float32(s))
                s_all.append(s_sub)

            # --- pivot: FROZEN (seeded interior) vs EXACT (seed tile / masked diagonal) ---
            if const_expr(freeze):
                # corr == 1 (no o_acc/l_run rescale -- the win). pivot stays at the frozen FA_max.
                m_new = m_run
                safe_m = m_run
                safe_m_p = safe_m - log2_pscale
            else:
                m_loc = s_all[0][0]
                for sub in fx.range_constexpr(NSUB):
                    for i in fx.range_constexpr(16):
                        if const_expr(sub == 0 and i == 0):
                            continue
                        m_loc = _fmax(m_loc, s_all[sub][i])
                m_loc = _fmax(m_loc, fx.Float32(m_loc.shuffle_xor(off32, width64)))
                m_new = _fmax(m_run, m_loc)
                m_is_neg = m_new < fx.Float32(-1.0e38)
                safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
                corr = fx.Float32(fx.rocdl.exp2(f32t, _ar(m_run - safe_m)))
                corr = m_is_neg.select(fx.Float32(0.0), corr)
                safe_m_p = safe_m - log2_pscale

            # exp + running-sum. FAST-EXP2 swaps the quarter-rate exp2 for fma + u32-cvt (see hk_fexp).
            if const_expr(_FEXP != 0):
                _AC = fx.Float32(_EXP2_AC)
                _negAC = fx.Float32(-_EXP2_AC)
                bias_eff = fx.Float32(fx.math.fma(_ar(safe_m_p), _ar(_negAC), _ar(fx.Float32(_EXP2_BIAS))))
            if const_expr(_FEXP == 2):
                _C0 = fx.Float32(_EXP2_C0)
                _A1 = fx.Float32(_EXP2_A1)
                _A2 = fx.Float32(_EXP2_A2)
                _MANT = fx.Int32(_EXP2_MANT)
            l_loc = fx.Float32(0.0)
            p_all = []
            g_loc = None  # per-lane max of the overflow-gate quantity (freeze only)
            for sub in fx.range_constexpr(NSUB):
                p_sub = []
                for i in fx.range_constexpr(16):
                    if const_expr(_FEXP != 0):
                        bits = fx.math.fma(_ar(s_all[sub][i]), _ar(_AC), _ar(bias_eff))
                        n = arith.fptoui(i32t, bits)
                        p = fx.Float32(arith.bitcast(f32t, n))
                        if const_expr(_FEXP == 2):
                            fm = fx.Float32(arith.uitofp(f32t, _ar(fx.Int32(n) & _MANT)))
                            inner = fx.Float32(fx.math.fma(_ar(fm), _ar(_A2), _ar(_A1)))
                            r = fx.Float32(fx.math.fma(_ar(inner), _ar(fm), _ar(_C0)))
                            p = p * r
                        if const_expr(freeze):
                            # gate on the PRE-cvt fma float (finite/monotonic; NaN-safe vs post-cvt P).
                            g = fx.Float32(bits)
                    else:
                        p = fx.Float32(fx.rocdl.exp2(f32t, _ar(s_all[sub][i] - safe_m_p)))
                        if const_expr(freeze):
                            g = p
                    p_sub.append(p)
                    l_loc = l_loc + p
                    if const_expr(freeze):
                        g_loc = g if g_loc is None else _fmax(g_loc, g)
                p_all.append(p_sub)
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)

            if const_expr(freeze):
                # Common (no-overflow) frozen update: just accumulate L; o_acc untouched (corr==1).
                l_common = l_run + l_loc
                # Wave-uniform overflow gate: any lane's frozen P over the fp8-pack ceiling?
                pred = g_loc > fx.Float32(_GATE_CAP)
                mask = fx.rocdl.ballot(i64t, fx.arith.unwrap(pred))
                cond = fx.Int64(mask) != fx.Int64(0)
                cond_raw = fx.arith.unwrap(cond)

                else_raw = [_raw(m_run), _raw(l_common)] + [_raw(o_acc[d]) for d in range(DT)]
                for sub in fx.range_constexpr(NSUB):
                    for i in fx.range_constexpr(16):
                        else_raw.append(_raw(p_all[sub][i]))

                def _recover():
                    # RARE path: reconstruct the exact online-softmax rescale from P alone (no QK re-run).
                    # (comprehensions/reduce only -- a plain `for` STATEMENT here is rewritten to scf.for.)
                    clamp = fx.Int32(_P_CLAMP_BITS)

                    def _clamp(pf):
                        pb = arith.minui(arith.bitcast(i32t, _raw(pf)), _raw(clamp))
                        return fx.Float32(arith.bitcast(f32t, pb))

                    p_cl = [[_clamp(p_all[sub][i]) for i in range(16)] for sub in range(NSUB)]
                    flat = [p_cl[sub][i] for sub in range(NSUB) for i in range(16)]
                    pmax_lane = functools.reduce(_fmax, flat)
                    pmax_q = _fmax(pmax_lane, fx.Float32(pmax_lane.shuffle_xor(off32, width64)))
                    pmax_q1 = _fmax(pmax_q, fx.Float32(1.0))
                    delta = fx.Float32(1.0) / pmax_q1
                    over = pmax_q > fx.Float32(_CAP_P)  # per-row gate: bit-exact where mx<=cap
                    delta = over.select(delta, fx.Float32(1.0))
                    dlog = over.select(fx.Float32(fx.math.log2(_ar(pmax_q1))), fx.Float32(0.0))
                    dvec = fx.Vector.filled(16, fx.Float32(delta), fx.Float32)
                    out = [_raw(m_run + dlog), _raw(l_common * delta)]
                    out += [_raw(fx.Vector(o_acc[d]) * dvec) for d in range(DT)]
                    out += [_raw(p_cl[sub][i] * delta) for sub in range(NSUB) for i in range(16)]
                    return out

                res = _scf_if_carry(cond_raw, else_raw, _recover)
                m_new = fx.Float32(res[0])
                l_run = fx.Float32(res[1])
                o_acc = [fx.Vector(res[2 + d]) for d in range(DT)]
                p_all = []
                idx = 2 + DT
                for sub in fx.range_constexpr(NSUB):
                    row = []
                    for i in fx.range_constexpr(16):
                        row.append(fx.Float32(res[idx]))
                        idx += 1
                    p_all.append(row)
            else:
                l_run = l_run * corr + l_loc
                corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
                for dt in fx.range_constexpr(DT):
                    o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

            # --- P transpose (ds_bpermute) for all subtiles ---
            p_i64_all = []
            for sub in fx.range_constexpr(NSUB):
                p_vals = p_all[sub]
                p_i64_s = []
                for s in fx.range_constexpr(2):
                    pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])
                    pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])
                    h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
                    h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
                    h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
                    h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
                    _wait_lds()
                    w0 = is_h0.select(h0_b0, h0_b1)
                    w1 = is_h0.select(h1_b0, h1_b1)
                    p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])
                p_i64_all.append(p_i64_s)

            # --- GEMM2 for all subtiles: O[d,q] += V^T @ P ---
            for sub in fx.range_constexpr(NSUB):
                p_i64_s = p_i64_all[sub]
                v_packs = []
                for dt in fx.range_constexpr(DT):
                    d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
                    for s in fx.range_constexpr(2):
                        v_lds_elem = vbuf + d_col * fx.Int32(_V_LDSW) + fx.Int32(sub * BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                        vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_lds_elem)])
                        v_packs.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
                for dt in fx.range_constexpr(DT):
                    acc2 = fx.Vector(o_acc[dt]).ir_value()
                    for s in fx.range_constexpr(2):
                        v_i64 = v_packs[dt * 2 + s]
                        p_i64 = p_i64_s[s]
                        a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                        b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                        acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
                    o_acc[dt] = fx.Vector(acc2)
            return m_new, l_run, o_acc

        def loop_body(kt_iv, m_run, l_run, o_acc, do_mask, freeze=False):
            kv0_outer = fx.Int32(kt_iv) * fx.Int32(KT)
            cur_buf = fx.Int32(kt_iv) % fx.Int32(2)
            kbuf = cur_buf * fx.Int32(_K_BYTES)
            vbuf = cur_buf * fx.Int32(_V_BYTES)
            nxt_buf = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
            kbuf_n = nxt_buf * fx.Int32(_K_BYTES)
            vbuf_n = nxt_buf * fx.Int32(_V_BYTES)
            kc_w_next, vc_w_next = load_kv_regs(kv0_outer + fx.Int32(KT))  # OPT3 prefetch
            fx.rocdl.s_setprio(1)
            m_run, l_run, o_acc = compute_kt_tile(kv0_outer, kbuf, vbuf, m_run, l_run, o_acc, do_mask, freeze)
            fx.rocdl.s_setprio(0)
            store_kv_to_lds(kc_w_next, vc_w_next, kbuf_n, vbuf_n)
            if const_expr(BUFK):
                _wait_vmem()
            fx.gpu.barrier()
            return m_run, l_run, o_acc

        # Prologue: stage outer tile 0 into LDS buffer 0.
        kc_w0, vc_w0 = load_kv_regs(fx.Int32(0))
        store_kv_to_lds(kc_w0, vc_w0, fx.Int32(0), fx.Int32(0))
        if const_expr(BUFK):
            _wait_vmem()
        fx.gpu.barrier()

        init_state = [m_run0, l_run0] + o_acc0
        if const_expr(_FREEZE):
            # SEED: peel the first unmasked interior tile (exact) so FA_max = the true tile-0 rowmax,
            # then run the rest of the interior FROZEN. n_seed in {0,1} (0 when no interior tiles).
            n_seed = (n_unmask > fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            for kt_iv, st in range(fx.Index(0), fx.Index(n_seed), fx.Index(1), init=init_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, False, False)
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]
            seed_state = [st[0], st[1]] + [st[2 + d] for d in range(DT)]
            for kt_iv, st in range(fx.Index(n_seed), fx.Index(n_unmask), fx.Index(1), init=seed_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, False, True)
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]
            mid_state = [st[0], st[1]] + [st[2 + d] for d in range(DT)]
        else:
            for kt_iv, st in range(fx.Index(0), fx.Index(n_unmask), fx.Index(1), init=init_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, False, False)
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]
            mid_state = [st[0], st[1]] + [st[2 + d] for d in range(DT)]

        # Phase 2: masked tiles (diagonal + any OOB tail) -- ALWAYS exact.
        for kt_iv, st in range(fx.Index(n_unmask), fx.Index(n_kt_rt), fx.Index(1), init=mid_state):
            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
            m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, True, False)
            st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]

        # epilogue: O[d,q] *= v_descale / l_run, cast bf16, store O[b, qrow, qhead, d]
        l_is_zero = l_run < fx.Float32(1.0e-30)
        inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
        scale_o = fx.Float32(v_descale * inv_l)
        scale_vec = fx.Vector.filled(16, scale_o, fx.Float32)
        in_b = qrow < sq_i
        o_row_base = ((batch * sq_i + qrow_safe) * fx.Int32(nq) + qhead) * fx.Int32(HD)
        if in_b:
            for dt in fx.range_constexpr(DT):
                ov = fx.Vector(o_acc[dt]) * scale_vec
                ov_bf16 = fx.Vector(ov).to(fx.BFloat16)
                for j in fx.range_constexpr(4):
                    d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                    v4 = fx.Vector.from_elements([fx.Vector(ov_bf16)[j * 4 + e] for e in range(4)], fx.BFloat16)
                    fx.buffer_ops.buffer_store(v4.ir_value(), ro, (o_row_base + d).ir_value())

    process_qtile(first_idx)
    if const_expr(DIAG):
        mirror_qtile = num_q_tiles - fx.Int32(1) - first_idx
        if mirror_qtile > first_idx:
            process_qtile(mirror_qtile)


@flyc.jit
def run_attn(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    Qd: fx.Tensor,
    Kd: fx.Tensor,
    Vd: fx.Tensor,
    LTD: fx.Tensor,
    LTP: fx.Tensor,
    Ps: fx.Tensor,
    O: fx.Tensor,
    sq: fx.Int32,
    sk: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    page_size: fx.Constexpr[int],
    k_page_stride: fx.Int32,
    v_page_stride: fx.Int32,
    sm_scale: fx.Constexpr[float],
    causal: fx.Constexpr[int],
    grid_blocks: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal
    ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)
