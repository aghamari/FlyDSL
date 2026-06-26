# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 — HAND-SCHEDULED inline-asm kernel.

Goal: beat CK-Tile by porting the PyISA / CK qr_ks_vs_async SCHEDULE (not just the tile config):
depth-N async K/V ring over XOR-swizzled LDS + a hand-scheduled hot loop (partial s_waitcnt,
s_setprio, sched_barrier, s_nop, cross-tile QK(i+1)//PV(i) overlap) that the 0.2.0 compiler will
not auto-produce. Built in staged levers, each behind an env flag so it falls back to the proven
hk5 behavior:

    FMHA_HSCHED  (0)  Stage C: depth-3 LDS ring + cross-tile software pipeline with the next tile's
                      GEMM1 hand-WOVEN into this tile's exp2 loop (so the matrix unit runs the next
                      QK matmul during the softmax VALU).

Stage 0 (HSCHED=0) == fmha_prefill_fp8_ck_hk5 exactly (same compute core: 32x32x16 MFMA, column-V
no-transpose, register-P ds_bpermute, fast exp2, per-token-head Q/K + per-head V descale, p_scale,
causal-skip, diagonal-pair, XCD remap). Identical run_attn signature -> harnesses drop-in.

★ MEASURED RESULT (2026-06-19, MI308X, device-fair graph-replay), sq16384/32768:
    HSCHED=0 (== hk5):                 129 / 142 TF   (VGPR 157, 3 waves/SIMD)
    HSCHED=1 (cross-tile pipeline):    105 / 115 TF   (VGPR 194, 2 waves/SIMD)   <- REGRESSES
CK-Tile fp8 reference: 141 / 146 TF.

WHY IT REGRESSES (the register-allocation wall, reconfirmed): software-pipelining REQUIRES carrying
the current tile's GEMM1 accumulator `sv` (16 VGPR) across the loop boundary while the next tile's
`sv_next` is being built -> VGPR 157->194, occupancy 3->2 waves/SIMD. The cross-tile MFMA/VALU
overlap (verified correct, ERR 0.039; tried naive, hand-woven, and just-in-time-K-load variants) does
NOT recover the lost wave. The depth-3 ring removes the LDS hazards, but cannot remove the carried-
accumulator VGPR cost. This is exactly lesson 19's outcome: PyISA/CK win this pipeline only because
hand-asm / C++ codegen PIN registers to hold occupancy while pipelining -- a control the 0.2.0 Python
wheel does not expose (maxnreg/waves_per_eu are dropped; see skill flydsl-fmha-prefill-opt). Kept as a
documented, gated-OFF negative result; default (HSCHED=0) is the safe hk5 equivalent.

Baselines: FlyDSL 26/55/129/142 ; CK-Tile 30/62/141/146 TF @ sq 1024/2048/16384/32768.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, memref
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
DIAG = int(os.environ.get("FMHA_DIAG", "1")) != 0
BM = (2 * TILE_BM) if DIAG else TILE_BM

KT = int(os.environ.get("FMHA_KT", "32"))
assert KT % BN == 0, "FMHA_KT must be a multiple of 32"
NSUB = KT // BN
NBUF = int(os.environ.get("FMHA_NBUF", "2"))  # LDS alloc depth (loop is 2-deep ping-pong)
BUFK = int(os.environ.get("FMHA_BUFK", "0")) != 0

# --- staged levers (Stage 0 defaults == hk5) ---
SWIZZLE = int(os.environ.get("FMHA_SWIZZLE", "0")) != 0  # Stage A
HSCHED = int(os.environ.get("FMHA_HSCHED", "0")) != 0    # Stage C: cross-tile software pipeline
# Lazy rescale (from flash_attn_gfx950): on tiles where the running max is unchanged (corr==1 for
# every lane), skip the 64-wide o_acc rescale + l_run*corr. A wave ballot makes the skip uniform.
LAZY = int(os.environ.get("FMHA_LAZY", "0")) != 0
# Stage C uses a depth-3 LDS ring (3 distinct buffers) so that at any iter the three live tiles
# (i: GEMM2/V, i+1: GEMM1/K, i+2: store) never alias -> no write-after-read hazard, one barrier/iter.
if HSCHED and NBUF < 3:
    NBUF = 3

VCOL = int(os.environ.get("FMHA_VCOL", "1")) != 0
V_COL = VCOL

NXCD = int(os.environ.get("FMHA_NXCD", "4"))
XCD_REMAP = int(os.environ.get("FMHA_XCD", "1")) != 0
XCD_C = int(os.environ.get("FMHA_XCD_C", "4"))

NSLOT = KT * 8
KVG = KT // 16
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_hs_smem")
_K_PAD = int(os.environ.get("FMHA_KPAD", "8"))  # bytes of pad per K LDS row
_V_PAD = int(os.environ.get("FMHA_VPAD", "8"))  # bytes of pad per V LDS row
_K_LDSW = HD + _K_PAD  # K LDS row width
_V_LDSW = KT + _V_PAD  # V LDS row width
_K_BYTES = KT * _K_LDSW  # K tile [KT kv x (HD+pad)]
_V_BYTES = HD * _V_LDSW  # V tile [HD d x (KT+pad)]
_K_OFF = 0
_V_OFF = _K_OFF + NBUF * _K_BYTES
_alloc.ptr = _V_OFF + NBUF * _V_BYTES


def const_expr(x):
    return fx.const_expr(x)


_LGKMCNT0 = 0xC07F


def _wait_lds():
    fx.rocdl.s_waitcnt(_LGKMCNT0)


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
):
    tid = fx.Int32(fx.thread_idx.x)
    wave_id = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)
    blk = fx.Int32(fx.block_idx.x)
    gqa = nq // nk
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)

    num_q_tiles = (sq_i + fx.Int32(TILE_BM - 1)) // fx.Int32(TILE_BM)
    if const_expr(DIAG):
        num_first = (num_q_tiles + fx.Int32(1)) // fx.Int32(2)
    else:
        num_first = num_q_tiles
    if const_expr(XCD_REMAP):
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
    v_tok_stride = fx.Int32(nk * HD)

    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)
    v_head_off = kvhead * fx.Int32(HD)
    v_head_off_col = kvhead * fx.Int32(HD * page_size)
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
    _ar = fx.arith.unwrap

    def _fmax(a, b):
        return fx.Float32(arith.maxnumf(_ar(a), _ar(b)))

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NBUF * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NBUF * _V_BYTES,)).get()
    if const_expr(BUFK):
        k_lds_base = memref.extract_aligned_pointer_as_index(k_lds)
        k_lds_ptr_base = fx.buffer_ops.create_llvm_ptr(arith.index_cast(fx.typing.T.i64, k_lds_base), address_space=3)

    pass_valid = []
    pass_kv = []
    pass_cg = []
    pass_dv = []
    pass_kvg = []
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
                kvg0 = kv0_ + pass_kvg[p] * fx.Int32(16)
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

        def gemm1_tile(kbuf):
            """GEMM1: S[kv,q] = K@Q^T for all NSUB subtiles, reading K from LDS buffer kbuf."""
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
            return sv

        def soft_gemm2(sv, kv0_outer, vbuf, m_run, l_run, o_acc, do_mask, kbuf_next=None):
            """Softmax(sv) + GEMM2: O[d,q] += V^T@P, reading V from LDS buffer vbuf.

            If kbuf_next is given, the next tile's GEMM1 (K@Q^T) MFMAs are WOVEN into this tile's
            exp2 loop so the matrix unit runs the next QK matmul during this tile's softmax VALU
            (the cross-tile overlap). Returns (m_new, l_run, o_acc, sv_next) with sv_next the next
            tile's GEMM1 result (None when not woven).
            """
            qs = q_descale * fx.Float32(sm_scale * LOG2E)
            s_all = []
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

            # Woven next-tile GEMM1: preload K(t+1) from LDS, then drip its MFMAs through the exp2.
            # Precompute a static schedule (one entry per global exp-iter) so no runtime while-drain
            # is needed -- the NSUB*KSTEPS MFMAs always fit in the NSUB*16 exp2 iters (stride==2).
            # Just-in-time K(t+1) load inside the weave keeps only ~1 K pack live (vs 16 VGPR for
            # all 8), to stay under the 170-VGPR / 3-waves-per-SIMD occupancy cliff.
            weave = kbuf_next is not None
            acc_next = []
            n_exp = NSUB * 16
            sched = [None] * n_exp
            if weave:
                acc_next = [fx.Vector.filled(16, 0.0, fx.Float32).ir_value() for _ in range(NSUB)]
                actions = [(sub, ks) for sub in range(NSUB) for ks in range(KSTEPS)]
                stride = max(1, n_exp // max(1, len(actions)))
                for j, a in enumerate(actions):
                    sched[min(j * stride, n_exp - 1)] = a

            l_loc = fx.Float32(0.0)
            p_all = []
            _ei = 0
            for sub in fx.range_constexpr(NSUB):
                p_sub = []
                for i in fx.range_constexpr(16):
                    p = fx.Float32(fx.rocdl.exp2(f32t, _ar(s_all[sub][i] - safe_m_p)))
                    p_sub.append(p)
                    l_loc = l_loc + p
                    act = sched[_ei]
                    if act is not None:
                        sa, ka = act
                        e = kbuf_next + (fx.Int32(sa * BN) + kv_local) * fx.Int32(_K_LDSW) + fx.Int32(ka * 16) + half * fx.Int32(8)
                        kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(e)])
                        a_raw = fx.Vector(kv8).bitcast(fx.Int64)[0].ir_value()
                        b_raw = q_i64[ka].ir_value() if hasattr(q_i64[ka], "ir_value") else q_i64[ka]
                        acc_next[sa] = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_next[sa], 0, 0, 0).res
                    _ei += 1
                p_all.append(p_sub)
            sv_next = [fx.Vector(acc_next[sub]) for sub in range(NSUB)] if weave else None
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
            if const_expr(LAZY):
                # Skip the rescale uniformly when no lane's max rose this tile (corr==1 everywhere).
                stable = m_loc <= m_run
                true_i1 = fx.Boolean(True).ir_value()
                exec_mask = fx.Int64(fx.rocdl.ballot(fx.typing.T.i64, true_i1))
                bal = fx.Int64(fx.rocdl.ballot(fx.typing.T.i64, _ar(stable)))
                all_stable = bal == exec_mask
                if all_stable:
                    l_run = l_run + l_loc
                else:
                    l_run = l_run * corr + l_loc
                    corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
                    for dt in fx.range_constexpr(DT):
                        o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec
            else:
                l_run = l_run * corr + l_loc
                corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
                for dt in fx.range_constexpr(DT):
                    o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

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
            return m_new, l_run, o_acc, sv_next

        def compute_kt_tile(kv0_outer, kbuf, vbuf, m_run, l_run, o_acc, do_mask):
            sv = gemm1_tile(kbuf)
            m_new, l_run, o_acc, _ = soft_gemm2(sv, kv0_outer, vbuf, m_run, l_run, o_acc, do_mask)
            return m_new, l_run, o_acc

        def loop_body(kt_iv, m_run, l_run, o_acc, do_mask):
            kv0_outer = fx.Int32(kt_iv) * fx.Int32(KT)
            cur_buf = fx.Int32(kt_iv) % fx.Int32(2)
            kbuf = cur_buf * fx.Int32(_K_BYTES)
            vbuf = cur_buf * fx.Int32(_V_BYTES)
            nxt_buf = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
            kbuf_n = nxt_buf * fx.Int32(_K_BYTES)
            vbuf_n = nxt_buf * fx.Int32(_V_BYTES)
            kc_w_next, vc_w_next = load_kv_regs(kv0_outer + fx.Int32(KT))
            fx.rocdl.s_setprio(1)
            m_run, l_run, o_acc = compute_kt_tile(kv0_outer, kbuf, vbuf, m_run, l_run, o_acc, do_mask)
            fx.rocdl.s_setprio(0)
            store_kv_to_lds(kc_w_next, vc_w_next, kbuf_n, vbuf_n)
            if const_expr(BUFK):
                _wait_vmem()
            fx.gpu.barrier()
            return m_run, l_run, o_acc

        if const_expr(not HSCHED):
            kc_w0, vc_w0 = load_kv_regs(fx.Int32(0))
            store_kv_to_lds(kc_w0, vc_w0, fx.Int32(0), fx.Int32(0))
            if const_expr(BUFK):
                _wait_vmem()
            fx.gpu.barrier()

            init_state = [m_run0, l_run0] + o_acc0
            for kt_iv, st in range(fx.Index(0), fx.Index(n_unmask), fx.Index(1), init=init_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, False)
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

            mid_state = [st[0], st[1]] + [st[2 + d] for d in range(DT)]
            for kt_iv, st in range(fx.Index(n_unmask), fx.Index(n_kt_rt), fx.Index(1), init=mid_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, True)
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
        else:
            # ---- Stage C: depth-3 ring cross-tile software pipeline ----
            # At iter t the three live tiles never alias mod NBUF(=3): t (GEMM2/V), t+1 (GEMM1/K
            # just-read), t+2 (store target). GEMM1(t+1) is data-independent of softmax(t), so
            # emitting it right before soft_gemm2(t) lets the MFMA unit run the next tile's QK
            # matmul during this tile's softmax VALU. load_kv_regs clamps OOB rows, so the t+1/t+2
            # prefetch+store stay memory-safe past the end and their results are simply unused.
            NB = fx.Int32(NBUF)
            KTi = fx.Int32(KT)
            KB = fx.Int32(_K_BYTES)
            VB = fx.Int32(_V_BYTES)

            def _bk(t):
                return (t % NB) * KB

            def _bv(t):
                return (t % NB) * VB

            kc_p0, vc_p0 = load_kv_regs(fx.Int32(0))
            store_kv_to_lds(kc_p0, vc_p0, fx.Int32(0) * KB, fx.Int32(0) * VB)
            kc_p1, vc_p1 = load_kv_regs(KTi)
            store_kv_to_lds(kc_p1, vc_p1, fx.Int32(1) * KB, fx.Int32(1) * VB)
            fx.gpu.barrier()
            sv0 = gemm1_tile(fx.Int32(0) * KB)

            init_state = [m_run0, l_run0] + o_acc0 + [sv0[s] for s in range(NSUB)]
            for kt_iv, st in range(fx.Index(0), fx.Index(n_kt_rt), fx.Index(1), init=init_state):
                m_run = st[0]
                l_run = st[1]
                o_acc = [st[2 + d] for d in range(DT)]
                sv = [st[2 + DT + s] for s in range(NSUB)]
                t = fx.Int32(kt_iv)
                kv0_outer = t * KTi
                kc_n, vc_n = load_kv_regs((t + fx.Int32(2)) * KTi)
                fx.rocdl.s_setprio(1)
                # GEMM1(t+1) woven into softmax(t) -> matrix unit busy during the softmax VALU
                m_run, l_run, o_acc, sv_next = soft_gemm2(
                    sv, kv0_outer, _bv(t), m_run, l_run, o_acc, True, kbuf_next=_bk(t + fx.Int32(1))
                )
                fx.rocdl.s_setprio(0)
                store_kv_to_lds(kc_n, vc_n, _bk(t + fx.Int32(2)), _bv(t + fx.Int32(2)))
                fx.gpu.barrier()
                st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)] + [sv_next[s] for s in range(NSUB)]

            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]

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
    # Occupancy control via the rocdl.waves_per_eu FUNCTION ATTRIBUTE (reachable from Python through
    # value_attrs, as flash_attn_gfx950.py does) -- NOT the broken maxnreg/--amdgpu-* CLI path. Forces
    # the compiler to cap VGPR to fit FMHA_WPEU waves/SIMD. 0 = let the compiler choose (default).
    # NOTE (measured 2026-06-19): FMHA_WPEU attaches "amdgpu-waves-per-eu"="N,N" to LLVM IR but the
    # in-process gpu-module->binary backend IGNORES it for regalloc (VGPR stayed 157 for N in {3,4}).
    # So occupancy cannot be forced from Python here -- the skill's "unreachable" is really
    # "reachable-but-ignored-by-the-embedded-backend". Kept env-gated for the record; default off.
    _wpeu = int(os.environ.get("FMHA_WPEU", "0"))
    _va = {}
    if _wpeu:
        _va["passthrough"] = [
            ["amdgpu-waves-per-eu", f"{_wpeu},{_wpeu}"],
            ["amdgpu-flat-work-group-size", f"{NTHREADS},{NTHREADS}"],
        ]

    def _do_launch():
        attn_kernel(
            Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale,
            causal, value_attrs=_va,
        ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)

    # LLVM scheduling/regalloc hints (flash_attn_gfx950 uses these): post-misched off + LSR drop.
    if int(os.environ.get("FMHA_HINTS", "0")):
        with CompilationContext.compile_hints({"llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True}}):
            _do_launch()
    else:
        _do_launch()
