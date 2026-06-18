# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 — SPLIT-KV (flash-decoding) variant.

★ PURPOSE: attack the ONE remaining headroom at SMALL seqlen (sq1024/2048) on MI308X/gfx942 —
GRID UNDER-FILL. The best small-seq kernel ``fmha_prefill_fp8_ck_log2dom`` (KT=64 DIAG=0) hits
26 TF @ sq1024 / 55 @ sq2048 (CK-Tile 30/62). At sq1024 the launch grid is
``b*nq*ceil(sq/BM) = 1*8*8 = 64`` workgroups < 80 CUs, so ~16 CUs sit idle (the VALU/MFMA axes
are exhausted: VGPR 157 -> 3 waves/SIMD, VALU:MFMA ~36:1). This kernel splits each
(head, q-tile)'s causal kv range across ``S = FMHA_SPLITKV`` CTAs (flash-decoding split-KV) to
fill the CUs, then COMBINEs the partials.

This file is a COPY of ``fmha_prefill_fp8_ck_log2dom`` (the current small-seq best — keeps all of
its VALU wins: K-descale-staged-in-LDS ``kdlds``, exp-bias hoist, and LOG2E-folded-into-descale so
the whole score domain is in log2 units) with ONLY these structural changes:

  1. **Split-KV launch.** Grid = ``b*nq*ceil(sq/BM) * S``. The low ``S`` factor of ``block_idx``
     selects the kv-slice; the rest decodes (q-tile, head, batch) exactly as log2dom. Each split
     runs the online softmax over its KT-tile sub-range ``[kt_lo, kt_hi)`` of the runtime causal
     range ``[0, n_kt_rt)`` (the unmasked/masked phase split is intersected with that sub-range),
     producing a PARTIAL flash state (m_partial, l_partial, UNNORMALIZED o_acc).
  2. **Partial scratch.** When ``S>1`` the epilogue writes (m,l,o) to HBM scratch
     ``Ms/Ls [b,nq,sq,S]`` (f32) and ``Os [b,nq,sq,S,HD]`` (f32, unnormalized) instead of the
     normalized bf16 O. ``S==1`` is the EXACT log2dom fast path (normalized bf16 O written direct).
  3. **Cheap combine kernel** (``combine_kernel``, a SECOND lightweight FlyDSL pass, NOT a torch
     multi-op combine): one 32-thread workgroup per (b,head,qrow), each thread owns 4 head-dims.
     Merges the S partials in log2 domain:
         m = max_s m_s ;  e_s = exp2(m_s - m)  (base-2: scores are log2-domain in log2dom) ;
         l = Σ_s e_s·l_s ;  o = (Σ_s e_s·o_s) / l · v_descale  ->  bf16 O.
     (p_scale cancels in o/l, so combine ignores it — same as the inline log2dom epilogue.)

WHY THE OLD ``fmha_prefill_fp8_ck_splitk`` WAS A WASH (+3% @ sq1024) AND WHAT CHANGED HERE:
  * It was built on the PRE-log2dom base: 16/16 LDS padding (vs swept-optimal 8/8), K-descale read
    from VMEM inside the loop (NOT staged in LDS), and natural-domain exp with a per-element *LOG2E
    — i.e. it MISSED all three log2dom VALU wins. -> We rebase on log2dom verbatim.
  * It defaulted to ``S=4`` with ``KT=32`` and FORCED ``DIAG`` off, so it over-subscribed the grid
    (64*4 = 256) and fragmented the already-short causal kv range so unevenly that the light early
    q-tiles' splits did ~0 work while still paying a full Q-load + prologue. -> We default ``S=2``
    (grid 64 -> 128, in the 80-160 sweet spot, half the per-split fixed overhead) and let ``DIAG``
    stay composable (so sq2048 can keep DIAG=1).
  * Its combine round-tripped the large f32 partials through HBM AND ran the merge as SEVERAL
    separate torch kernels (amax/exp/sum/sum/where/mul/permute) whose launch latency dominated a
    sub-0.1 ms attention kernel. -> We do ONE fused FlyDSL combine pass; partials stay small (S=2)
    and ``run_attn`` itself chains main+combine on one stream (no torch in the hot path).

PER-SHAPE DISPATCH (env, compile-time consts — pick the build per seqlen):
    sq1024 : FMHA_KT=64 FMHA_DIAG=0 FMHA_SPLITKV=2  -> grid 1*8*8*2 = 128
    sq2048 : FMHA_KT=64 FMHA_DIAG=1 FMHA_SPLITKV=2  -> grid 1*8*8*2 = 128  (DIAG halves num_q_tiles)

Feature parity with log2dom/hk5 (identical Q/K/V layouts, descales, p_scale, GQA, paged col-V), so
the torch reference + pack helpers are drop-in. The compute per subtile is byte-identical to
log2dom; only the launch + epilogue + combine differ.

GEMMs use ``mfma_f32_32x32x16_fp8_fp8``. GEMM1 = K@Qᵀ (S as [kv,q]); GEMM2 = Vᵀ@P (O as [d,q]).
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

# SPLIT-KV (flash-decoding): split each (head, q-tile)'s causal kv range across SPLITKV CTAs to
# fill the 80 CUs at small seqlen, then combine the partials. SPLITKV==1 is the EXACT log2dom path
# (normalized bf16 O written directly; no scratch, no combine). Default 2: at sq1024 grid goes
# 64 -> 128 (in the 80-160 grid-fill sweet spot) with the lowest per-split fixed overhead.
SPLITKV = int(os.environ.get("FMHA_SPLITKV", "2"))

# Diagonal-pair tiling (CK's causal load-balancer): each CTA does q-tile t AND its causal mirror
# num_q_tiles-1-t. +8-24% at sq>=2048; a loss at sq1024 (grid halving) -> per-shape (off at sq1024).
# UNLIKE ck_splitk we do NOT force DIAG off when splitting: split-KV slices the kv range PER
# q-tile, and the combine is per-qrow, so the mirror tile composes cleanly with the split. This
# lets sq2048 keep DIAG=1 (grid 64*S) instead of paying the un-diagonalized 128*S over-subscribe.
DIAG = int(os.environ.get("FMHA_DIAG", "1")) != 0
BM = (2 * TILE_BM) if DIAG else TILE_BM

# CK kN0: outer KV tile per cooperative load / barrier. NSUB MFMA subtiles per tile.
# KT=64 amortizes softmax VALU better at SMALL seq (the small-seq dispatch default); KT>64 or KT=32
# regress at small seq differently — keep per-shape via env (this kernel targets small seq -> 64).
KT = int(os.environ.get("FMHA_KT", "64"))
assert KT % BN == 0, "FMHA_KT must be a multiple of 32"
NSUB = KT // BN
NBUF = int(os.environ.get("FMHA_NBUF", "2"))  # LDS alloc depth (loop is 2-deep ping-pong)
# CK async global->LDS for K via buffer_load_to_lds. DISABLED: broken in flydsl 0.2.0.
BUFK = int(os.environ.get("FMHA_BUFK", "0")) != 0

# COLUMN-MAJOR V (CK's true vec_k_col_v): kv (GEMM2 contraction) contiguous => no transpose.
VCOL = int(os.environ.get("FMHA_VCOL", "1")) != 0
V_COL = VCOL  # consumed by ck_check.py / bench_fmha_compare.py to pack the matching V pool

NSLOT = KT * 8  # KT kv x 8 feature-groups of 16 fp8 = 16B slots per tile (same count K and V)
KVG = KT // 16  # column-V: kv-groups-of-16 per d (HD*KVG == NSLOT)
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

# Combine pass: 32 threads/wg, each thread owns 4 contiguous head-dims (32*4 == HD). One wg per
# (batch, qhead, qrow). Memory-bound; reads S*(2 + HD) f32, writes HD bf16.
CTHREADS = 32

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_splitkv_smem")
# HK5 LDS bank-conflict fix via row PADDING. 8/8 is the swept optimum (see log2dom/hk5).
_K_PAD = int(os.environ.get("FMHA_KPAD", "8"))  # bytes of pad per K LDS row
_V_PAD = int(os.environ.get("FMHA_VPAD", "8"))  # bytes of pad per V LDS row
_K_LDSW = HD + _K_PAD  # K LDS row width (bytes/elements, fp8=1B)
_V_LDSW = KT + _V_PAD  # V LDS row width
_K_BYTES = KT * _K_LDSW  # K tile [KT kv x (HD+pad)]
_V_BYTES = HD * _V_LDSW  # V tile [HD d x (KT+pad)]
_KD_BYTES = KT * 4  # K descale tile [KT] as f32, ping-ponged with K/V LDS buffers
_K_OFF = 0
_V_OFF = _K_OFF + NBUF * _K_BYTES
_KD_OFF = _V_OFF + NBUF * _V_BYTES
_alloc.ptr = _KD_OFF + NBUF * _KD_BYTES


def const_expr(x):
    return fx.const_expr(x)


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
    Ms: fx.Tensor,  # f32 [batch, nq, sq, SPLITKV] partial per-row running max (split-KV only)
    Ls: fx.Tensor,  # f32 [batch, nq, sq, SPLITKV] partial per-row running sum (split-KV only)
    Os: fx.Tensor,  # f32 [batch, nq, sq, SPLITKV, HD] partial UNNORMALIZED o_acc (split-KV only)
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

    # Grid mapping. Split-KV: the low SPLITKV factor of blk selects the kv-slice; the rest decodes
    # the (q-tile, head, batch) exactly as log2dom. total grid = (nq * num_q_tiles_first) * SPLITKV.
    if const_expr(SPLITKV > 1):
        split = blk % fx.Int32(SPLITKV)
        rest = blk // fx.Int32(SPLITKV)
    else:
        split = fx.Int32(0)
        rest = blk
    num_q_tiles = (sq_i + fx.Int32(TILE_BM - 1)) // fx.Int32(TILE_BM)
    if const_expr(DIAG):
        num_first = (num_q_tiles + fx.Int32(1)) // fx.Int32(2)
    else:
        num_first = num_q_tiles
    first_idx = rest % num_first
    tmp = rest // num_first
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
    if const_expr(SPLITKV > 1):
        rms = fx.buffer_ops.create_buffer_resource(Ms)
        rls = fx.buffer_ops.create_buffer_resource(Ls)
        ros = fx.buffer_ops.create_buffer_resource(Os)

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
    _ar = fx.arith.unwrap

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NBUF * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NBUF * _V_BYTES,)).get()
    kd_lds = SmemPtr(_alloc.get_base(), _KD_OFF, fx.typing.T.f32, shape=(NBUF * KT,)).get()
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

    # Issue the cooperative global loads for a KT-kv tile.
    def load_kv_regs(kv0_):
        kc = []
        vc_words = []
        for p in fx.range_constexpr(NPASS):
            # --- K (kv-major, vectorized) ---
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
            # --- V ---
            if const_expr(VCOL):
                # column V [pages, nk, hd, ps]: load 16 CONTIGUOUS kv for fixed d (GEMM2-ready).
                kvg0 = kv0_ + pass_kvg[p] * fx.Int32(16)  # group base (multiple of 16)
                kvg0_safe = (kvg0 < sk_i).select(kvg0, fx.Int32(0))
                vslot = page0 + kvg0_safe // ps_i
                vphys = fx.buffer_ops.buffer_load(rltd, vslot, vec_width=1, dtype=fx.Int32)
                vtok = kvg0_safe % ps_i
                vc_vidx = vphys * v_page_stride + v_head_off_col + pass_dv[p] * ps_i + vtok
            else:
                # row-major V [pages, ps, nk, hd]: load 16 contiguous d for fixed kv (needs transpose).
                vc_vidx = kphys * v_page_stride + kintra * v_tok_stride + v_head_off + pass_cg[p] * fx.Int32(16)
            vc_words.append(fx.buffer_ops.buffer_load(rv, vc_vidx // fx.Int32(4), vec_width=4, dtype=fx.Int32))
        return kc, vc_words

    def store_kv_to_lds(kc, vc_words, kv0_, kbuf_off, vbuf_off, kdbuf_off):
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
                    # straight contiguous copy into the [d x KT] LDS tile (NO transpose).
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

        # Stage K descales once per KT tile so score scaling consumes LDS instead of late VMEM.
        kd_g = tid
        if kd_g < fx.Int32(KT // 4):
            kv_g0 = kv0_ + kd_g * fx.Int32(4)
            kv_g0_safe = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
            kd_vec = fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0_safe, vec_width=4, dtype=fx.Float32)
            fx.Vector(kd_vec).store(kd_lds, [fx.Index(kdbuf_off + kd_g * fx.Int32(4))])

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    # Process one BM-row q-tile end to end (Q load -> KV loop -> epilogue).
    def process_qtile(qtile):
        wave_q0 = qtile * fx.Int32(TILE_BM) + wave_id * fx.Int32(WAVE_ROWS)
        qrow = wave_q0 + q_local
        qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
        q_base = batch * (sq_i * q_tok_stride) + qrow_safe * q_tok_stride + qhead * fx.Int32(HD)

        # Q LOADED ONCE into registers (CK kQLoadOnce), reused over the whole KV loop.
        q_i64 = []
        for ks in fx.range_constexpr(KSTEPS):
            off = q_base + fx.Int32(ks * 16) + half * fx.Int32(8)
            w = fx.buffer_ops.buffer_load(rq, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
            q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])

        qd_idx = (batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe
        q_descale = fx.buffer_ops.buffer_load(rqd, qd_idx, vec_width=1, dtype=fx.Float32)

        # OPT2: per-lane loop-invariant causal/bounds limit. valid kv iff kv <= eff_bound.
        if const_expr(causal != 0):
            cb = qrow + (sk_i - sq_i)
            eff_bound = (cb < sk_m1).select(cb, sk_m1)
        else:
            eff_bound = sk_m1

        # Causal/total KV bound in outer KT-tile units (skip fully-masked tiles entirely).
        if const_expr(causal == 0):
            n_kt_rt = n_kt_full
        else:
            q_max = qtile * fx.Int32(TILE_BM) + fx.Int32(TILE_BM - 1)
            kv_max = q_max + (sk_i - sq_i)
            n_kt_caus = (kv_max + fx.Int32(KT)) // fx.Int32(KT)
            n_kt_rt = (n_kt_caus < n_kt_full).select(n_kt_caus, n_kt_full)

        # KT tiles fully below the causal diagonal AND fully in-bounds need NO masking.
        if const_expr(causal == 0):
            lim = sk_i
        else:
            bnd = wave_q0 + (sk_i - sq_i) + fx.Int32(1)
            lim = (bnd < sk_i).select(bnd, sk_i)
        lim = (lim > fx.Int32(0)).select(lim, fx.Int32(0))
        n_unmask = lim // fx.Int32(KT)
        n_unmask = (n_unmask < n_kt_rt).select(n_unmask, n_kt_rt)

        # SPLIT-KV kv range: this split owns KT-tiles [kt_lo, kt_hi) of the runtime range
        # [0, n_kt_rt). Even ceil split across SPLITKV workgroups, clamped to n_kt_rt. The
        # phase-split (unmasked interior then masked diagonal) is intersected with [kt_lo, kt_hi).
        if const_expr(SPLITKV > 1):
            per = (n_kt_rt + fx.Int32(SPLITKV - 1)) // fx.Int32(SPLITKV)
            kt_lo = split * per
            kt_lo = (kt_lo < n_kt_rt).select(kt_lo, n_kt_rt)
            kt_hi = kt_lo + per
            kt_hi = (kt_hi < n_kt_rt).select(kt_hi, n_kt_rt)
        else:
            kt_lo = fx.Int32(0)
            kt_hi = n_kt_rt
        # Unmasked sub-range: [kt_lo, min(n_unmask, kt_hi)); masked sub-range: [max(n_unmask, kt_lo), kt_hi).
        um_hi = (n_unmask < kt_hi).select(n_unmask, kt_hi)
        um_hi = (um_hi > kt_lo).select(um_hi, kt_lo)
        mk_lo = (n_unmask > kt_lo).select(n_unmask, kt_lo)
        mk_lo = (mk_lo < kt_hi).select(mk_lo, kt_hi)

        # Process the whole KT tile (NSUB 32-kv subtiles) with the softmax done ONCE over all of it.
        def compute_kt_tile(kv0_outer, kbuf, vbuf, kdbuf, m_run, l_run, o_acc, do_mask):
            # --- GEMM1 for all subtiles: S[kv,q] = K @ Q^T ---
            sv = []
            for sub in fx.range_constexpr(NSUB):
                k_packs = []
                for ks in fx.range_constexpr(KSTEPS):
                    k_lds_elem = kbuf + (fx.Int32(sub * BN) + kv_local) * fx.Int32(_K_LDSW) + fx.Int32(ks * 16) + half * fx.Int32(8)
                    kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
                    k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
                fx.rocdl.sched_dsrd(KSTEPS)
                acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
                for ks in fx.range_constexpr(KSTEPS):
                    a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
                    b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
                    acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
                    fx.rocdl.sched_mfma(1)
                sv.append(fx.Vector(acc_raw))

            # --- descale + causal mask for all subtiles -> s_vals[sub][i] ---
            # Fold LOG2E into the descale -> whole score domain is in log2 units, so the
            # per-element exp arg loses its *LOG2E (mul->nothing) and corr loses its mul.
            qs = q_descale * fx.Float32(sm_scale * LOG2E)  # per-lane const (sm + descale + log2e)
            s_all = []  # flat list over subtiles
            for sub in fx.range_constexpr(NSUB):
                kv0 = kv0_outer + fx.Int32(sub * BN)
                kdv = []
                for g in fx.range_constexpr(4):
                    kd_lds_elem = kdbuf + fx.Int32(sub * BN) + fx.Int32(g * 8) + half * fx.Int32(4)
                    kdv.append(fx.Vector.load(fx.typing.T.vec(4, fx.typing.T.f32), kd_lds, [fx.Index(kd_lds_elem)]))
                s_sub = []
                for i in fx.range_constexpr(16):
                    s = sv[sub][i] * (qs * kdv[i // 4][i % 4])
                    if const_expr(do_mask):
                        kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
                        s = (kv <= eff_bound).select(s, neg_inf)
                    s_sub.append(fx.Float32(s))
                s_all.append(s_sub)

            # --- single softmax over the whole KT tile ---
            m_loc = s_all[0][0]
            for sub in fx.range_constexpr(NSUB):
                for i in fx.range_constexpr(16):
                    if const_expr(sub == 0 and i == 0):
                        continue
                    m_loc = m_loc.maximumf(s_all[sub][i])
            m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
            m_new = m_run.maximumf(m_loc)
            m_is_neg = m_new < fx.Float32(-1.0e38)
            safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
            # scores are already in log2 units (LOG2E folded into qs), so no *LOG2E here.
            corr = fx.Float32(fx.rocdl.exp2(f32t, _ar(m_run - safe_m)))
            corr = m_is_neg.select(fx.Float32(0.0), corr)

            # exp + running-sum (per element), then a single rescale of o_acc.
            # Domain is log2 already: exp arg = s + exp_bias, where exp_bias = log2_pscale - safe_m.
            exp_bias = fx.Float32(log2_pscale) - safe_m
            l_loc = fx.Float32(0.0)
            p_all = []
            for sub in fx.range_constexpr(NSUB):
                p_sub = []
                for i in fx.range_constexpr(16):
                    p = fx.Float32(fx.rocdl.exp2(f32t, _ar(s_all[sub][i] + exp_bias)))
                    p_sub.append(p)
                    l_loc = l_loc + p
                p_all.append(p_sub)
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
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
                fx.rocdl.sched_dsrd(DT * 2)
                for dt in fx.range_constexpr(DT):
                    acc2 = fx.Vector(o_acc[dt]).ir_value()
                    for s in fx.range_constexpr(2):
                        v_i64 = v_packs[dt * 2 + s]
                        p_i64 = p_i64_s[s]
                        a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                        b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                        acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
                        fx.rocdl.sched_mfma(1)
                    o_acc[dt] = fx.Vector(acc2)
            return m_new, l_run, o_acc

        # One outer-tile step (prefetch next tile, compute this tile, store next, barrier).
        def loop_body(kt_iv, m_run, l_run, o_acc, do_mask):
            kv0_outer = fx.Int32(kt_iv) * fx.Int32(KT)
            cur_buf = fx.Int32(kt_iv) % fx.Int32(2)
            kbuf = cur_buf * fx.Int32(_K_BYTES)
            vbuf = cur_buf * fx.Int32(_V_BYTES)
            kdbuf = cur_buf * fx.Int32(KT)
            nxt_buf = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
            kbuf_n = nxt_buf * fx.Int32(_K_BYTES)
            vbuf_n = nxt_buf * fx.Int32(_V_BYTES)
            kdbuf_n = nxt_buf * fx.Int32(KT)
            kc_w_next, vc_w_next = load_kv_regs(kv0_outer + fx.Int32(KT))  # OPT3 prefetch
            fx.rocdl.s_setprio(1)
            m_run, l_run, o_acc = compute_kt_tile(kv0_outer, kbuf, vbuf, kdbuf, m_run, l_run, o_acc, do_mask)
            fx.rocdl.s_setprio(0)
            store_kv_to_lds(kc_w_next, vc_w_next, kv0_outer + fx.Int32(KT), kbuf_n, vbuf_n, kdbuf_n)
            if const_expr(BUFK):
                _wait_vmem()
            fx.gpu.barrier()
            return m_run, l_run, o_acc

        # Prologue: stage this split's FIRST outer tile (kt_lo) into its LDS ping-pong buffer
        # (kt_lo % 2 -- matches the loop_body buffer index). For SPLITKV==1 kt_lo==0 (buffer 0),
        # exactly the log2dom behaviour.
        pf0_buf = (kt_lo % fx.Int32(2))
        pf0_k = pf0_buf * fx.Int32(_K_BYTES)
        pf0_v = pf0_buf * fx.Int32(_V_BYTES)
        pf0_kd = pf0_buf * fx.Int32(KT)
        kc_w0, vc_w0 = load_kv_regs(kt_lo * fx.Int32(KT))
        store_kv_to_lds(kc_w0, vc_w0, kt_lo * fx.Int32(KT), pf0_k, pf0_v, pf0_kd)
        if const_expr(BUFK):
            _wait_vmem()
        fx.gpu.barrier()

        # Phase 1: unmasked interior tiles (no per-element mask VALU) within [kt_lo, um_hi).
        init_state = [m_run0, l_run0] + o_acc0
        for kt_iv, st in range(fx.Index(kt_lo), fx.Index(um_hi), fx.Index(1), init=init_state):
            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
            m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, False)
            st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

        # Phase 2: masked tiles (diagonal + any OOB tail) within [mk_lo, kt_hi).
        mid_state = [st[0], st[1]] + [st[2 + d] for d in range(DT)]
        for kt_iv, st in range(fx.Index(mk_lo), fx.Index(kt_hi), fx.Index(1), init=mid_state):
            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
            m_run, l_run, o_acc = loop_body(kt_iv, m_run, l_run, o_acc, True)
            st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]

        in_b = qrow < sq_i
        if const_expr(SPLITKV > 1):
            # SPLIT-KV epilogue: write PARTIAL state to scratch (do NOT normalize). m_run/l_run are
            # per-q-row scalars replicated across both lane-halves (shuffle_xor(32) in compute), so
            # write them once from half==0. o_acc is the UNNORMALIZED V^T@P accumulation (still in
            # log2-domain weighting). Scratch: Ms/Ls [b,nq,sq,S]; Os [b,nq,sq,S,HD].
            ml_idx = ((batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe) * fx.Int32(SPLITKV) + split
            if in_b:
                if is_h0:
                    fx.buffer_ops.buffer_store(fx.Float32(m_run).ir_value(), rms, ml_idx.ir_value())
                    fx.buffer_ops.buffer_store(fx.Float32(l_run).ir_value(), rls, ml_idx.ir_value())
                o_row_base = (
                    ((batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe) * fx.Int32(SPLITKV) + split
                ) * fx.Int32(HD)
                for dt in fx.range_constexpr(DT):
                    ov = fx.Vector(o_acc[dt])
                    for j in fx.range_constexpr(4):
                        d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                        v4 = fx.Vector.from_elements([fx.Vector(ov)[j * 4 + e] for e in range(4)], fx.Float32)
                        fx.buffer_ops.buffer_store(v4.ir_value(), ros, (o_row_base + d).ir_value())
        else:
            # epilogue: O[d,q] *= v_descale / l_run, cast bf16, store O[b, qrow, qhead, d]
            l_is_zero = l_run < fx.Float32(1.0e-30)
            inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
            scale_o = fx.Float32(v_descale * inv_l)
            scale_vec = fx.Vector.filled(16, scale_o, fx.Float32)
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


# ---------------------------------------------------------------------------
# COMBINE pass: merge the SPLITKV partials per (batch, qhead, qrow) into the final bf16 O.
# One CTHREADS-thread workgroup per (batch, qhead, qrow); each thread owns 4 contiguous head-dims
# (CTHREADS*4 == HD). Log2-domain flash merge (matches log2dom's base-2 m_run/l_run/exp2):
#     m = max_s m_s ; e_s = exp2(m_s - m) ; l = Σ e_s·l_s ; o = (Σ e_s·o_s)/l · v_descale.
# ---------------------------------------------------------------------------
@flyc.kernel(known_block_size=[CTHREADS, 1, 1])
def combine_kernel(
    O: fx.Tensor,
    Ms: fx.Tensor,
    Ls: fx.Tensor,
    Os: fx.Tensor,
    Vd: fx.Tensor,
    sq: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
):
    tid = fx.Int32(fx.thread_idx.x)
    blk = fx.Int32(fx.block_idx.x)
    gqa = nq // nk
    sq_i = fx.Int32(sq)

    qrow = blk % sq_i
    tmp = blk // sq_i
    qhead = tmp % fx.Int32(nq)
    batch = tmp // fx.Int32(nq)
    kvhead = qhead // fx.Int32(gqa)

    ro = fx.buffer_ops.create_buffer_resource(O)
    rms = fx.buffer_ops.create_buffer_resource(Ms)
    rls = fx.buffer_ops.create_buffer_resource(Ls)
    ros = fx.buffer_ops.create_buffer_resource(Os)
    rvd = fx.buffer_ops.create_buffer_resource(Vd)

    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap

    ml_base = ((batch * fx.Int32(nq) + qhead) * sq_i + qrow) * fx.Int32(SPLITKV)
    d0 = tid * fx.Int32(4)

    m_list = []
    l_list = []
    for s in fx.range_constexpr(SPLITKV):
        m_list.append(fx.Float32(fx.buffer_ops.buffer_load(rms, ml_base + fx.Int32(s), vec_width=1, dtype=fx.Float32)))
        l_list.append(fx.Float32(fx.buffer_ops.buffer_load(rls, ml_base + fx.Int32(s), vec_width=1, dtype=fx.Float32)))

    gm = m_list[0]
    for s in fx.range_constexpr(SPLITKV):
        if const_expr(s == 0):
            continue
        gm = gm.maximumf(m_list[s])

    acc = [fx.Float32(0.0) for _ in range(4)]
    lsum = fx.Float32(0.0)
    for s in fx.range_constexpr(SPLITKV):
        m_s = m_list[s]
        is_neg = m_s < fx.Float32(-1.0e38)
        e = fx.Float32(fx.rocdl.exp2(f32t, _ar(m_s - gm)))
        e = is_neg.select(fx.Float32(0.0), e)
        lsum = lsum + e * l_list[s]
        os_idx = (ml_base + fx.Int32(s)) * fx.Int32(HD) + d0
        o4 = fx.Vector(fx.buffer_ops.buffer_load(ros, os_idx, vec_width=4, dtype=fx.Float32))
        for j in range(4):
            acc[j] = acc[j] + e * fx.Float32(o4[j])

    l_is_zero = lsum < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / lsum)
    v_descale = fx.buffer_ops.buffer_load(rvd, batch * fx.Int32(nk) + kvhead, vec_width=1, dtype=fx.Float32)
    scale = fx.Float32(v_descale * inv_l)

    out_f32 = fx.Vector.from_elements([fx.Float32(acc[j] * scale) for j in range(4)], fx.Float32)
    out_bf16 = fx.Vector(out_f32).to(fx.BFloat16)
    o_row_base = ((batch * sq_i + qrow) * fx.Int32(nq) + qhead) * fx.Int32(HD)
    fx.buffer_ops.buffer_store(out_bf16.ir_value(), ro, (o_row_base + d0).ir_value())


@flyc.jit
def _run_main(
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
    Ms: fx.Tensor,
    Ls: fx.Tensor,
    Os: fx.Tensor,
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
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, Ms, Ls, Os,
        sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal,
    ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)


@flyc.jit
def _run_combine(
    O: fx.Tensor,
    Ms: fx.Tensor,
    Ls: fx.Tensor,
    Os: fx.Tensor,
    Vd: fx.Tensor,
    sq: fx.Int32,
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    grid_blocks: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    combine_kernel(O, Ms, Ls, Os, Vd, sq, nq, nk).launch(
        grid=(grid_blocks,), block=(CTHREADS,), stream=stream
    )


# Cache for the HBM partial scratch, keyed by (b, nq, sq), so repeated run_attn calls (e.g. inside
# do_bench) reuse the buffers instead of reallocating each launch. Every (qrow, split) is written
# by exactly one CTA per launch, so no pre-initialization is required.
_SCRATCH = {}


def _get_scratch(b, nq, sq):
    import torch

    key = (int(b), int(nq), int(sq))
    t = _SCRATCH.get(key)
    if t is None:
        Ms = torch.empty((b, nq, sq, SPLITKV), device="cuda", dtype=torch.float32)
        Ls = torch.empty((b, nq, sq, SPLITKV), device="cuda", dtype=torch.float32)
        Os = torch.empty((b, nq, sq, SPLITKV, HD), device="cuda", dtype=torch.float32)
        t = (Ms, Ls, Os)
        _SCRATCH[key] = t
    return t


def run_attn(
    Q,
    K,
    V,
    Qd,
    Kd,
    Vd,
    LTD,
    LTP,
    Ps,
    O,
    sq,
    sk,
    nq,
    nk,
    page_size,
    k_page_stride,
    v_page_stride,
    sm_scale,
    causal,
    grid_blocks,
    stream: fx.Stream = fx.Stream(None),
):
    """Drop-in for the standard FMHA ``run_attn`` signature (ck_check.py / bench_fmha_compare.py
    call this UNCHANGED). For SPLITKV>1 this orchestrates main(partials) + combine on one stream,
    allocating cached HBM scratch internally; ``grid_blocks`` is the un-split grid
    ``b*nq*ceil(sq/BM)`` and is multiplied by SPLITKV for the main launch. For SPLITKV==1 it is the
    exact log2dom fast path (writes normalized bf16 O directly, no scratch/combine)."""
    if SPLITKV <= 1:
        import torch

        d = torch.empty(1, device="cuda", dtype=torch.float32)
        _run_main(
            Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, d, d, d,
            sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal,
            grid_blocks, stream=stream,
        )
        return

    b = Q.shape[0]
    Ms, Ls, Os = _get_scratch(b, nq, sq)
    _run_main(
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, Ms, Ls, Os,
        sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal,
        grid_blocks * SPLITKV, stream=stream,
    )
    combine_grid = b * nq * sq
    _run_combine(O, Ms, Ls, Os, Vd, sq, nq, nk, combine_grid, stream=stream)


# ---------------------------------------------------------------------------
# Self-contained correctness + perf harness (run ONE shape per process; the module-global
# SmemAllocator finalizes once). Drives main + the FlyDSL combine end to end.
#   HIP_VISIBLE_DEVICES=<g> FMHA_KT=64 FMHA_DIAG=0 FMHA_SPLITKV=2 \
#     python3 kernels/fmha_prefill_fp8_splitkv.py 1024
# ---------------------------------------------------------------------------
def _run_one(nq, sq, sk, causal, b=1, nk=1, page_size=16, pscale=1.0, iters=50, warmup=10):
    import sys
    import time
    from pathlib import Path

    _REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO / "tests" / "kernels"))
    import torch

    import fmha_prefill_fp8_ref as R

    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD)
    k = torch.randn(b, sk, nk, HD)
    v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q)
    kf, kd = R.quantize_per_token_head(k)
    vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, page_size, scatter=True, v_col=True)
    args = [
        qf.to("cuda"),
        c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"),
        kd.to("cuda"),
        vd.to("cuda"),
        c.page_ids.to("cuda"),
        c.kv_indptr.to("cuda"),
        torch.full((b * nq,), pscale, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + BM - 1) // BM)

    def _launch():
        run_attn(*args, Og, sq, sk, nq, nk, page_size, c.k_page_stride, c.v_page_stride, sm, causal, grid)

    _launch()
    torch.cuda.synchronize()
    out = Og.float().cpu()

    ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))
    err = (out.float().cpu() - ref.float()).abs().max().item()
    ok = err < 6e-2

    for _ in range(warmup):
        _launch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _launch()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = nq * (4 * sq * sk * HD) / 2 / 1e9 / ms
    print(
        f"SPLITKV={SPLITKV} KT={KT} DIAG={int(DIAG)} sq{sq} sk{sk} nq{nq} c{causal} -> "
        f"ERR {err:.4f} {'PASS' if ok else 'FAIL'} | {ms:.3f} ms  {tflops:.1f} TF  "
        f"(BM={BM} main_grid={grid * SPLITKV} combine_grid={b * nq * sq})"
    )
    return err, ms, tflops


if __name__ == "__main__":
    import sys

    sq = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    sk = int(sys.argv[2]) if len(sys.argv) > 2 else sq
    nq = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    causal = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    _run_one(nq, sq, sk, causal)
