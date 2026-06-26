# SPDX-License-Identifier: Apache-2.0
"""FP8 causal FMHA prefill — PING-PONG (warp-specialized anti-phase) EXPERIMENT.  [MEASURED DEAD-END]

★ WHAT THIS IS: a port of the CK/PyISA "ping-pong" schedule used by the colleague's CDNA4 kernel
(aiter fix-flydsl `flash_attn_fp8_pingpong.py`).  The KV loop is split so the two wave-groups
(G0 = waves [0,NWAVES//2), G1 = the rest) run the tile pipeline in OPPOSITE order: G0 does the
deferred PV(i-1) (MFMA) first then QK(i)+softmax(i) (VALU), while G1 does QK+softmax first then PV.
PV is deferred one tile; both groups carry the previous tile's transposed P (+corr/row-sum).  The
two orderings are numerically identical (PV touches only o/l, softmax only m) — the point is to
stagger the MFMA-heavy and VALU-heavy work of the two groups so the MFMA units stay fed during the
softmax transcendental shadow (the in-wheel realization of warp specialization).  Buffering is
deepened to NBUF_K=2 / NBUF_V=3 (deferred-PV V lifetime), KT is forced to 64 (NSUB=2), and a
``sched_barrier(0)`` fences the two phases so the post-RA scheduler can't collapse the anti-phase.

★ RESULT (MI308X / gfx942, device-fair, KT64 DIAG1, 2026-06-26): CORRECT (err 0.04-0.05 < 6e-2) but
SLOWER than the plain multi-wave baseline at every seqlen, so it is NOT used in production:
      NWAVES=8:  pingpong 82/91/96/110 TF  vs  layout_small 87/98/104/121 TF  @ sq 4k/8k/16k/32k
      NWAVES=4:  pingpong 20/50/58/63      vs  layout_small 21/53/64/70       @ sq 1k/2k/4k/8k
WHY IT LOSES on gfx942: the kernel is VALU/occupancy-bound, and the only thing that hides the
softmax shadow here is OCCUPANCY.  At NWAVES=8 (2 waves/SIMD) the SIMD scheduler ALREADY switches to
the co-resident independent wave's MFMA when a wave stalls on softmax — explicit anti-phase adds no
extra overlap, only overhead (deeper V buffer + carried-P register pressure + always-mask +
divergent scf.if).  At NWAVES=4 (1 wave/SIMD, LDS-bound) G0/G1 sit on different SIMDs so there is no
cross-fill at all.  The colleague's win is CDNA4-specific (wide-K 32x32x64 MFMA + 160KB LDS for free
deep buffering); on gfx942's narrow MFMA the lever is just "more waves", which the existing kernel
already gets.  Kept as a documented negative result.  Bench: tests/kernels/bench_fmha_fair.py.

────────────────────────────────────────────────────────────────────────────────────────────────
Base (data path / layout algebra, unchanged from fmha_prefill_fp8_layout_small.py):

★ Readability-focused clean rewrite of ``fmha_prefill_fp8_layout_max.py``: same validated numerics
and same device-fair performance (20/50/125/145 TF @ sq 1024/2048/16384/32768, = hk5), but with the
dead/non-default paths removed (async-DMA ``BUFK`` and row-major-V — column-V is now HARDWIRED, the
swept optimum) and the paged-gather offsets factored into named helpers (``_k_gather_off`` /
``_v_gather_off``). Layout algebra is applied everywhere it is free + correct (3 layers below).

★ Built on ``fmha_prefill_fp8_ck_hk5.py`` (the measured large-seq winner). Three layers of layout-API:
  1. GEMMs through the typed MMA *atom* — ``fx.make_mma_atom(fx.rocdl.MFMA(32,32,16, fp8))`` +
     ``fly.mma_atom_call_ssa`` (declared once, reused) vs the raw intrinsic.
  2. LDS K/V tile ADDRESSING through ``crd2idx`` over a layout — the padded row strides
     (``_K_LDSW``/``_V_LDSW``) live in ``make_layout((KT,_K_LDSW),(_K_LDSW,1))`` etc.; the element
     offset is ``get_scalar(crd2idx(make_coord(kv,col), layout))`` instead of ``kv*_K_LDSW + col``.
  3. ALL global I/O (Q/K/V/O) through ``logical_divide`` + ``slice`` + copy atoms (the
     flash_attn_gfx950.py idiom). Each tensor is re-viewed FLAT (``make_view(get_iter(
     make_buffer_tensor(X)), make_layout(<big>, 1))``), ``logical_divide``'d by a 1-element tile, and
     loaded via ``copy_atom_call_ssa(atom, slice(view, (None, off)))``. fp8 views are byte-indexed
     (the old ``//4`` byte->dword is gone). MEASURED NEUTRAL: 118/143 TF, ~165 VGPR, 0 spills (= hk5).

  Key gotcha that made (3) work: ``make_buffer_tensor(X)`` keeps X's N-D ([b,sq,nq,HD]) layout, so a
  naive ``logical_divide(make_buffer_tensor(X), <1>)`` divides the BATCH axis and a flat byte offset
  indexes the wrong dim (err 2.54). Re-viewing FLAT first fixes it; the flat extent must be a
  COMPILE-TIME constant (a runtime ``sq_i`` extent regressed to 98/115 TF — runtime-strided layouts
  cost address-compute VGPR on this occupancy-locked kernel).

★ WHAT STAYS HAND-COMPUTED (structural, NOT an oversight): the flat OFFSET fed to ``slice`` is still
  computed by hand, because:
  * Paged K/V gather: a data-dependent page-table indirection (base = ``buffer_load(LTD)``); no
    layout coordinate can express a gather whose base comes from another load. divide/slice still
    applies to the LOAD (flat view), but the offset carries the gathered page id.
  * fp8 lane packing / ds_bpermute P-transpose / chained-GEMM bridge: bespoke per-lane register
    packing; ``fx.gemm``/``make_tiled_mma`` provably can't express the GEMM1->GEMM2 lane handoff
    (the ds_bpermute IS the fix). Documented dead-end (flash_attn_gfx950.py + pi_* clean-rooms).
  * Causal-mask loop bounds (n_kt/n_unmask/eff_bound) + online-softmax: control-flow arithmetic,
    not memory addressing — outside layout algebra's scope entirely.

★ Net: this kernel is VGPR/occupancy-bound (165 VGPR @ 3 waves/SIMD; 4 waves needs <=128,
unreachable in the 0.2.0 wheel), so layout-algebra address abstraction can only be neutral
(compile-time strides) or a regression (runtime strides) — never faster. This file is the maximal
neutral form. See the flydsl-fmha-prefill-opt skill for the full ledger.

★ Canonical hk5 design notes, perf history, the VALU-fold/COLUMN-V wins, occupancy diagnosis, and
the dead-end ledger live in ``learn_fmha/docs/JOURNAL.md`` §2–3.

Tunables (env): FMHA_NWAVES (waves/wg, default 4 -> TILE_BM=128), FMHA_KT (outer kv tile,
default 32), FMHA_VCOL (column-V, default 1), FMHA_DIAG (diagonal-pair, default 1),
FMHA_XCD (chiplet block-ID remap, default 1; FMHA_XCD_C grouping, default 4),
FMHA_BUFK (async K DMA, default 0 -- broken in this wheel).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, fly
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
KT = int(os.environ.get("FMHA_KT", "64"))  # pingpong requires NSUB=2 (KT=64)
assert KT % BN == 0, "FMHA_KT must be a multiple of 32"
NSUB = KT // BN
NBUF = int(os.environ.get("FMHA_NBUF", "2"))  # LDS alloc depth (loop is 2-deep ping-pong)
# PING-PONG anti-phase needs deeper buffering: PV is deferred one tile, so at iter i the live V
# tiles are V(i-1)[consumed], V(i)[idle], V(i+1)[prefetched] -> NBUF_V=3; K only needs K(i),K(i+1)
# -> NBUF_K=2.  (colleague's fix-flydsl pingpong uses the same 2/3 split.)
NBUF_K = int(os.environ.get("FMHA_NBUF_K", "2"))
NBUF_V = int(os.environ.get("FMHA_NBUF_V", "3"))
assert NSUB == 2, "pingpong variant requires KT=64 (NSUB=2): the anti-phase loop carries a fixed 4-word transposed-P state"

# COLUMN-MAJOR V is HARDWIRED here (the swept optimum). V pool [pages, nk, hd, page_size] keeps kv
# (the GEMM2 contraction dim) contiguous per (head, d), so V->LDS is one 128-bit store/slot with NO
# transpose (vs the row-major path's 16x ds_write_b8 scatter). This clean rewrite drops the
# row-major-V and async-DMA (BUFK) toggles entirely — both were non-default / broken.
V_COL = True  # consumed by ck_check.py / bench_fmha_compare.py to pack the matching col-V pool

# XCD/chiplet block-ID remap (HipKittens Algorithm 1, phase-1 grouping). MI308X = 4 XCDs; HW routes
# physical block b -> XCD (b % NXCD). Invert that round-robin so XCD_C consecutive *logical* blocks
# (ordered qhead-fast => all GQA q-heads of one q-tile share the identical causal K/V range) land on
# the SAME XCD's private L2. MEASURED (MI308X, bs1 nq8 nk1 causal): C=4 lifts sq16384 110->117 TF and
# sq32768 138->140, with L2 misses 1.92M->1.70M (hit 95.1%->95.7%) on the single profiled launch. The
# gain is small because the kernel is VALU-bound (mem off the critical path); C in {3,4,5} all tie.
# Default ON with C=4. FMHA_XCD=0 restores the flat grid.
NXCD = int(os.environ.get("FMHA_NXCD", "4"))
XCD_REMAP = int(os.environ.get("FMHA_XCD", "1")) != 0
XCD_C = int(os.environ.get("FMHA_XCD_C", "4"))  # blocks grouped onto one XCD (knob C)

NSLOT = KT * 8  # KT kv x 8 feature-groups of 16 fp8 = 16B slots per tile (same count K and V)
KVG = KT // 16  # column-V: kv-groups-of-16 per d (HD*KVG == NSLOT)
NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_layout_small_smem")
# HK5 — LDS BANK-CONFLICT FIX via row PADDING. Measured baseline: SQ_LDS_BANK_CONFLICT = 68% of busy
# cycles. Cause: K LDS rows have stride HD=128 bytes = 32 banks*4B, so consecutive kv rows alias to
# the SAME bank (up to 32-way conflict on the ds_read). Same for V (stride KT). Fix: pad each LDS row
# by 16 bytes so the row stride is coprime-ish with the 32-bank (128B) period, spreading rows across
# banks. Padding only affects the LDS BUFFER strides; GLOBAL-memory strides keep HD/KT.
# Padding swept (2026-06-14, sq16384): K_PAD=V_PAD=8 is the optimum (108.5 TF, LDS 18944) — beats the
# original 16/16 (106.8, LDS 21504) with LESS LDS. The response is bank-period-sensitive & non-monotonic
# (KPAD=0 -> 63 conflicts back; VPAD=4 -> 59; VPAD=32 -> 75). 8/8 is the sweet spot. Sweepable via env.
_K_PAD = int(os.environ.get("FMHA_KPAD", "8"))  # bytes of pad per K LDS row
_V_PAD = int(os.environ.get("FMHA_VPAD", "8"))  # bytes of pad per V LDS row
_K_LDSW = HD + _K_PAD  # K LDS row width (bytes/elements, fp8=1B)
_V_LDSW = KT + _V_PAD  # V LDS row width
_K_BYTES = KT * _K_LDSW  # K tile [KT kv x (HD+pad)]
_V_BYTES = HD * _V_LDSW  # V tile [HD d x (KT+pad)]
_K_OFF = 0
_V_OFF = _K_OFF + NBUF_K * _K_BYTES
_alloc.ptr = _V_OFF + NBUF_V * _V_BYTES


def const_expr(x):
    return fx.const_expr(x)


# s_waitcnt lgkmcnt(0): wait for all LDS/scalar ops (vmcnt/expcnt = don't-care).
_LGKMCNT0 = 0xC07F


def _wait_lds():
    fx.rocdl.s_waitcnt(_LGKMCNT0)




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
        # Remap is a bijection only over the prefix of FULL NXCD*XCD_C chunks; the partial tail
        # (the last gridDim % (NXCD*XCD_C) blocks) keeps the identity map so the whole map stays a
        # permutation of [0, gridDim) -> no logical tile dropped/duplicated.
        gdim = fx.Int32(fx.grid_dim.x)
        chunk_span = fx.Int32(NXCD * XCD_C)
        n_full = (gdim // chunk_span) * chunk_span
        xcd = blk % fx.Int32(NXCD)
        local = blk // fx.Int32(NXCD)
        chunk_idx = local // fx.Int32(XCD_C)
        pos = local % fx.Int32(XCD_C)
        lblk_r = chunk_idx * chunk_span + xcd * fx.Int32(XCD_C) + pos
        lblk = (blk < n_full).select(lblk_r, blk)
        # Decode QHEAD-FAST: lblk = (batch*num_first + first_idx)*nq + qhead, so an XCD chunk shares
        # the same first_idx (== same causal K/V range) across the GQA q-heads.
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
    half4 = half * fx.Int32(4)   # loop-invariant lane offset, reused in the per-tile mask index

    rqd = fx.buffer_ops.create_buffer_resource(Qd)
    rkd = fx.buffer_ops.create_buffer_resource(Kd)
    rvd = fx.buffer_ops.create_buffer_resource(Vd)
    rltd = fx.buffer_ops.create_buffer_resource(LTD)
    rltp = fx.buffer_ops.create_buffer_resource(LTP)

    q_tok_stride = fx.Int32(nq * HD)

    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)        # vec_k pool: [pages, nk, hd/16, ps, 16]
    v_head_off_col = kvhead * fx.Int32(HD * page_size)    # col-V pool: [pages, nk, hd, ps]
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

    # LAYOUT-API GEMM: declare the 32x32x16 fp8 MFMA ONCE as a typed atom and drive both GEMMs
    # through it (vs repeating the raw `rocdl.mfma_f32_32x32x16_fp8_fp8` intrinsic). Operands are
    # the SAME packed-fp8 i64 / v16f32 acc, so the lane dataflow (and ds_bpermute P-feed) is unchanged.
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.typing.T.f8))

    def _mfma(a_raw, b_raw, c_raw):
        return fly.mma_atom_call_ssa([f32x16], _mma_atom, a_raw, b_raw, c_raw)

    # LAYOUT-ALGEBRA GLOBAL I/O — divide + slice + copy atoms (the flash_attn_gfx950.py idiom).
    # make_buffer_tensor keeps each tensor's N-D layout, so we re-view it FLAT (1-D, stride 1, with a
    # compile-time-large extent — the buffer resource is max-sized, so bounds aren't enforced and the
    # nominal extent is irrelevant). `logical_divide` by a 1-element tile makes that flat view
    # element-indexable; `slice(view, (None, off))` picks the element at the (hand-computed) flat
    # offset and a typed COPY ATOM moves it. fp8 views are byte-indexed (off = byte offset, so the
    # old `//4` byte->dword conversion disappears); the bf16 O view is element-indexed.
    # The flat OFFSET is still computed by hand — attention's paged gather + per-lane fp8 packing are
    # data-dependent/bespoke and cannot be a layout coordinate (see the module docstring).
    v2i32 = fx.typing.T.vec(2, fx.typing.T.i32)
    v4i32 = fx.typing.T.vec(4, fx.typing.T.i32)
    _ld64_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
    _ld128_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
    _st64_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
    _o_store_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)

    def _flat_div(T_):
        flat = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(T_)), fx.make_layout(1 << 26, 1))
        return fx.logical_divide(flat, fx.make_layout(1, 1))

    _q_div = _flat_div(Q)
    _k_div = _flat_div(K)
    _v_div = _flat_div(V)
    _o_div = _flat_div(O)

    def _load_q64(byte_off):
        """64-bit Q load: slice the flat fp8 Q view at a byte offset, copy-atom load -> v2i32."""
        return fly.copy_atom_call_ssa([v2i32], _ld64_atom, fx.slice(_q_div, (None, fx.Int32(byte_off))))

    def _load128(div, byte_off):
        """128-bit K/V load: slice the flat fp8 view at a byte offset, copy-atom load -> v4i32."""
        return fly.copy_atom_call_ssa([v4i32], _ld128_atom, fx.slice(div, (None, fx.Int32(byte_off))))

    def _store_o(v4_bf16, o_elem):
        """Store one 4xbf16 O fragment: pack -> reg, copy-atom store into the flat O view (bf16 elem)."""
        fx.memref_store_vec(fx.Vector(v4_bf16).bitcast(fx.Int32).ir_value(), _o_store_reg)
        fx.copy(_st64_atom, _o_store_reg, fx.slice(_o_div, (None, fx.Int32(o_elem))))

    # LAYOUT-ALGEBRA ADDRESSING: the LDS K/V tiles have COMPILE-TIME strides (the padded row
    # widths _K_LDSW/_V_LDSW), so their element offsets are expressed as crd2idx over a layout
    # instead of hand-multiplying the stride. `_kd(kv,col)` = K tile [KT x (HD+pad)] byte offset;
    # `_vd(d,kv)` = column-V tile [HD x (KT+pad)] byte offset. (The within-row coordinate still
    # carries the lane/subtile arithmetic — that is the logical coordinate, not a stride mul.)
    _k_lds_layout = fx.make_layout((KT, _K_LDSW), (_K_LDSW, 1))
    _v_lds_layout = fx.make_layout((HD, _V_LDSW), (_V_LDSW, 1))

    def _kd(kv, col):
        return fx.Int32(fx.get_scalar(fx.crd2idx(fx.make_coord(kv, col), _k_lds_layout)))

    def _vd(d, kv):
        return fx.Int32(fx.get_scalar(fx.crd2idx(fx.make_coord(d, kv), _v_lds_layout)))

    def _fmax(a, b):
        # maxnum (non-NaN-propagating) fuses into v_max3_f32 (3-input max), halving the softmax
        # reduction VALU; the DSL's maximumf -> llvm.maximum.f32 does NOT fuse. Softmax max needs
        # no NaN propagation (scores are finite or -inf). CK uses maxnum (v_max3) too.
        return fx.Float32(arith.maxnumf(_ar(a), _ar(b)))

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NBUF_K * _K_BYTES,)).get()
    vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NBUF_V * _V_BYTES,)).get()

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

    # --- Paged-gather byte offsets (data-dependent: page id is LOADED from the page table, so this
    # arithmetic cannot be a layout coordinate — it is the one genuinely hand-computed address). ---
    def _k_gather_off(kv0_, p):
        """K byte offset for cooperative pass p (vec_k layout [pages, nk, hd/16, ps, 16])."""
        kvrow = kv0_ + pass_kv[p]
        kvrow = (kvrow < sk_i).select(kvrow, fx.Int32(0))                 # OOB-safe row
        kphys = fx.buffer_ops.buffer_load(rltd, page0 + kvrow // ps_i, vec_width=1, dtype=fx.Int32)
        return kphys * k_page_stride + k_head_off + pass_cg[p] * (ps_i * fx.Int32(16)) + (kvrow % ps_i) * fx.Int32(16)

    def _v_gather_off(kv0_, p):
        """Column-V byte offset for pass p ([pages, nk, hd, ps]): 16 contiguous kv per d, GEMM2-ready."""
        kvg0 = kv0_ + pass_kvg[p] * fx.Int32(16)                          # group base (multiple of 16)
        kvg0 = (kvg0 < sk_i).select(kvg0, fx.Int32(0))
        vphys = fx.buffer_ops.buffer_load(rltd, page0 + kvg0 // ps_i, vec_width=1, dtype=fx.Int32)
        return vphys * v_page_stride + v_head_off_col + pass_dv[p] * ps_i + (kvg0 % ps_i)

    def load_kv_regs(kv0_):
        """Cooperative global->register load of one KT-kv tile. Gather OFFSET is hand-computed (paged),
        but the LOAD is layout-algebra: slice the flat K/V view + 128-bit copy atom (`_load128`)."""
        kc, vc = [], []
        for p in fx.range_constexpr(NPASS):
            kc.append(_load128(_k_div, _k_gather_off(kv0_, p)))
            vc.append(_load128(_v_div, _v_gather_off(kv0_, p)))
        return kc, vc

    def store_kv_to_lds(kc, vc, kbuf_off, vbuf_off):
        """Register->LDS store: K into the [KT x (HD+pad)] row tile, V into the [HD x (KT+pad)] column
        tile (no transpose). One 128-bit store per slot; LDS addresses via crd2idx (`_kd`/`_vd`)."""
        for p in fx.range_constexpr(NPASS):
            def _do_store():
                fx.Vector(kc[p]).bitcast(fx.Int8).store(k_lds, [fx.Index(kbuf_off + _kd(pass_kv[p], pass_cg[p] * fx.Int32(16)))])
                fx.Vector(vc[p]).bitcast(fx.Int8).store(vt_lds, [fx.Index(vbuf_off + _vd(pass_dv[p], pass_kvg[p] * fx.Int32(16)))])

            if const_expr(NPASS > 1):
                if pass_valid[p]:
                    _do_store()
            else:
                _do_store()

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    # Process one BM-row q-tile end to end (Q load -> KV loop -> epilogue). Called once (non-diag)
    # or twice (diagonal-pair: tile + causal mirror) per CTA.
    def process_qtile(qtile):
        wave_q0 = qtile * fx.Int32(TILE_BM) + wave_id * fx.Int32(WAVE_ROWS)
        qrow = wave_q0 + q_local
        qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
        # NOTE: Q/O global addressing is kept DIRECT — converting it to crd2idx over a layout with a
        # RUNTIME extent (sq_i) measured a regression (116/140 -> 98/115 TF) from added address-compute
        # VGPR on this occupancy-locked kernel. Only the compile-time-strided LDS tiles use crd2idx.
        q_base = batch * (sq_i * q_tok_stride) + qrow_safe * q_tok_stride + qhead * fx.Int32(HD)

        # Q LOADED ONCE into registers (CK kQLoadOnce), reused over the whole KV loop.
        q_i64 = []
        for ks in fx.range_constexpr(KSTEPS):
            off = q_base + fx.Int32(ks * 16) + half * fx.Int32(8)
            w = _load_q64(off)   # layout-algebra: slice(flat Q view) + copy atom
            q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])

        qd_idx = (batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe
        q_descale = fx.buffer_ops.buffer_load(rqd, qd_idx, vec_width=1, dtype=fx.Float32)
        # HOIST: per-lane score-scale const (sm * log2e * q_descale), invariant over the KV loop.
        qs = q_descale * fx.Float32(sm_scale * LOG2E)

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

        # KT tiles fully below the causal diagonal AND fully in-bounds need NO masking (CK splits
        # the loop into unmasked-interior + masked-diagonal). `bnd` = smallest q-row in the wave's
        # exclusive causal kv bound; tiles entirely < min(bnd, sk) skip the per-element mask VALU.
        if const_expr(causal == 0):
            lim = sk_i
        else:
            bnd = wave_q0 + (sk_i - sq_i) + fx.Int32(1)
            lim = (bnd < sk_i).select(bnd, sk_i)
        lim = (lim > fx.Int32(0)).select(lim, fx.Int32(0))
        n_unmask = lim // fx.Int32(KT)
        n_unmask = (n_unmask < n_kt_rt).select(n_unmask, n_kt_rt)

        # Process the whole KT tile (NSUB 32-kv subtiles) with the softmax done ONCE over all of
        # it. GEMM1 for all subtiles is emitted first (the K@Q MFMAs are mutually independent, so
        # the scheduler can fill the MFMA unit during the softmax VALU = in-wave overlap), then a
        # single max/corr/rescale, then all GEMM2 MFMAs. Amortises the 64-wide o_acc rescale +
        # max-reduction over NSUB subtiles. NSUB=1 degenerates to the plain online step.
        # ── PING-PONG anti-phase: QK / softmax / PV split into separable phases so the two
        # wave-groups (G0 = waves [0,NWAVES//2), G1 = the rest) run them in OPPOSITE order.
        # PV is deferred one tile: both groups carry the previous tile's transposed P (+ its
        # corr / row-sum) and apply it next iteration.  PV(prev) touches only o/l; softmax(cur)
        # touches only m -> the two are independent, so emitting PV-first (G0) vs PV-last (G1) is
        # numerically identical but staggers the MFMA-heavy (PV+QK) and VALU-heavy (softmax
        # exp/transcendental) work of the two groups.  Since the kernel is ~1 wg/CU (LDS-bound),
        # G0 and G1 waves sit on different SIMDs, so their MFMA and VALU units overlap at the CU
        # level — the in-wheel realization of CK/PyISA ping-pong scheduling.  do_mask is applied
        # on every tile (the n_unmask split is dropped) to keep the loop trip-count CTA-uniform,
        # which is required for the per-iteration s_barrier to be deadlock-safe.
        def do_qk_scores(kbuf, kv0_outer):
            sv = []
            for sub in fx.range_constexpr(NSUB):
                k_packs = []
                for ks in fx.range_constexpr(KSTEPS):
                    k_lds_elem = kbuf + _kd(fx.Int32(sub * BN) + kv_local, fx.Int32(ks * 16) + half * fx.Int32(8))
                    kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_lds_elem)])
                    k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
                acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
                for ks in fx.range_constexpr(KSTEPS):
                    a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
                    b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
                    acc_raw = _mfma(a_raw, b_raw, acc_raw)  # GEMM1: S = K @ Q^T (typed atom)
                sv.append(fx.Vector(acc_raw))
            s_all = []  # flat list over subtiles
            for sub in fx.range_constexpr(NSUB):
                kv0 = kv0_outer + fx.Int32(sub * BN)
                kv_lane = kv0 + half4
                kdv = []
                for g in fx.range_constexpr(4):
                    kv_g0 = kv_lane + fx.Int32(g * 8)
                    kv_g0 = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
                    kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0, vec_width=4, dtype=fx.Float32)))
                s_sub = []
                for i in fx.range_constexpr(16):
                    s = sv[sub][i] * (qs * kdv[i // 4][i % 4])
                    kv = kv_lane + fx.Int32((i // 4) * 8 + (i % 4))
                    s = (kv <= eff_bound).select(s, neg_inf)
                    s_sub.append(fx.Float32(s))
                s_all.append(s_sub)
            return s_all

        def do_softmax(s_all, m_run):
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
            l_loc = fx.Float32(0.0)
            p_all = []
            for sub in fx.range_constexpr(NSUB):
                p_sub = []
                for i in fx.range_constexpr(16):
                    p = fx.Float32(fx.rocdl.exp2(f32t, _ar(s_all[sub][i] - safe_m_p)))
                    p_sub.append(p)
                    l_loc = l_loc + p
                p_all.append(p_sub)
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
            # P transpose (ds_bpermute) for all subtiles -> packed i64 GEMM2 A-operand.
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
            return m_new, corr, l_loc, p_i64_all

        def do_pv(p_i64_all, corr, l_loc, vbuf, o_acc, l_run):
            # Deferred rescale + GEMM2 accumulate: o = o*corr + V^T@P ; l = l*corr + l_loc.
            l_run = l_run * corr + l_loc
            corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
            o_out = []
            for sub in fx.range_constexpr(NSUB):
                p_i64_s = p_i64_all[sub]
                v_packs = []
                for dt in fx.range_constexpr(DT):
                    d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
                    for s in fx.range_constexpr(2):
                        v_lds_elem = vbuf + _vd(d_col, fx.Int32(sub * BN) + fx.Int32(s * 16) + half * fx.Int32(8))
                        vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_lds_elem)])
                        v_packs.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
                for dt in fx.range_constexpr(DT):
                    acc2 = (fx.Vector(o_acc[dt]) * corr_vec).ir_value() if const_expr(sub == 0) else fx.Vector(o_acc[dt]).ir_value()
                    for s in fx.range_constexpr(2):
                        v_i64 = v_packs[dt * 2 + s]
                        p_i64 = p_i64_s[s]
                        a_raw = v_i64.ir_value() if hasattr(v_i64, "ir_value") else v_i64
                        b_raw = p_i64.ir_value() if hasattr(p_i64, "ir_value") else p_i64
                        acc2 = _mfma(a_raw, b_raw, acc2)  # GEMM2: O = V^T @ P (typed atom)
                    o_acc[dt] = fx.Vector(acc2)
            return o_acc, l_run

        def kbuf_of(i32):
            return (i32 % fx.Int32(NBUF_K)) * fx.Int32(_K_BYTES)

        def vbuf_of(i32):
            return (i32 % fx.Int32(NBUF_V)) * fx.Int32(_V_BYTES)

        NPP = NSUB * 2  # number of carried transposed-P i64 words

        # Prologue: stage outer tile 0 into K/V buffer 0.
        kc_w0, vc_w0 = load_kv_regs(fx.Int32(0))
        store_kv_to_lds(kc_w0, vc_w0, kbuf_of(fx.Int32(0)), vbuf_of(fx.Int32(0)))
        fx.gpu.barrier()

        # iter 0: QK(0)+softmax(0) -> P(0); prefetch+stage tile 1; barrier. No deferred PV yet.
        kc_w1, vc_w1 = load_kv_regs(fx.Int32(KT))
        fx.rocdl.s_setprio(1)
        s0 = do_qk_scores(kbuf_of(fx.Int32(0)), fx.Int32(0))
        m0, corr0, lloc0, p0 = do_softmax(s0, m_run0)
        fx.rocdl.s_setprio(0)
        store_kv_to_lds(kc_w1, vc_w1, kbuf_of(fx.Int32(1)), vbuf_of(fx.Int32(1)))
        fx.gpu.barrier()
        p0_flat = [p0[sub][s] for sub in range(NSUB) for s in range(2)]

        is_g0 = wave_id < fx.Int32(NWAVES // 2)

        # Main loop tiles 1..n_kt_rt-1: anti-phase PV(prev) | QK(cur)+softmax(cur).
        init_state = [m0, l_run0] + list(o_acc0) + [corr0, lloc0] + p0_flat
        for kt_iv, st in range(fx.Index(1), fx.Index(n_kt_rt), fx.Index(1), init=init_state):
            m_run = st[0]
            l_run = st[1]
            o_acc = [st[2 + d] for d in range(DT)]
            corr_p = st[2 + DT]
            lloc_p = st[3 + DT]
            p_prev = [[st[4 + DT + sub * 2 + s] for s in range(2)] for sub in range(NSUB)]
            i32 = fx.Int32(kt_iv)
            kc_n, vc_n = load_kv_regs((i32 + fx.Int32(1)) * fx.Int32(KT))  # prefetch tile i+1 (OOB-safe)
            kbuf_cur = kbuf_of(i32)
            vbuf_prev = vbuf_of(i32 - fx.Int32(1))
            fx.rocdl.s_setprio(1)
            # The runtime `if is_g0` lowers to a STATEFUL scf.if: the carried vars must be individual
            # MLIR values that ALREADY EXIST before the if (so the transformer threads them as scf.if
            # results), then are reassigned in BOTH branches. Pre-seed them here (o_*/pc_* from the
            # carry, corr_c/lloc_c as dummies) — both branches overwrite all of them.
            o_0, o_1, o_2, o_3 = o_acc[0], o_acc[1], o_acc[2], o_acc[3]
            corr_c = corr_p
            lloc_c = lloc_p
            pc_0, pc_1, pc_2, pc_3 = p_prev[0][0], p_prev[0][1], p_prev[1][0], p_prev[1][1]
            if is_g0:
                # G0: deferred PV(i-1) FIRST (MFMA), then QK(i)+softmax(i) (VALU tail).
                o_list, l_run = do_pv(p_prev, corr_p, lloc_p, vbuf_prev, list(o_acc), l_run)
                o_0, o_1, o_2, o_3 = o_list
                fx.rocdl.sched_barrier(0)  # fence: keep PV (MFMA) before QK+softmax so the
                s_cur = do_qk_scores(kbuf_cur, i32 * fx.Int32(KT))  # anti-phase order survives RA-sched
                m_run, corr_c, lloc_c, p_cur = do_softmax(s_cur, m_run)
                pc_0, pc_1, pc_2, pc_3 = p_cur[0][0], p_cur[0][1], p_cur[1][0], p_cur[1][1]
            else:
                # G1: QK(i)+softmax(i) FIRST (VALU), then deferred PV(i-1) (MFMA tail).
                s_cur = do_qk_scores(kbuf_cur, i32 * fx.Int32(KT))
                m_run, corr_c, lloc_c, p_cur = do_softmax(s_cur, m_run)
                pc_0, pc_1, pc_2, pc_3 = p_cur[0][0], p_cur[0][1], p_cur[1][0], p_cur[1][1]
                fx.rocdl.sched_barrier(0)  # fence: keep softmax (VALU) before PV (MFMA tail)
                o_list, l_run = do_pv(p_prev, corr_p, lloc_p, vbuf_prev, list(o_acc), l_run)
                o_0, o_1, o_2, o_3 = o_list
            fx.rocdl.s_setprio(0)
            store_kv_to_lds(kc_n, vc_n, kbuf_of(i32 + fx.Int32(1)), vbuf_of(i32 + fx.Int32(1)))
            fx.gpu.barrier()
            st = yield [m_run, l_run, o_0, o_1, o_2, o_3, corr_c, lloc_c, pc_0, pc_1, pc_2, pc_3]

        # Post-loop: unpack final carry (deferred P of tile n_kt_rt-1) and drain it.
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        corr_p = st[2 + DT]
        lloc_p = st[3 + DT]
        p_prev = [[st[4 + DT + sub * 2 + s] for s in range(2)] for sub in range(NSUB)]
        v_last = vbuf_of(fx.Int32(n_kt_rt) - fx.Int32(1))
        o_acc, l_run = do_pv(p_prev, corr_p, lloc_p, v_last, o_acc, l_run)

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
                    _store_o(v4, o_row_base + d)

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
    # Occupancy lever (colleague's fix-flydsl/aiter pingpong kernel mechanism): attach the
    # LLVM function attribute amdgpu-waves-per-eu via value_attrs on the kernel call. The
    # FMHA skill marked this "unreachable from Python in the 0.2.0 wheel" — testing if it works.
    _wpe = int(os.environ.get("FMHA_WPE", "0"))  # 0 = don't set (baseline)
    _va = {}
    if _wpe:
        _va["rocdl.waves_per_eu"] = _wpe
        _va["rocdl.flat_work_group_size"] = f"{NTHREADS},{NTHREADS}"
    attn_kernel(
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal,
        **({"value_attrs": _va} if _va else {}),
    ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)


# ── Per-shape config dispatch (small-seq tuned; device-fair swept 2026-06-26, MI308X) ──────────
# Full NWAVES×KT×DIAG sweep over sq{1024,2048,4096,8192}: NWAVES=4 is optimal everywhere (NWAVES=2
# = smaller tiles/bigger grid did NOT help — this kernel is VALU/occupancy-bound, not grid-bound).
# Best (KT,DIAG) per seqlen (device-fair median, all err<6e-2):
#     sq<=1024 : KT=64 DIAG=0  -> 26 TF / 0.083 ms
#     sq<=2048 : KT=64 DIAG=1  -> 53 TF / 0.161 ms
#     sq<=4096 : KT=32 DIAG=1  -> 88 TF / 0.390 ms
#     else     : KT=32 DIAG=0  -> 112 TF / 1.23 ms (@ sq8192)
# (Default module constants are the large-seq picks; use get_kernel(sq) to compile the per-shape best.)
_MODULE = "fmha_prefill_fp8_layout_small"


def best_config(sq: int) -> tuple[int, int]:
    """Return the swept-best (FMHA_KT, FMHA_DIAG) for this seqlen."""
    if sq <= 1024:
        return 64, 0
    if sq <= 2048:
        return 64, 1
    if sq <= 4096:
        return 32, 1
    return 32, 0


def get_kernel(sq: int):
    """Set the swept-best (KT,DIAG) env for this seqlen and (re)import a fresh module instance
    (KT/DIAG are import-time constexpr; SmemAllocator finalizes once/process). Returns the module
    exposing run_attn + BM. Call ONCE per process per seqlen-class (mirrors
    fmha_prefill_fp8_dispatch.get_kernel — the bench/tests fork a subprocess per shape)."""
    import importlib
    import os
    import sys

    kt, diag = best_config(sq)
    os.environ["FMHA_KT"] = str(kt)
    os.environ["FMHA_DIAG"] = str(diag)
    sys.modules.pop(_MODULE, None)
    return importlib.import_module(_MODULE)
