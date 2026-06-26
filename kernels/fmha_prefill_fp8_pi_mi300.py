# SPDX-License-Identifier: Apache-2.0
"""MI300-structured clean-room FP8 causal FMHA prefill (paged, vec_k_col_v) for gfx942 / MI308X.

Phase 2 of the from-scratch PyISA rewrite. This EVOLVES ``fmha_prefill_fp8_pi_base`` (the
simple, validated 4-wave/128-row online-softmax base) into the gfx942 PyISA MI300 STRUCTURE:

  * TWO-BAND 8-WAVE LAYOUT (NTHREADS=512, BM=256). The CTA owns SUB_QD=256 query rows. The
    8 waves q-partition them: waves 0-3 ("band 0") own q-rows [base+0 .. base+127], waves
    4-7 ("band 1") own q-rows [base+128 .. base+255]. Each wave owns WAVE_ROWS=32 rows
    (MFMA M=32). Both bands run the SAME QK->softmax->PV schedule over the SAME shared,
    double-buffered K/V LDS (K/V content is q-independent, loaded cooperatively by all 512
    threads). This is the direct FlyDSL analogue of PyISA's core_loop(0)/core_loop(1):
    waves03 vs waves47 doing identical work on adjacent 128-row q-halves. Per-wave (hence
    per-band) running m/l/o softmax state lives in registers, so it is naturally private.

  * RESUMABLE-SOFTMAX SCHED INTERLEAVE. PyISA's fmha_alu(v_idx, n) splits the softmax VALU
    into source-level chunks emitted *between* the QK/PV MFMA bursts (FIRST_SOFTMAX_ALU_CUT
    etc.) so the softmax ALU overlaps the matmuls. FlyDSL 0.2.0 exposes NO sched_valu, so the
    faithful in-wheel approximation is: bracket each MFMA burst with s_setprio(1)/s_setprio(0)
    and pin the MFMA/DS issue groups with sched_group_barrier (sched_mfma / sched_dsrd) so the
    LLVM post-RA scheduler keeps the MFMA cadence and lets the independent softmax VALU fill
    the MFMA latency shadows. The feasibility study predicts this does NOT reach CK parity in
    this wheel (no sched_valu; the true cross-tile S-bank pipeline inflates VGPR) -- we build
    it as faithfully as expressible, keep it correct, and MEASURE device-fair to confirm/deny.

  * MID-TILE PHASE-GROUPING BARRIER (PINGPONG). A workgroup-uniform s_barrier between the
    VALU-heavy front (GEMM1+softmax+P-transpose) and the MFMA-heavy back (GEMM2). With 2
    waves/SIMD slightly out of phase this lets one wave's GEMM2 MFMA run during another
    wave's softmax VALU -- the deadlock-safe cross-wave approximation of the staggered
    two-band schedule. (A truly divergent ``if band==0: scheduleA else: scheduleB`` staggered
    pipeline is NOT used: it requires byte-matched barrier counts across both branch bodies
    and deadlocks otherwise -- see fmha_prefill_fp8_ck_8wave's CRITICAL warning.)

CRITICAL (deadlock safety): EVERY s_barrier / gpu.barrier here is workgroup-uniform and hit
an identical number of times by all 512 threads. n_kt (the outer-tile loop count) uses the
tile's MAX q-row (CTA-uniform), so all waves run the same iteration count -> matched barrier
counts. No barrier is ever placed inside a divergent ``if wave_id<N`` / ``if band==`` branch.

Inherits pi_base correctness machinery verbatim: column-V (GEMM2 contraction contiguous, no
transpose), +8B/row LDS padding, register-resident P via ds_bpermute, rocdl.exp2 log2-domain
softmax, per-token-head Q/K descale folded with sm_scale+log2e, per-head V descale, p_scale,
per-element causal+bounds mask on every tile. GEMMs: raw mfma_f32_32x32x16_fp8_fp8.

ABI: identical run_attn signature + tensor layouts as the rest of the family, so
tests/kernels/ck_check.py and the benches are drop-in.

Tunables (env, read once at import -> one process per config):
  FMHA_NWAVES  waves/wg, default 8 -> TILE_BM=256 two-band  (set 4 to recover pi_base layout)
  FMHA_PINGPONG  mid-tile workgroup barrier, default 1 (auto-off unless 2 waves/SIMD)
  FMHA_RESUME    resumable-softmax sched interleave (s_setprio + sched_group_barrier), default 1
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# ---------------------------------------------------------------------------
# Shape / tiling constants
# ---------------------------------------------------------------------------
HD = 128
KSTEPS = HD // 16          # GEMM1 contraction steps over head dim (8 * 16 = 128)
DT = HD // 32              # GEMM2 d-tiles (4 * 32 = 128)
NWAVES = int(os.environ.get("FMHA_NWAVES", "8"))
NTHREADS = NWAVES * 64
WAVE_ROWS = 32             # q rows owned by each wave (MFMA M=32)
TILE_BM = NWAVES * WAVE_ROWS  # q rows per q-tile (256 @ 8 waves = SUB_QD; two 128-row bands)
BM = TILE_BM               # grid divisor exported to the harness (no diagonal-pair)

# 2 waves per SIMD when NWAVES==8 (the two-band ping-pong configuration).
TWO_PER_SIMD = NWAVES == 8

KT = int(os.environ.get("FMHA_KT", "32"))   # outer kv tile == one 32x32 MFMA subtile
assert KT == 32, "pi_mi300 is the simple single-subtile variant (KT must be 32)"

# Mid-tile workgroup-uniform barrier (phase-groups MFMA bursts across the 2 waves/SIMD).
PINGPONG = int(os.environ.get("FMHA_PINGPONG", "1")) != 0 and TWO_PER_SIMD
# Resumable-softmax sched interleave (s_setprio + sched_group_barrier MFMA/DS pinning).
RESUME = int(os.environ.get("FMHA_RESUME", "1")) != 0

V_COL = True               # column-major V pool (harness packs to match)
LOG2E = 1.4426950408889634

# Cooperative-load partition: NSLOT 16-byte slots per KT tile. At 8 waves NTHREADS(512) >
# NSLOT(256): only the first NSLOT threads load+store a slot; the surplus skip (guarded).
NSLOT = KT * 8             # 32 kv * 8 feature-groups-of-16 == 256
KVG = KT // 16             # column-V kv-groups-of-16 per d (= 2); HD * KVG == NSLOT
NEED_SLOT_GUARD = NTHREADS > NSLOT

# ---------------------------------------------------------------------------
# LDS layout: double-buffered K [KT x (HD+pad)] and V [HD x (KT+pad)], padded +8B/row
# to break the 32-bank (128B) aliasing of consecutive rows.
# ---------------------------------------------------------------------------
NBUF = 2
_K_PAD = int(os.environ.get("FMHA_KPAD", "8"))
_V_PAD = int(os.environ.get("FMHA_VPAD", "8"))
_K_LDSW = HD + _K_PAD      # K LDS row width in bytes (fp8 = 1B)
_V_LDSW = KT + _V_PAD      # V LDS row width in bytes
_K_BYTES = KT * _K_LDSW    # one K tile
_V_BYTES = HD * _V_LDSW    # one V tile
_K_OFF = 0
_V_OFF = _K_OFF + NBUF * _K_BYTES

# UNIQUE module-global smem symbol so this kernel coexists with pi_base in one process.
_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="fmha_prefill_fp8_pi_mi300_smem")
_alloc.ptr = _V_OFF + NBUF * _V_BYTES


def const_expr(x):
    return fx.const_expr(x)


# s_waitcnt lgkmcnt(0): drain all LDS/scalar ops before reading the bpermute result.
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
    LTD: fx.Tensor,   # int32 [total_pages]  physical page id per slot
    LTP: fx.Tensor,   # int32 [batch+1]      kv_indptr (page0 per batch)
    Ps: fx.Tensor,    # f32   [batch*nq]     per-(batch,qhead) p_scale
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
    q_local = lane % fx.Int32(32)     # MFMA q-row (N) / V d-col within a d-tile
    half = lane // fx.Int32(32)       # which half of the 16-wide MFMA K dim
    kv_local = lane % fx.Int32(32)    # MFMA kv-row (M) for K LDS read
    blk = fx.Int32(fx.block_idx.x)

    gqa = nq // nk
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    sk_m1 = sk_i - fx.Int32(1)

    # Grid decode: blk = (batch*nq + qhead)*num_q_tiles + qtile  (qtile-fastest bijection).
    num_q_tiles = (sq_i + fx.Int32(TILE_BM - 1)) // fx.Int32(TILE_BM)
    qtile = blk % num_q_tiles
    tmp = blk // num_q_tiles
    qhead = tmp % fx.Int32(nq)
    batch = tmp // fx.Int32(nq)
    kvhead = qhead // fx.Int32(gqa)

    # Buffer resources.
    rq = fx.buffer_ops.create_buffer_resource(Q)
    rk = fx.buffer_ops.create_buffer_resource(K)
    rv = fx.buffer_ops.create_buffer_resource(V)
    rqd = fx.buffer_ops.create_buffer_resource(Qd)
    rkd = fx.buffer_ops.create_buffer_resource(Kd)
    rvd = fx.buffer_ops.create_buffer_resource(Vd)
    rltd = fx.buffer_ops.create_buffer_resource(LTD)
    rltp = fx.buffer_ops.create_buffer_resource(LTP)
    rps = fx.buffer_ops.create_buffer_resource(Ps)
    ro = fx.buffer_ops.create_buffer_resource(O)

    q_tok_stride = fx.Int32(nq * HD)
    ps_i = fx.Int32(page_size)

    page0 = fx.buffer_ops.buffer_load(rltp, batch, vec_width=1, dtype=fx.Int32)
    k_head_off = kvhead * fx.Int32(HD * page_size)   # K pool [pages, nk, hd/16, ps, 16]
    v_head_off = kvhead * fx.Int32(HD * page_size)   # col-V  [pages, nk, hd, ps]
    v_descale = fx.buffer_ops.buffer_load(rvd, batch * fx.Int32(nk) + kvhead, vec_width=1, dtype=fx.Float32)

    p_scale = fx.buffer_ops.buffer_load(rps, batch * fx.Int32(nq) + qhead, vec_width=1, dtype=fx.Float32)
    _ps_raw = p_scale.ir_value() if hasattr(p_scale, "ir_value") else p_scale
    log2_pscale = fx.Float32(fx.math.log2(_ps_raw))

    kd_row_base = (batch * fx.Int32(nk) + kvhead) * sk_i

    f32x16 = fx.typing.T.vec(16, fx.typing.T.f32)
    f32t = fx.typing.T.f32
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)
    is_h0 = half == fx.Int32(0)
    q_byte = q_local * fx.Int32(4)
    q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)
    _ar = fx.arith.unwrap

    def _fmax(a, b):
        # maxnum (non-NaN-propagating) fuses into v_max3_f32; softmax max never sees NaN.
        return fx.Float32(arith.maxnumf(_ar(a), _ar(b)))

    k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(NBUF * _K_BYTES,)).get()
    v_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(NBUF * _V_BYTES,)).get()

    # Per-thread cooperative-load slot: one 16B K group and one 16x-kv V group. At 8 waves
    # only slots < NSLOT(256) are valid; the surplus 256 threads skip the store (guard).
    slot = tid
    slot_valid = slot < fx.Int32(NSLOT)
    slot_s = slot_valid.select(slot, fx.Int32(0))
    k_kv = slot_s // fx.Int32(8)       # kv row 0..31
    k_cg = slot_s % fx.Int32(8)        # K feature-group 0..7 (16 fp8 each)
    v_dv = slot_s // fx.Int32(KVG)     # V head-dim 0..127
    v_kvg = slot_s % fx.Int32(KVG)     # V kv-group-of-16 index 0..1

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(),
                                     fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(),
                                       lo, True)

    # --- cooperative global load of a KT tile into registers ---
    def load_kv_regs(kv0):
        kvrow = kv0 + k_kv
        kvrow_safe = (kvrow < sk_i).select(kvrow, fx.Int32(0))
        kslot = page0 + kvrow_safe // ps_i
        kphys = fx.buffer_ops.buffer_load(rltd, kslot, vec_width=1, dtype=fx.Int32)
        kintra = kvrow_safe % ps_i
        kc_off = kphys * k_page_stride + k_head_off + k_cg * (ps_i * fx.Int32(16)) + kintra * fx.Int32(16)
        kw = fx.buffer_ops.buffer_load(rk, kc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)

        # column-V: 16 CONTIGUOUS kv for a fixed d (GEMM2-ready, no transpose).
        kvg0 = kv0 + v_kvg * fx.Int32(16)
        kvg0_safe = (kvg0 < sk_i).select(kvg0, fx.Int32(0))
        vslot = page0 + kvg0_safe // ps_i
        vphys = fx.buffer_ops.buffer_load(rltd, vslot, vec_width=1, dtype=fx.Int32)
        vtok = kvg0_safe % ps_i
        vc_off = vphys * v_page_stride + v_head_off + v_dv * ps_i + vtok
        vw = fx.buffer_ops.buffer_load(rv, vc_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)
        return kw, vw

    def store_kv_to_lds(kw, vw, kbuf, vbuf):
        def _do_store():
            k_dst = kbuf + k_kv * fx.Int32(_K_LDSW) + k_cg * fx.Int32(16)
            fx.Vector(kw).bitcast(fx.Int8).store(k_lds, [fx.Index(k_dst)])
            v_dst = vbuf + v_dv * fx.Int32(_V_LDSW) + v_kvg * fx.Int32(16)
            fx.Vector(vw).bitcast(fx.Int8).store(v_lds, [fx.Index(v_dst)])

        if const_expr(NEED_SLOT_GUARD):
            if slot_valid:
                _do_store()
        else:
            _do_store()

    m_run0 = fx.Float32(-3.0e38)
    l_run0 = fx.Float32(0.0)
    o_acc0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]

    # Two-band q-partition: wave_id*32 covers 0..255; waves 0-3 -> band-0 rows 0-127,
    # waves 4-7 -> band-1 rows 128-255. (band = wave_id // 4, purely documentary here.)
    wave_q0 = qtile * fx.Int32(TILE_BM) + wave_id * fx.Int32(WAVE_ROWS)
    qrow = wave_q0 + q_local
    qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
    q_base = batch * (sq_i * q_tok_stride) + qrow_safe * q_tok_stride + qhead * fx.Int32(HD)

    # Q loaded once into registers, reused over the whole KV loop.
    q_i64 = []
    for ks in fx.range_constexpr(KSTEPS):
        off = q_base + fx.Int32(ks * 16) + half * fx.Int32(8)
        w = fx.buffer_ops.buffer_load(rq, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
        q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])

    qd_idx = (batch * fx.Int32(nq) + qhead) * sq_i + qrow_safe
    q_descale = fx.buffer_ops.buffer_load(rqd, qd_idx, vec_width=1, dtype=fx.Float32)

    # Per-lane exclusive kv bound (causal + sk): valid kv iff kv <= eff_bound.
    if const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1

    # KT tiles needed for this q-tile (skip fully-future tiles when causal). Uses the tile's
    # MAX q-row (qtile*TILE_BM + TILE_BM-1) so n_kt is CTA-uniform -> matched barrier counts.
    if const_expr(causal == 0):
        n_kt = (sk_i + fx.Int32(KT - 1)) // fx.Int32(KT)
    else:
        q_max = qtile * fx.Int32(TILE_BM) + fx.Int32(TILE_BM - 1)
        kv_max = q_max + (sk_i - sq_i)
        n_kt_caus = (kv_max + fx.Int32(KT)) // fx.Int32(KT)
        n_kt_full = (sk_i + fx.Int32(KT - 1)) // fx.Int32(KT)
        n_kt = (n_kt_caus < n_kt_full).select(n_kt_caus, n_kt_full)
    n_kt = (n_kt > fx.Int32(0)).select(n_kt, fx.Int32(1))

    qs = q_descale * fx.Float32(sm_scale * LOG2E)   # folds sm_scale AND log2e into the descale

    # --- one online-softmax step over a single 32-kv tile already staged in LDS ---
    def compute_tile(kv0, kbuf, vbuf, m_run, l_run, o_acc):
        # GEMM1: S[kv,q] = K @ Qᵀ  (accumulate over the 8 head-dim steps).
        k_packs = []
        for ks in fx.range_constexpr(KSTEPS):
            k_elem = kbuf + kv_local * fx.Int32(_K_LDSW) + fx.Int32(ks * 16) + half * fx.Int32(8)
            kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_elem)])
            k_packs.append(fx.Vector(kv8).bitcast(fx.Int64)[0])
        if const_expr(RESUME):
            fx.rocdl.sched_dsrd(KSTEPS)
        acc_raw = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        for ks in fx.range_constexpr(KSTEPS):
            a_raw = k_packs[ks].ir_value() if hasattr(k_packs[ks], "ir_value") else k_packs[ks]
            b_raw = q_i64[ks].ir_value() if hasattr(q_i64[ks], "ir_value") else q_i64[ks]
            acc_raw = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc_raw, 0, 0, 0).res
            if const_expr(RESUME):
                fx.rocdl.sched_mfma(1)
        sv = fx.Vector(acc_raw)

        # descale + causal/bounds mask -> s_vals[16]  (each lane owns q=q_local, 16 kv).
        kdv = []
        for g in fx.range_constexpr(4):
            kv_g0 = kv0 + fx.Int32(g * 8) + half * fx.Int32(4)
            kv_g0 = (kv_g0 + fx.Int32(3) < sk_i).select(kv_g0, fx.Int32(0))
            kdv.append(fx.Vector(fx.buffer_ops.buffer_load(rkd, kd_row_base + kv_g0, vec_width=4, dtype=fx.Float32)))
        s_vals = []
        for i in fx.range_constexpr(16):
            s = sv[i] * (qs * kdv[i // 4][i % 4])
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            s = (kv <= eff_bound).select(s, neg_inf)
            s_vals.append(fx.Float32(s))

        # --- RESUMABLE-SOFTMAX CHUNK 1: row max (over 16 owned kv, then xor-32). ---
        m_loc = s_vals[0]
        for i in fx.range_constexpr(16):
            if const_expr(i == 0):
                continue
            m_loc = _fmax(m_loc, s_vals[i])
        m_loc = _fmax(m_loc, fx.Float32(m_loc.shuffle_xor(off32, width64)))
        m_new = _fmax(m_run, m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar(m_run - safe_m)))
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        # p_scale folded into the pivot once (s - safe_m + log2_ps == s - (safe_m - log2_ps)).
        safe_m_p = safe_m - log2_pscale

        # --- RESUMABLE-SOFTMAX CHUNK 2: exp + running sum. ---
        l_loc = fx.Float32(0.0)
        p_vals = []
        for i in fx.range_constexpr(16):
            p = fx.Float32(fx.rocdl.exp2(f32t, _ar(s_vals[i] - safe_m_p)))
            p_vals.append(p)
            l_loc = l_loc + p
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        for dt in fx.range_constexpr(DT):
            o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec

        # P transpose via ds_bpermute: from [q-by-lane, kv-by-elem] to GEMM2 B [kv, q].
        p_i64 = []
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
            p_i64.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])

        # PING-PONG mid-tile barrier: between the VALU-heavy front (GEMM1+softmax+P) and the
        # MFMA-heavy back (GEMM2). Workgroup-uniform (compute_tile runs n_kt times, identical
        # across all 8 waves), so one wave's GEMM2 can overlap another's softmax VALU.
        if const_expr(PINGPONG):
            fx.rocdl.s_barrier()

        # GEMM2: O[d,q] += Vᵀ @ P  (column-V already [d,kv] in LDS).
        for dt in fx.range_constexpr(DT):
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            v_packs = []
            for s in fx.range_constexpr(2):
                v_elem = vbuf + d_col * fx.Int32(_V_LDSW) + fx.Int32(s * 16) + half * fx.Int32(8)
                vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), v_lds, [fx.Index(v_elem)])
                v_packs.append(fx.Vector(vv8).bitcast(fx.Int64)[0])
            if const_expr(RESUME):
                fx.rocdl.sched_dsrd(2)
            acc2 = fx.Vector(o_acc[dt]).ir_value()
            for s in fx.range_constexpr(2):
                a_raw = v_packs[s].ir_value() if hasattr(v_packs[s], "ir_value") else v_packs[s]
                b_raw = p_i64[s].ir_value() if hasattr(p_i64[s], "ir_value") else p_i64[s]
                acc2 = fx.rocdl.mfma_f32_32x32x16_fp8_fp8(f32x16, a_raw, b_raw, acc2, 0, 0, 0).res
                if const_expr(RESUME):
                    fx.rocdl.sched_mfma(1)
            o_acc[dt] = fx.Vector(acc2)
        return m_new, l_run, o_acc

    # Double-buffered KV loop: prologue stages tile 0, each step prefetches tile+1.
    kw0, vw0 = load_kv_regs(fx.Int32(0))
    store_kv_to_lds(kw0, vw0, fx.Int32(0), fx.Int32(0))
    fx.gpu.barrier()

    init_state = [m_run0, l_run0] + o_acc0
    for kt_iv, st in range(fx.Index(0), fx.Index(n_kt), fx.Index(1), init=init_state):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]

        kv0 = fx.Int32(kt_iv) * fx.Int32(KT)
        cur = fx.Int32(kt_iv) % fx.Int32(2)
        nxt = (fx.Int32(kt_iv) + fx.Int32(1)) % fx.Int32(2)
        kbuf = cur * fx.Int32(_K_BYTES)
        vbuf = cur * fx.Int32(_V_BYTES)
        kbuf_n = nxt * fx.Int32(_K_BYTES)
        vbuf_n = nxt * fx.Int32(_V_BYTES)

        kw_n, vw_n = load_kv_regs(kv0 + fx.Int32(KT))     # prefetch next tile to regs
        if const_expr(RESUME):
            fx.rocdl.s_setprio(1)
        m_run, l_run, o_acc = compute_tile(kv0, kbuf, vbuf, m_run, l_run, o_acc)
        if const_expr(RESUME):
            fx.rocdl.s_setprio(0)
        store_kv_to_lds(kw_n, vw_n, kbuf_n, vbuf_n)
        fx.gpu.barrier()

        st = yield [m_run, l_run] + [o_acc[d] for d in range(DT)]

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]

    # epilogue: O[d,q] *= v_descale / l_run, cast bf16, store O[b, qrow, qhead, d].
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    scale_vec = fx.Vector.filled(16, fx.Float32(v_descale * inv_l), fx.Float32)
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
        Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size,
        k_page_stride, v_page_stride, sm_scale, causal,
    ).launch(grid=(grid_blocks,), block=(NTHREADS,), stream=stream)
