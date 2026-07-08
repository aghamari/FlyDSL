# SPDX-License-Identifier: Apache-2.0
"""Ladder 05 — MULTIWAVE (Lesson 08): NWAVES waves per workgroup to fill the CUs.

FIRST STRUCTURAL rung. The local tier (00-04) ran ONE wave per workgroup on ONE of 80 CUs —
correct but occupancy-starved. Here a workgroup holds NWAVES waves (default 4), each owning its
own 32 q-rows, so BM = NWAVES*32 and grid = ceil(sq/BM). The occupancy win only shows at real
seqlens where grid*NWAVES fills the machine, so this tier is benched in TFLOPS at seqlens.

Each wave still loads K/V for its kv-tiles independently (redundant global traffic across the
waves that share a kv range) -> rung 06 (cooperative-LDS) removes that redundancy.

Everything per-wave (GEMM1, softmax, register-P, column-V GEMM2, epilogue) is identical to
04_column_v; only the wave/grid geometry changed. NWAVES is env-overridable (FMHA_NWAVES).

Run one shape:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/05_multiwave.py 1024 1024 1
All shapes:     HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/05_multiwave.py
"""
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench  # noqa: E402

HD = 128
HDV = 128
KSTEPS = HD // 16    # GEMM1 hd contraction (32x32x16 -> K=16): 8 steps
DT = HDV // 32       # GEMM2 d-tiles: 4
BN = 32              # kv per MFMA tile
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
WAVE_ROWS = 32       # each wave owns 32 q-rows
BM = NWAVES * WAVE_ROWS
LOG2E = 1.4426950408889634
_LGKMCNT0 = 0xC07F   # s_waitcnt mask: drain LDS/bpermute before consuming


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    tid = fx.Int32(fx.thread_idx.x)
    wave = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)         # wave-local lane
    q_local = lane % fx.Int32(32)     # C column / q index (32x32x16)
    half = lane // fx.Int32(32)       # 0/1: which kv half-group
    blk = fx.Int32(fx.block_idx.x)
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    f32t = fx.typing.T.f32
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)

    # workgroup blk owns q-rows [blk*BM, blk*BM+BM); each wave owns its 32-row slice.
    wave_q0 = blk * fx.Int32(BM) + wave * fx.Int32(WAVE_ROWS)
    qrow = wave_q0 + q_local
    qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))

    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # ---- layout-algebra MMA + tiled copies: typed atom -> tiled_mma -> fx.gemm; ----
    # ---- global->register loads for Q/K via make_tiled_copy_A/B + partition_S + retile. ----
    Qb = fx.rocdl.make_buffer_tensor(Q)
    Kb = fx.rocdl.make_buffer_tensor(K)
    Ob = fx.rocdl.make_buffer_tensor(O)
    g_f8 = fx.slice(fx.flat_divide(Kb, (32, 16)), (None, None, 0, 0))   # (32,16) fp8 A/B donor
    g_c = fx.slice(fx.flat_divide(Ob, (32, 32)), (None, None, 0, 0))    # (32,32) f32 C donor
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.typing.T.f8))
    tiled_mma = fx.make_tiled_mma(_mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(lane)   # MMA is per-wave: use the wave-local lane
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.typing.T.f8)
    tcK = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(lane)
    tcQ = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(lane)

    # preload this WAVE's Q fragments (its own 32 q-rows) via a tiled copy.
    q_tile = blk * fx.Int32(NWAVES) + wave      # this wave's global q-tile index
    gQ = fx.slice(fx.flat_divide(Qb, (WAVE_ROWS, 16)), (None, None, q_tile, None))
    thr_gQ = tcQ.partition_S(gQ)
    q_frags = []
    for ks in fx.range_constexpr(KSTEPS):
        fq = thr_mma.make_fragment_B(g_f8)
        fx.copy(cp_ab, fx.slice(thr_gQ, (None, None, None, ks)), tcQ.retile(fq))
        q_frags.append(fq)
    qk_descale = fx.Float32(qd_s * kd_s * sm_scale)

    sk_m1 = sk_i - fx.Int32(1)
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1

    kv_local = lane % fx.Int32(32)
    # causal kv-tile cap (Lesson 14): skip tiles that are fully masked for this q-tile.
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)
    if fx.const_expr(causal == 0):
        n_kv = n_kv_full
    else:
        q_max = blk * fx.Int32(BM) + fx.Int32(BM - 1)
        kv_max = q_max + (sk_i - sq_i)
        n_kv_c = (kv_max + fx.Int32(BN)) // fx.Int32(BN)
        n_kv = (n_kv_c < n_kv_full).select(n_kv_c, n_kv_full)

    m0 = fx.Float32(-3.0e38)
    l0 = fx.Float32(0.0)
    o0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]
    init = [m0, l0] + o0
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BN)

        # ---- GEMM1: S[kv,q] = K @ Q^T (tiled-copy K load + fx.gemm, 8 hd k-steps) ----
        gK = fx.slice(fx.flat_divide(Kb, (BN, 16)), (None, None, fx.Int32(kt), None))
        thr_gK = tcK.partition_S(gK)
        frag_S = thr_mma.make_fragment_C(g_c)
        frag_S.fill(0)
        frag_K = thr_mma.make_fragment_A(g_f8)
        for ks in fx.range_constexpr(KSTEPS):
            fx.copy(cp_ab, fx.slice(thr_gK, (None, None, None, ks)), tcK.retile(frag_K))
            fx.gemm(_mma_atom, frag_S, frag_K, q_frags[ks], frag_S)
        sv_raw = frag_S.load()
        # 32x32x16 C-layout: lane holds S[kv=(i//4)*8+half*4+i%4, q=q_local], i=0..15.
        sv = []
        for i in fx.range_constexpr(16):
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            s = fx.Float32(sv_raw[i]) * qk_descale
            sv.append((kv <= eff_bound).select(s, neg_inf))

        # ---- online softmax (baseline: generic Float32.exp2) ----
        m_loc = sv[0]
        for i in fx.range_constexpr(15):
            m_loc = m_loc.maximumf(sv[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        p_vals = []
        l_loc = fx.Float32(0.0)
        for i in fx.range_constexpr(16):
            pv = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap((sv[i] - safe_m) * fx.Float32(LOG2E))))
            p_vals.append(pv)
            l_loc = l_loc + pv
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # ---- register-resident P-transpose (Lesson 12): ds_bpermute across lanes, NO LDS ----
        def _cvt4(v0, v1, v2, v3):   # 4 f32 -> 4 fp8 packed in one i32
            lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, v0.ir_value(), v1.ir_value(), fx.Int32(0).ir_value(), False)
            return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, v2.ir_value(), v3.ir_value(), lo, True)
        q_byte = q_local * fx.Int32(4)
        q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)
        is_h0 = half == fx.Int32(0)
        frag_B2 = thr_mma.make_fragment_B(g_f8)
        frag_A2 = thr_mma.make_fragment_A(g_f8)
        frag_O = thr_mma.make_fragment_C(g_c)
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        p_i64_s = []
        for s in fx.range_constexpr(2):
            pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])
            pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])
            h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
            h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
            h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
            h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
            fx.rocdl.s_waitcnt(_LGKMCNT0)
            w0 = is_h0.select(h0_b0, h0_b1)
            w1 = is_h0.select(h1_b0, h1_b1)
            p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32))
        for dt in fx.range_constexpr(DT):
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            frag_O.store(fx.Vector(o_acc[dt]) * corr_vec)   # rescale running accumulator
            for s in fx.range_constexpr(2):
                # COLUMN-V: V stored [d, kv] -> 8 contiguous kv for fixed d = ONE wide load.
                # A = V^T: lane holds V[d = d_col, kv = kv0 + s*16 + half*8 + e], e=0..7.
                v_off = d_col * sk_i + kv0 + fx.Int32(s * 16) + half * fx.Int32(8)
                vw = fx.buffer_ops.buffer_load(rV, v_off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
                frag_A2.store(fx.Vector(vw).bitcast(fx.Float8E4M3FNUZ))
                frag_B2.store(fx.Vector(p_i64_s[s]).bitcast(fx.Float8E4M3FNUZ))
                fx.gemm(_mma_atom, frag_O, frag_A2, frag_B2, frag_O)
            o_acc[dt] = frag_O.load()
        st = yield [m_new, l_run] + o_acc

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    out_scale = fx.Float32(vd_s) * inv_l
    in_b = qrow < sq_i
    if in_b:
        for dt in fx.range_constexpr(DT):
            ov = fx.Vector(o_acc[dt]) * fx.Vector.filled(16, out_scale, fx.Float32)
            ov_bf = fx.Vector(ov).to(fx.BFloat16)
            for j in fx.range_constexpr(4):
                d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                v4 = fx.Vector.from_elements([fx.Vector(ov_bf)[j * 4 + e] for e in range(4)], fx.BFloat16)
                o_idx = fx.Int32(qrow_safe * fx.Int32(HDV) + d)
                fx.buffer_ops.buffer_store(v4.ir_value(), rO, o_idx.ir_value())


@flyc.jit
def run_attn(Q, K, V, O, qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
             sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
             grid_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, sm_scale, causal).launch(
        grid=(grid_blocks,), block=(NWAVES * 64,), stream=stream)


def _run_one(sq, sk, causal):
    Qq, qd, Kq, kd, Vq, vd = _bench.make_inputs(sq, sk)
    O = torch.zeros(sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD ** 0.5
    grid = (sq + BM - 1) // BM
    Qc, Kc = Qq.cuda(), Kq.cuda()
    Vc = Vq.t().contiguous().cuda()   # COLUMN-major V: [d, kv]

    def enqueue(strm):
        run_attn(Qc, Kc, Vc, O, qd, kd, vd, sq, sk, sm, causal, grid, stream=strm)

    enqueue(torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _bench.reference(Qq, qd, Kq, kd, Vq, vd, sm, causal, sq, sk)
    ok = _bench.check(f"05_multiwave sq={sq} sk={sk} causal={causal}", O, ref)
    us = _bench.graph_us(enqueue)
    print(f"    graph {us:.2f} us   {_bench.tflops(sq, sk, causal, us):.2f} TFLOPS")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        import subprocess
        # correctness (small) + structural-tier perf (larger seqlens fill more of the grid)
        for shp in [(256, 256, 1), (256, 256, 0), (1024, 1024, 1), (2048, 2048, 1)]:
            subprocess.run([sys.executable, __file__, *map(str, shp)], check=False)
