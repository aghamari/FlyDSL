# SPDX-License-Identifier: Apache-2.0
"""Ladder 00 — BASELINE fused fp8 attention (correct, un-optimized), layout-algebra GEMMs.

The bottom rung: a correct fused causal fp8 attention kernel that later rungs optimize one
step at a time. Single head, a GRID of single-wave workgroups (BM=32 q-rows each,
grid=ceil(sq/32)), 32x32x16 fp8 MFMA, runtime kv-loop with online softmax.

DELIBERATELY UN-OPTIMIZED (each becomes a later rung):
  - generic `Float32.exp2()` softmax           -> rung 01 (rocdl.exp2)
  - loops ALL kv-tiles (no causal skip)        -> rung 02 (causal-bound)
  - P-transpose through LDS (store/reload)      -> rung 03 (register-P via ds_bpermute)
  - row-major V byte-gather in GEMM2            -> rung 04 (column-major V, transpose deleted)
  - single wave / workgroup                     -> rung 05 (multiwave)

LAYOUT ALGEBRA SCOPE: both GEMMs run through the layout API — one typed 32x32x16 fp8
`make_mma_atom` feeds a `make_tiled_mma`, whose derived A/B/C fragment layouts drive
`fx.gemm` (operands filled by hand because fp8 packing / LDS / gathers are the skill's
stay-direct cases). The online softmax reduction, the fp8 cvt_pk packing, the P-transpose,
and the row-major V gather stay direct.

Run one shape:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/00_baseline.py 256 256 1
All shapes:     HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/00_baseline.py
"""
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench  # noqa: E402

HD = 128
HDV = 128
KSTEPS = HD // 16    # GEMM1 hd contraction (32x32x16 -> K=16): 8 steps
DT = HDV // 32       # GEMM2 d-tiles: 4
BN = 32              # kv per MFMA tile
BM = 32              # q-rows per workgroup (single wave)
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="ladder00_smem")
_P_BYTES = 32 * BN   # P scratch [q=32, kv=32] fp8 (1 byte each)
_alloc.ptr = _P_BYTES


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    tid = fx.thread_idx.x
    lane = fx.Int32(fx.thread_idx.x)
    q_local = lane % fx.Int32(32)     # C column / q index (32x32x16)
    half = lane // fx.Int32(32)       # 0/1: which kv half-group
    blk = fx.Int32(fx.block_idx.x)
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    f32t = fx.typing.T.f32
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)

    qrow = blk * fx.Int32(BM) + q_local
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
    thr_mma = tiled_mma.thr_slice(tid)
    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.typing.T.f8)
    tcK = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcQ = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)

    # preload this workgroup's Q fragments (reused across kv-tiles) via a tiled copy.
    gQ = fx.slice(fx.flat_divide(Qb, (BM, 16)), (None, None, blk, None))   # (32,16,KSTEPS)
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
    n_kv = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)   # baseline: loop ALL kv-tiles

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
        corr = fx.Float32(((m_run - safe_m) * fx.Float32(LOG2E)).exp2())
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        p_vals = []
        l_loc = fx.Float32(0.0)
        for i in fx.range_constexpr(16):
            pv = fx.Float32(((sv[i] - safe_m) * fx.Float32(LOG2E)).exp2())
            p_vals.append(pv)
            l_loc = l_loc + pv
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # ---- P-transpose through LDS (baseline): pack fp8, scatter [q,kv], reload 8 kv/lane ----
        p_lds = SmemPtr(_alloc.get_base(), 0, fx.typing.T.i8, shape=(_P_BYTES,)).get()
        words = []
        for g in fx.range_constexpr(4):
            lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p_vals[g * 4 + 0].ir_value(),
                                         p_vals[g * 4 + 1].ir_value(), fx.Int32(0).ir_value(), False)
            w = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p_vals[g * 4 + 2].ir_value(),
                                        p_vals[g * 4 + 3].ir_value(), lo, True)
            words.append(w)
        p_i8 = fx.Vector(fx.Vector.from_elements(words, fx.Int32)).bitcast(fx.Int8)
        for i in fx.range_constexpr(16):
            kv = fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            fx.Vector.from_elements([fx.Vector(p_i8)[i]], fx.Int8).store(
                p_lds, [fx.Index(q_local * fx.Int32(BN) + kv)])
        fx.gpu.barrier()

        # ---- GEMM2: O[d,q] += V^T @ P (fx.gemm). Row-major V => byte-gather for A operand. ----
        # B = P reloaded from LDS: lane holds P[q=q_local, kv = s*16 + half*8 + e], 8 contiguous.
        frag_B2 = thr_mma.make_fragment_B(g_f8)
        frag_A2 = thr_mma.make_fragment_A(g_f8)
        frag_O = thr_mma.make_fragment_C(g_c)
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        p_i64_s = []
        for s in fx.range_constexpr(2):
            p_base = q_local * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
            pv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), p_lds, [fx.Index(p_base)])
            p_i64_s.append(pv8)
        for dt in fx.range_constexpr(DT):
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            frag_O.store(fx.Vector(o_acc[dt]) * corr_vec)   # rescale running accumulator
            for s in fx.range_constexpr(2):
                # A = V^T: lane holds V[kv = kv0 + s*16 + half*8 + e, d = d_col], row-major gather.
                v_elems = []
                for e in fx.range_constexpr(8):
                    kv = kv0 + fx.Int32(s * 16) + half * fx.Int32(8) + fx.Int32(e)
                    kv_ok = kv < sk_i
                    kv_safe = kv_ok.select(kv, fx.Int32(0))
                    vv = fx.buffer_ops.buffer_load(rV, kv_safe * fx.Int32(HDV) + d_col,
                                                   vec_width=1, dtype=fx.Int8)
                    v_elems.append(kv_ok.select(vv, fx.Int8(0)))
                frag_A2.store(fx.Vector(fx.Vector.from_elements(v_elems, fx.Int8)).bitcast(fx.Float8E4M3FNUZ))
                frag_B2.store(fx.Vector(p_i64_s[s]).bitcast(fx.Float8E4M3FNUZ))
                fx.gemm(_mma_atom, frag_O, frag_A2, frag_B2, frag_O)
            o_acc[dt] = frag_O.load()
        fx.gpu.barrier()
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
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, sm_scale, causal).launch(
        grid=(grid_blocks,), block=(64,), stream=stream)


def _run_one(sq, sk, causal):
    Qq, qd, Kq, kd, Vq, vd = _bench.make_inputs(sq, sk)
    O = torch.zeros(sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD ** 0.5
    grid = (sq + BM - 1) // BM
    Qc, Kc, Vc = Qq.cuda(), Kq.cuda(), Vq.cuda()

    def enqueue(strm):
        run_attn(Qc, Kc, Vc, O, qd, kd, vd, sq, sk, sm, causal, grid, stream=strm)

    enqueue(torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _bench.reference(Qq, qd, Kq, kd, Vq, vd, sm, causal, sq, sk)
    ok = _bench.check(f"00_baseline sq={sq} sk={sk} causal={causal}", O, ref)
    us = _bench.graph_us(enqueue)
    print(f"    graph {us:.2f} us   {_bench.tflops(sq, sk, causal, us):.2f} TFLOPS")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        import subprocess
        for shp in [(256, 256, 1), (128, 128, 1), (256, 384, 1), (256, 256, 0)]:
            subprocess.run([sys.executable, __file__, *map(str, shp)], check=False)
