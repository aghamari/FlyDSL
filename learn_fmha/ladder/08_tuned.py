# SPDX-License-Identifier: Apache-2.0
"""Ladder 08 — TUNED endpoint: a PMC-driven optimization pass over 07_multihead.

This rung is the result of running the optimization-runbook loop on the endpoint. The key
finding (rocprofv3 PMC at sq=2048, nq=8): the kernel is **VALU-bound, not LDS-bound** —
`SQ_WAIT_INST_LDS / SQ_BUSY = 0.6%` but `SQ_INSTS_VALU : SQ_INSTS_MFMA = 18.4 : 1`. That is
*why* the production `ck_log2dom` kernel beats the plain endpoint: its wins are VALU/structural,
not layout.

Applied (kept — correctness preserved, small measured gains):
  - LOG2E-into-descale (log2 domain): softmax runs in log2 units so exp2 needs no per-element
    *log2e multiply (the `log2dom` VALU idea).
  - `maxnreg=96` occupancy hint (+~4%) and `fast_fp_math` (+~1%).

Measured DEAD ENDS (evidence in the ladder README) — NOT applied:
  - LDS row padding (hk5's big win): 0% here — LDS is not our bottleneck at this shape.
  - Diagonal-pair (Lesson 16): -9% here — with nq=8 the grid already fills the 80 CUs, so
    halving the workgroup count under-fills. (It only helps single-head / very large seq.)

Result: ~50.8 TF @sq2048, ~104 TF @sq4096 (nq=8, causal), fair CUDA-graph. On par with the mid
production kernel `fmha_prefill_fp8_ck` (48 TF); still ~65-75% of the best `ck_log2dom`
(68/158 TF). Closing that last gap needs deeper VALU reduction (e.g. Schraudolph fast-exp
approximation) and per-shape tuning that trade accuracy/generality — see README.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/08_tuned.py 2048 2048 1 8
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
KSTEPS = HD // 16
DT = HDV // 32
BN = 32
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
WAVE_ROWS = 32
BM = NWAVES * WAVE_ROWS
NTHREADS = NWAVES * 64
LOG2E = 1.4426950408889634
_LGKMCNT0 = 0xC07F

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="ladder08_smem")
_K_BYTES = BN * HD
_V_BYTES = HDV * BN
_K_OFF = 0
_V_OFF = _K_OFF + _K_BYTES
_alloc.ptr = _V_OFF + _V_BYTES


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
                n_qtiles: fx.Int32):
    tid = fx.Int32(fx.thread_idx.x)
    wave = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)
    q_local = lane % fx.Int32(32)
    half = lane // fx.Int32(32)
    blk = fx.Int32(fx.block_idx.x)
    head = fx.Int32(fx.block_idx.y)
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    nqt_i = fx.Int32(n_qtiles)
    f32t = fx.typing.T.f32
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)

    q_head_off = head * sq_i * fx.Int32(HD)
    k_head_off = head * sk_i * fx.Int32(HD)
    v_head_off = head * fx.Int32(HDV) * sk_i
    o_head_off = head * sq_i * fx.Int32(HDV)

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    g_f8 = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(K)), fx.make_layout((32, 16), (16, 1)))
    g_c = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(O)), fx.make_layout((32, 32), (32, 1)))
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.typing.T.f8))
    tiled_mma = fx.make_tiled_mma(_mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(lane)
    # log2 domain: fold LOG2E into the descale so exp2 needs no per-element multiply.
    qk_descale = fx.Float32(qd_s * kd_s * sm_scale * LOG2E)
    kv_local = lane % fx.Int32(32)
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)

    def _do_qtile(qt):
        wave_q0 = qt * fx.Int32(BM) + wave * fx.Int32(WAVE_ROWS)
        qrow = wave_q0 + q_local
        qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))

        q_frag = []
        for ks in fx.range_constexpr(KSTEPS):
            off = q_head_off + qrow_safe * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
            w = fx.buffer_ops.buffer_load(rQ, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
            q_frag.append(fx.Vector(w).bitcast(fx.Float8E4M3FNUZ))

        sk_m1 = sk_i - fx.Int32(1)
        if fx.const_expr(causal != 0):
            cb = qrow + (sk_i - sq_i)
            eff_bound = (cb < sk_m1).select(cb, sk_m1)
        else:
            eff_bound = sk_m1

        if fx.const_expr(causal == 0):
            n_kv = n_kv_full
        else:
            q_max = qt * fx.Int32(BM) + fx.Int32(BM - 1)
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

            # cooperative K[kv,hd] + V[d,kv] staging into LDS
            k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(_K_BYTES,)).get()
            vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(_V_BYTES,)).get()
            NSLOT_K = BN * (HD // 16)
            for p in fx.range_constexpr((NSLOT_K + NTHREADS - 1) // NTHREADS):
                slot = fx.Int32(p * NTHREADS) + tid
                ok = slot < fx.Int32(NSLOT_K)
                slot_s = ok.select(slot, fx.Int32(0))
                kv_row = slot_s // fx.Int32(HD // 16)
                cg = slot_s % fx.Int32(HD // 16)
                kvg = kv0 + kv_row
                kvg_s = (kvg < sk_i).select(kvg, fx.Int32(0))
                k_off = k_head_off + kvg_s * fx.Int32(HD) + cg * fx.Int32(16)
                kw = fx.buffer_ops.buffer_load(rK, k_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)
                fx.Vector(kw).bitcast(fx.Int8).store(k_lds, [fx.Index(kv_row * fx.Int32(HD) + cg * fx.Int32(16))])
            NSLOT_V = HDV * (BN // 16)
            for p in fx.range_constexpr((NSLOT_V + NTHREADS - 1) // NTHREADS):
                s2 = fx.Int32(p * NTHREADS) + tid
                ok2 = s2 < fx.Int32(NSLOT_V)
                s2s = ok2.select(s2, fx.Int32(0))
                d_row = s2s // fx.Int32(BN // 16)
                kvg16 = (s2s % fx.Int32(BN // 16)) * fx.Int32(16)
                gkv = kv0 + kvg16
                gkv_s = (gkv < sk_i).select(gkv, fx.Int32(0))
                voff = v_head_off + d_row * sk_i + gkv_s
                vw = fx.buffer_ops.buffer_load(rV, voff // fx.Int32(4), vec_width=4, dtype=fx.Int32)
                fx.Vector(vw).bitcast(fx.Int8).store(vt_lds, [fx.Index(d_row * fx.Int32(BN) + kvg16)])
            fx.gpu.barrier()

            # GEMM1
            frag_S = thr_mma.make_fragment_C(g_c)
            frag_S.fill(0)
            frag_K = thr_mma.make_fragment_A(g_f8)
            frag_Q = thr_mma.make_fragment_B(g_f8)
            for ks in fx.range_constexpr(KSTEPS):
                k_elem = kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
                kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_elem)])
                frag_K.store(fx.Vector(kv8).bitcast(fx.Float8E4M3FNUZ))
                frag_Q.store(q_frag[ks])
                fx.gemm(_mma_atom, frag_S, frag_K, frag_Q, frag_S)
            sv_raw = frag_S.load()
            sv = []
            for i in fx.range_constexpr(16):
                kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
                s = fx.Float32(sv_raw[i]) * qk_descale
                sv.append((kv <= eff_bound).select(s, neg_inf))

            # online softmax (log2 domain)
            m_loc = sv[0]
            for i in fx.range_constexpr(15):
                m_loc = m_loc.maximumf(sv[i + 1])
            m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
            m_new = m_run.maximumf(m_loc)
            m_is_neg = m_new < fx.Float32(-1.0e38)
            safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
            corr = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap(m_run - safe_m)))
            corr = m_is_neg.select(fx.Float32(0.0), corr)
            p_vals = []
            l_loc = fx.Float32(0.0)
            for i in fx.range_constexpr(16):
                pv = fx.Float32(fx.rocdl.exp2(f32t, fx.arith.unwrap(sv[i] - safe_m)))
                p_vals.append(pv)
                l_loc = l_loc + pv
            l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
            l_run = l_run * corr + l_loc

            # register-resident P-transpose (ds_bpermute)
            def _cvt4(v0, v1, v2, v3):
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
                frag_O.store(fx.Vector(o_acc[dt]) * corr_vec)
                for s in fx.range_constexpr(2):
                    v_elem = d_col * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                    vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_elem)])
                    frag_A2.store(fx.Vector(vv8).bitcast(fx.Float8E4M3FNUZ))
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
                    o_idx = fx.Int32(o_head_off + qrow_safe * fx.Int32(HDV) + d)
                    fx.buffer_ops.buffer_store(v4.ir_value(), rO, o_idx.ir_value())

    # NOTE: diagonal-pair (blk + mirror) MEASURED SLOWER at nq=8 — the multihead grid already
    # fills the 80 CUs, so halving the workgroup count under-fills. Single q-tile per workgroup.
    _do_qtile(blk)


@flyc.jit
def run_attn(Q, K, V, O, qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
             sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
             n_qtiles: fx.Int32, nq: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    grid_x = n_qtiles   # single q-tile per workgroup (diagonal-pair under-fills at nq=8)
    _hints = {}
    _mnr = os.environ.get("FMHA_MAXNREG", "96")   # measured best occupancy hint
    _wpe = os.environ.get("FMHA_WPE")
    if _mnr:
        _hints["maxnreg"] = int(_mnr)
    if _wpe:
        _hints["waves_per_eu"] = int(_wpe)
    if os.environ.get("FMHA_FASTMATH", "1") == "1":
        _hints["fast_fp_math"] = True
    _launch = lambda: attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, sm_scale, causal, n_qtiles).launch(
        grid=(grid_x, nq), block=(NWAVES * 64,), stream=stream)
    if _hints:
        with CompilationContext.compile_hints(_hints):
            _launch()
    else:
        _launch()


def _run_one(sq, sk, causal, nq=8):
    torch.manual_seed(0)
    Qq, qd = _bench.quant(torch.randn(nq, sq, HD))
    Kq, kd = _bench.quant(torch.randn(nq, sk, HD))
    Vq, vd = _bench.quant(torch.randn(nq, sk, HDV))
    O = torch.zeros(nq, sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD ** 0.5
    n_qtiles = (sq + BM - 1) // BM
    Qc, Kc = Qq.cuda(), Kq.cuda()
    Vc = Vq.transpose(1, 2).contiguous().cuda()

    def enqueue(strm):
        run_attn(Qc, Kc, Vc, O, qd, kd, vd, sq, sk, sm, causal, n_qtiles, nq, stream=strm)

    enqueue(torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = torch.stack([
        _bench.reference(Qq[h], qd, Kq[h], kd, Vq[h], vd, sm, causal, sq, sk) for h in range(nq)
    ])
    ok = _bench.check(f"08_tuned sq={sq} sk={sk} nq={nq} causal={causal}", O, ref)
    us = _bench.graph_us(enqueue)
    print(f"    graph {us:.2f} us   {_bench.tflops(sq, sk, causal, us, nheads=nq):.2f} TFLOPS")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        nq = int(sys.argv[4]) if len(sys.argv) >= 5 else 8
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), nq)
    else:
        import subprocess
        for shp in [(256, 256, 1, 8), (1024, 1024, 1, 8), (2048, 2048, 1, 8), (4096, 4096, 1, 8)]:
            subprocess.run([sys.executable, __file__, *map(str, shp)], check=False)
