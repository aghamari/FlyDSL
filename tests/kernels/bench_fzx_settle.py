# SPDX-License-Identifier: Apache-2.0
"""Device-fair FMHA bench with a fixed wall-clock GPU warmup (clock-settle) before timing.

Same prebind + CUDA-graph-replay methodology as bench_fmha_fair.py, but adds a ~3s busy warmup
loop so the GPU is boosted to a steady clock REGARDLESS of how long the module took to compile.
The stock harness times right after a short warmup; a larger module (longer compile -> GPU idles
-> down-clocks) then reads ~8% slower even when the emitted ISA is byte-identical. This neutralizes
that bias so the freeze/fexp levers can be compared fairly. New file -- does not touch the harness.

Usage: HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fzx_settle.py <module> <sq> [settle_s]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import importlib
import flydsl.compiler as flyc
import flydsl.expr as fx
import fmha_prefill_fp8_ref as R

HD = 128


def causal_tflops(b, sq, sk, nq, ms):
    return (b * nq * (2.0 * sq * sk * HD + 2.0 * sq * sk * HD) / 2.0) / 1e9 / ms


def bench(mod_name, sq, settle_s=3.0, b=1, nq=8, nk=1, ps=16):
    K = importlib.import_module(mod_name)
    sk = sq
    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"), c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"), kd.to("cuda"), vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    tail = (sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, 1, grid)
    compiled = flyc.compile(K.run_attn, *args, Og, *tail, fx.Stream(torch.cuda.current_stream()))

    def call():
        compiled(*args, Og, *tail, fx.Stream(torch.cuda.current_stream()))

    g = torch.cuda.CUDAGraph()
    call(); call(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        call()

    # Fixed wall-clock busy warmup so the GPU clock is boosted to steady state before timing.
    t0 = time.time()
    while time.time() - t0 < settle_s:
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()

    ts = []
    for _ in range(80):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    ms = ts[len(ts) // 2]
    return ms, causal_tflops(b, sq, sk, nq, ms)


def main():
    mod = sys.argv[1]
    sq = int(sys.argv[2])
    settle = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    ms, tf = bench(mod, sq, settle)
    print(f"  {mod} sq{sq}: {ms:.4f}ms / {tf:.0f}TF (device, graph, settle={settle}s)")


if __name__ == "__main__":
    main()
