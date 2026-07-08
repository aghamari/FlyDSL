# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the FMHA fp8 optimization ladder.

Provides:
  - quant / make_inputs      : fp8 e4m3fnuz quantization (per-tensor) of Q/K/V.
  - reference                : torch golden attention from the DEQUANTIZED fp8 inputs
                               (so we measure the kernel, not quant noise).
  - check                    : max-abs-error vs reference, PASS if < 6e-2.
  - graph_us                 : CUDA-graph replay timing (strips launch overhead so the
                               real kernel delta is visible) -> microseconds/launch.
  - tflops                   : attention TFLOPS from us (2 GEMMs, causal ~half).

Every ladder rung imports this and drives its own kernel (signatures differ per rung).
"""
import torch

HD = 128
HDV = 128
TOL = 6e-2


def quant(t):
    """Per-tensor fp8 e4m3fnuz quantization. Returns (fp8_tensor, scale)."""
    s = t.abs().max().item() / 224.0  # e4m3 max ~448; 224 leaves headroom
    return (t / s).to(torch.float8_e4m3fnuz), s


def make_inputs(sq, sk, seed=0):
    torch.manual_seed(seed)
    Qq, qd = quant(torch.randn(sq, HD))
    Kq, kd = quant(torch.randn(sk, HD))
    Vq, vd = quant(torch.randn(sk, HDV))
    return Qq, qd, Kq, kd, Vq, vd


def reference(Qq, qd, Kq, kd, Vq, vd, sm, causal, sq, sk):
    """Golden single-head attention from the dequantized fp8 tensors -> O[sq, HDV]."""
    S = (Qq.float() * qd @ (Kq.float() * kd).T) * sm
    if causal:
        qi = torch.arange(sq).view(-1, 1)
        ki = torch.arange(sk).view(1, -1)
        S = S.masked_fill(ki > qi + (sk - sq), float("-inf"))
    return torch.softmax(S, dim=1) @ (Vq.float() * vd)


def check(tag, O, ref, tol=TOL):
    err = (O.float().cpu() - ref).abs().max().item()
    ok = err < tol
    print(f"{tag}  err={err:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


def graph_us(enqueue, warmup=30, iters=2000):
    """CUDA-graph replay timing. `enqueue(stream)` must launch the kernel on `stream`."""
    cap = torch.cuda.Stream()
    with torch.cuda.stream(cap):
        for _ in range(warmup):
            enqueue(cap)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=cap):
        enqueue(cap)
    torch.cuda.synchronize()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0


def tflops(sq, sk, causal, us, nheads=1):
    flop = 4.0 * sq * sk * HD * nheads  # QK + PV, each *2 for multiply-add
    if causal:
        flop *= 0.5
    return flop / (us * 1e-6) / 1e12
