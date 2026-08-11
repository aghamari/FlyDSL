from __future__ import annotations

from typing import Any

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

import lesson_04_softmax


def trace_softmax_kernel(device: str, /) -> None:
    dev = torch.device(device)
    m = lesson_04_softmax
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    P = torch.zeros(m.BKV, m.BQ, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5
    flyc.compile(m.run_softmax, Q, K, P, sm_scale, fx.Stream(None))


def verify_softmax_kernel(device: str, /) -> None:
    dev = torch.device(device)
    m = lesson_04_softmax
    torch.manual_seed(0)
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    P = torch.zeros(m.BKV, m.BQ, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5

    m.run_softmax(Q, K, P, sm_scale, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: softmax over kv (rows) of the scaled scores S[kv, q] = K @ Qᵀ.
    ref = torch.softmax((K.float() @ Q.float().T) * sm_scale, dim=0)
    err = (P - ref).abs().max().item()
    if not err < 1e-2:
        raise AssertionError(f"softmax result mismatch: max abs err {err:.5f} (tol 1e-2)")


def bench_softmax_kernel(device: str, stream, /):
    dev = torch.device(device)
    m = lesson_04_softmax
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    P = torch.zeros(m.BKV, m.BQ, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5

    # Compile with the injected stream (a null stream on the device-less trace pass, a real
    # one on device). The stream is only needed here — trace short-circuits inside compile.
    compiled = flyc.compile(m.run_softmax, Q, K, P, sm_scale, fx.Stream(stream))

    def closure():
        # Launch on the CURRENT stream, read per call: under torch.cuda.graph capture the
        # current stream is the capture stream, so the launch is recorded (not dropped).
        compiled(Q, K, P, sm_scale, fx.Stream(torch.cuda.current_stream(device)))

    return closure


def adapter_config() -> list[dict[str, Any]]:
    return [
        dict(
            module=lesson_04_softmax,
            # Provide either trace or bench (bench derives trace); verify is optional.
            # Delete the capabilities you don't use, along with their *_configs.
            trace=trace_softmax_kernel,
            trace_configs=dict(default=dict()),  # TODO: name configs mapping to trace params
            verify=verify_softmax_kernel,
            verify_configs=dict(default=dict()),  # TODO: name configs mapping to verify params
            bench=bench_softmax_kernel,
            bench_configs=dict(default=dict(warmup_ms=250, bench_ms=250, config=())),  # TODO: fill config
        ),
    ]
