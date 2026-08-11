from __future__ import annotations

from typing import Any

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

import lesson_05_pv_fused


def trace_attn_kernel(device: str, /) -> None:
    dev = torch.device(device)
    m = lesson_05_pv_fused
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    V = torch.randn(m.BKV, m.HDV, dtype=torch.bfloat16, device=dev)
    O = torch.zeros(m.BQ, m.HDV, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5
    flyc.compile(m.run_attn, Q, K, V, O, sm_scale, fx.Stream(None))


def verify_attn_kernel(device: str, /) -> None:
    dev = torch.device(device)
    m = lesson_05_pv_fused
    torch.manual_seed(0)
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    V = torch.randn(m.BKV, m.HDV, dtype=torch.bfloat16, device=dev)
    O = torch.zeros(m.BQ, m.HDV, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5

    m.run_attn(Q, K, V, O, sm_scale, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: full attention O[q, d] = softmax_kv(Q @ Kᵀ · scale) @ V.
    S = (Q.float() @ K.float().T) * sm_scale
    ref = torch.softmax(S, dim=1) @ V.float()
    err = (O - ref).abs().max().item()
    if not err < 5e-2:
        raise AssertionError(f"attention result mismatch: max abs err {err:.4f} (tol 5e-2)")


def bench_attn_kernel(device: str, stream, /):
    dev = torch.device(device)
    m = lesson_05_pv_fused
    Q = torch.randn(m.BQ, m.HD, dtype=torch.bfloat16, device=dev)
    K = torch.randn(m.BKV, m.HD, dtype=torch.bfloat16, device=dev)
    V = torch.randn(m.BKV, m.HDV, dtype=torch.bfloat16, device=dev)
    O = torch.zeros(m.BQ, m.HDV, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / m.HD**0.5

    # Compile with the injected stream (a null stream on the device-less trace pass, a real
    # one on device). The stream is only needed here — trace short-circuits inside compile.
    compiled = flyc.compile(m.run_attn, Q, K, V, O, sm_scale, fx.Stream(stream))

    def closure():
        # Launch on the CURRENT stream, read per call: under torch.cuda.graph capture the
        # current stream is the capture stream, so the launch is recorded (not dropped).
        compiled(Q, K, V, O, sm_scale, fx.Stream(torch.cuda.current_stream(device)))

    return closure


def adapter_config() -> list[dict[str, Any]]:
    return [
        dict(
            module=lesson_05_pv_fused,
            # Provide either trace or bench (bench derives trace); verify is optional.
            # Delete the capabilities you don't use, along with their *_configs.
            trace=trace_attn_kernel,
            trace_configs=dict(default=dict()),  # TODO: name configs mapping to trace params
            verify=verify_attn_kernel,
            verify_configs=dict(default=dict()),  # TODO: name configs mapping to verify params
            bench=bench_attn_kernel,
            bench_configs=dict(default=dict(warmup_ms=250, bench_ms=250, config=())),  # TODO: fill config
        ),
    ]
