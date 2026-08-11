from __future__ import annotations

import os
import sys
from typing import Any

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

import fmha_prefill_fp8_layoutmax_hk6

# The reference + paged-cache packing helpers live under FlyDSL/tests/kernels (same helpers the
# bench_fmha_compare.py / ck_check.py harnesses use). Put that dir on sys.path so the adapter can
# build the exact fp8 paged inputs the kernel expects and score against the golden reference.
_REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "kernels")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

# Default benchmark case (mirrors ck_check.py's customer-ish shape): bs=1, causal, GQA 8:1, paged
# with page_size 16. Small seq so verify/bench stay quick while still exercising the KV loop,
# diagonal-pair tiling, and the masked-diagonal path.
_B, _SQ, _SK, _NK, _GQA, _CAUSAL, _PS = 1, 256, 256, 1, 8, 1, 16
_PSCALE = 1.0


def _build_inputs(device: str):
    """Construct the kernel's fp8 paged-attention inputs exactly like bench_fmha_compare.py.

    Returns `(call_args, out, ref_inputs)` where `call_args` is the positional argument tuple for
    `run_attn` up to (and including) `grid_blocks`, `out` is the output tensor to read back, and
    `ref_inputs` are the logical fp8 tensors + descales the golden reference needs."""
    import fmha_prefill_fp8_ref as R

    m = fmha_prefill_fp8_layoutmax_hk6
    hd = m.HD
    nq = _NK * _GQA
    sm = 1.0 / hd**0.5
    dev = torch.device(device)

    torch.manual_seed(0)
    q = torch.randn(_B, _SQ, nq, hd)
    k = torch.randn(_B, _SK, _NK, hd)
    v = torch.randn(_B, _SK, _NK, hd)
    qf, qd = R.quantize_per_token_head(q)
    kf, kd = R.quantize_per_token_head(k)
    vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, _PS, scatter=True, v_col=getattr(m, "V_COL", False))

    out = torch.zeros(_B, _SQ, nq, hd, device=dev, dtype=torch.bfloat16)
    grid = _B * nq * ((_SQ + m.BM - 1) // m.BM)
    call_args = (
        qf.to(dev),
        c.k_pool.view(torch.float8_e4m3fnuz).to(dev),
        c.v_pool.view(torch.float8_e4m3fnuz).to(dev),
        qd.to(dev),
        kd.to(dev),
        vd.to(dev),
        c.page_ids.to(dev),
        c.kv_indptr.to(dev),
        torch.full((_B * nq,), _PSCALE, device=dev),
        out,
        _SQ,
        _SK,
        nq,
        _NK,
        _PS,
        c.k_page_stride,
        c.v_page_stride,
        sm,
        _CAUSAL,
        grid,
    )
    ref_inputs = dict(qf=qf, kf=kf, vf=vf, qd=qd, kd=kd, vd=vd, sm=sm, causal=_CAUSAL)
    return call_args, out, ref_inputs


def trace_attn_kernel(device: str, /) -> None:
    m = fmha_prefill_fp8_layoutmax_hk6
    call_args, _out, _ref = _build_inputs(device)
    flyc.compile(m.run_attn, *call_args, fx.Stream(None))


def verify_attn_kernel(device: str, /) -> None:
    import fmha_prefill_fp8_ref as R

    m = fmha_prefill_fp8_layoutmax_hk6
    call_args, out, ref = _build_inputs(device)

    m.run_attn(*call_args, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    golden = R.fmha_prefill_reference(
        ref["qf"], ref["kf"], ref["vf"], ref["qd"], ref["kd"], ref["vd"], ref["sm"], causal=bool(ref["causal"])
    )
    err = (out.float().cpu() - golden.float()).abs().max().item()
    # fp8 quant + the kernel's fp8-P re-quantize give ~5e-2 spread; the harness uses 6e-2.
    if not err < 6e-2:
        raise AssertionError(f"fp8 FMHA result mismatch: max abs err {err:.4f} (tol 6e-2)")


def bench_attn_kernel(device: str, stream, /):
    m = fmha_prefill_fp8_layoutmax_hk6
    call_args, _out, _ref = _build_inputs(device)

    # Compile with the injected stream (a null stream on the device-less trace pass, a real
    # one on device). The stream is only needed here — trace short-circuits inside compile.
    compiled = flyc.compile(m.run_attn, *call_args, fx.Stream(stream))

    def closure():
        # Launch on the CURRENT stream, read per call: under torch.cuda.graph capture the
        # current stream is the capture stream, so the launch is recorded (not dropped).
        compiled(*call_args, fx.Stream(torch.cuda.current_stream(device)))

    return closure


def adapter_config() -> list[dict[str, Any]]:
    return [
        dict(
            module=fmha_prefill_fp8_layoutmax_hk6,
            trace=trace_attn_kernel,
            trace_configs=dict(default=dict()),
            verify=verify_attn_kernel,
            verify_configs=dict(default=dict()),
            bench=bench_attn_kernel,
            bench_configs=dict(default=dict(warmup_ms=250, bench_ms=250, config=())),
        ),
    ]
