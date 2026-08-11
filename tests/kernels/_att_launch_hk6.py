# SPDX-License-Identifier: Apache-2.0
"""Minimal single-dispatch launcher for ATT capture of fmha_prefill_fp8_layoutmax_hk6.

Builds the paged fp8 inputs exactly like bench_fmha_compare.py / ck_check.py and issues ONE
run_attn dispatch (no CPU reference — that is intractable at long seq). Meant to be wrapped by
`flylens capture --command "..."` so rocprofv3 --att traces the single kernel dispatch.

Usage:  python3 tests/kernels/_att_launch_hk6.py <sq> [sk]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import torch  # noqa: E402
import fmha_prefill_fp8_ref as R  # noqa: E402

MOD = "fmha_prefill_fp8_layoutmax_hk6"


def main() -> None:
    sq = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    sk = int(sys.argv[2]) if len(sys.argv) > 2 else sq
    b, nk, gqa, causal, ps = 1, 1, 8, 1, 16
    nq = nk * gqa

    # Match the module's per-seqlen tuned (KT, DIAG) before importing it (env is read at import).
    import importlib

    kmod = importlib.import_module(MOD)
    kt, diag = kmod.best_kt_diag(sq)
    os.environ["FMHA_KT"] = str(kt)
    os.environ["FMHA_DIAG"] = str(diag)
    sys.modules.pop(MOD, None)
    K = importlib.import_module(MOD)

    HD = K.HD
    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD)
    k = torch.randn(b, sk, nk, HD)
    v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q)
    kf, kd = R.quantize_per_token_head(k)
    vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"),
        c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"),
        kd.to("cuda"),
        vd.to("cuda"),
        c.page_ids.to("cuda"),
        c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)

    # Warm the JIT/compile off the traced dispatch, then issue the single dispatch ATT captures.
    K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    torch.cuda.synchronize()
    K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    torch.cuda.synchronize()
    print(f"dispatched {MOD} sq={sq} sk={sk} KT={kt} DIAG={diag} grid={grid}")


if __name__ == "__main__":
    main()
