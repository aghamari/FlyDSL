# SPDX-License-Identifier: Apache-2.0
"""Phase-0 occupancy probe: does compile_hints({maxnreg,waves_per_eu}) change VGPR/occupancy
on the DEFAULT in-process gpu-module-to-binary path? (External-LLVM path needs mlir-opt, which
is absent on this box, so this probes the only available path.)

Run: HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/probe_occupancy.py
"""
from __future__ import annotations
import os, sys, glob, re
from pathlib import Path

os.environ.setdefault("FMHA_KT", "32")
os.environ.setdefault("FMHA_DIAG", "0")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
import fmha_prefill_fp8_ref as R

HD = 128


def build_args(b=1, sq=2048, nk=1, gqa=8, ps=16):
    sk = sq
    nq = nk * gqa
    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
    import fmha_prefill_fp8_ck_hk5 as K
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"), c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"), kd.to("cuda"), vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    tail = (sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, 1, grid)
    return K, args, Og, tail


def vgpr_from_dump(dump_dir):
    out = {}
    for f in glob.glob(f"{dump_dir}/**/19_gpu_module_to_binary.mlir", recursive=True) + \
             glob.glob(f"{dump_dir}/**/21_final_isa.s", recursive=True):
        txt = Path(f).read_text(errors="ignore")
        for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "group_segment_fixed_size",
                    ".vgpr_count", ".sgpr_count"):
            m = re.search(rf"{re.escape(key)}\D+(\d+)", txt)
            if m:
                out[key.lstrip(".")] = int(m.group(1))
    return out


def compile_with(hints, tag):
    dump = f"/tmp/occ_{tag}"
    os.system(f"rm -rf {dump}")
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = dump
    K, args, Og, tail = build_args()
    ctx = CompilationContext.compile_hints(hints) if hints else None
    if ctx:
        with ctx:
            K.run_attn(*args, Og, *tail)
    else:
        K.run_attn(*args, Og, *tail)
    torch.cuda.synchronize()
    info = vgpr_from_dump(dump)
    print(f"[{tag}] hints={hints} -> {info}", flush=True)
    return info


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "base":
        compile_with({}, "base")
    else:
        compile_with({"maxnreg": 128, "waves_per_eu": 4}, "maxnreg128_wpe4")
