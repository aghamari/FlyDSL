# SPDX-License-Identifier: Apache-2.0
"""Full-grid autotune sweep for the FMHA prefill kernel (CK generate.py style).

Each (config, shape) is a SEPARATE forked subprocess with the knobs set via env
(FMHA_KT/DIAG/KPAD/VPAD/NWAVES/NBUF). This respects FlyDSL's once-per-process
SmemAllocator finalize AND matches CK, which compiles each instance separately.

Per (config, shape): first ck_check (correctness gate, must be OK), then
bench_fmha_fair (device-fair graph-replay TF). Configs are spread across all free
GPUs for throughput. Emits a TSV and the best config per shape.

Usage: python3 tests/kernels/sweep_fmha.py [--shapes 1024,2048,...] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = "fmha_prefill_fp8_grind"
CK_CHECK = str(REPO / "tests" / "kernels" / "ck_check.py")
BENCH = str(REPO / "tests" / "kernels" / "bench_fmha_fair.py")

# Full grid. KT must be a multiple of BN=32. KPAD/VPAD in bytes. NWAVES in {2,4,8}.
GRID = {
    "FMHA_KT": ["32", "64"],
    "FMHA_DIAG": ["0", "1"],
    "FMHA_KPAD": ["0", "8", "16"],
    "FMHA_VPAD": ["0", "8", "16"],
    "FMHA_NWAVES": ["2", "4", "8"],
    "FMHA_NBUF": ["2"],
}
QUICK_GRID = {  # smoke: just KT x DIAG
    "FMHA_KT": ["32", "64"],
    "FMHA_DIAG": ["0", "1"],
    "FMHA_KPAD": ["8"],
    "FMHA_VPAD": ["8"],
    "FMHA_NWAVES": ["4"],
    "FMHA_NBUF": ["2"],
}


def gpu_list() -> list[int]:
    """GPUs with no KFD pids (idle)."""
    try:
        out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return [0]
    if "No KFD" in out:
        return list(range(8))
    # fallback: assume 0-7, caller can override
    return list(range(8))


def configs(grid):
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def cfg_tag(cfg: dict) -> str:
    return "_".join(f"{k.replace('FMHA_','')}{v}" for k, v in cfg.items())


def run_one(cfg: dict, shape: int, gpu: int) -> tuple[dict, int, str, float]:
    """Fork ck_check then bench for one (cfg, shape) on one GPU. Returns (cfg, shape, status, tf)."""
    env = {**os.environ, **cfg, "HIP_VISIBLE_DEVICES": str(gpu)}
    # correctness gate
    chk = subprocess.run(
        [sys.executable, CK_CHECK, MODULE, "1", str(shape), str(shape), "1", "8", "1", "16"],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900,
    )
    if "OK" not in chk.stdout:
        return cfg, shape, "FAIL", 0.0
    # device-fair bench
    bn = subprocess.run(
        [sys.executable, BENCH, MODULE, str(shape)],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900,
    )
    m = re.search(r"/\s*(\d+)TF", bn.stdout)
    if not m:
        return cfg, shape, "BENCHERR", 0.0
    return cfg, shape, "OK", float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1024,2048,16384,32768")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/tmp/fmha_sweep.tsv")
    ap.add_argument("--workers", type=int, default=0, help="0 = #free GPUs")
    ap.add_argument("--gpus", default="", help="comma list of GPU ids to use (default: all idle)")
    ap.add_argument("--exclude-gpus", default="", help="comma list of GPU ids to AVOID")
    args = ap.parse_args()

    shapes = [int(s) for s in args.shapes.split(",")]
    grid = QUICK_GRID if args.quick else GRID
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = gpu_list()
    if args.exclude_gpus:
        excl = {int(g) for g in args.exclude_gpus.split(",")}
        gpus = [g for g in gpus if g not in excl]
    assert gpus, "no GPUs left after exclusion"
    workers = args.workers or len(gpus)
    cfgs = list(configs(grid))
    jobs = [(c, s) for c in cfgs for s in shapes]
    print(f"sweep: {len(cfgs)} configs x {len(shapes)} shapes = {len(jobs)} jobs on {len(gpus)} GPUs", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i, (c, s) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, c, s, gpu)] = (c, s)
        done = 0
        for fut in as_completed(futs):
            cfg, shape, status, tf = fut.result()
            results.append((cfg, shape, status, tf))
            done += 1
            if status == "OK":
                print(f"[{done}/{len(jobs)}] sq{shape} {cfg_tag(cfg)}: {tf:.0f}TF", flush=True)
            else:
                print(f"[{done}/{len(jobs)}] sq{shape} {cfg_tag(cfg)}: {status}", flush=True)

    # write TSV
    with open(args.out, "w") as f:
        f.write("shape\tstatus\ttflops\t" + "\t".join(GRID.keys()) + "\n")
        for cfg, shape, status, tf in sorted(results, key=lambda r: (r[1], -r[3])):
            f.write(f"{shape}\t{status}\t{tf:.0f}\t" + "\t".join(cfg.get(k, "") for k in GRID) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    # best per shape
    print("\n=== BEST CONFIG PER SHAPE ===")
    for shape in shapes:
        ok = [(tf, cfg) for cfg, s, st, tf in results if s == shape and st == "OK"]
        if not ok:
            print(f"  sq{shape}: no passing config")
            continue
        tf, cfg = max(ok)
        print(f"  sq{shape}: {tf:.0f}TF  {cfg_tag(cfg)}")


if __name__ == "__main__":
    main()
