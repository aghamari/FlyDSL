# SPDX-License-Identifier: Apache-2.0
"""Self-contained per-level reproduction driver for the FlyDSL fp8 paged causal FMHA
prefill kernel (AMD MI308X / gfx942).

Structural analog of ``ck_dsl/examples/gfx950/fused_mega_moe/reproduce_levels.py``:
one entry point that rebuilds each optimization *level* (a named lever), runs the
hardened parity gate, then the device-fair perf bench, and prints a numeric per-level
ledger plus a machine-readable ``results.tsv``.

Each level maps to an EXISTING production kernel module in ``kernels/`` plus a set of
``FMHA_*`` env overrides (the levers are flag-tunable). Nothing here shadows or mutates
the production kernels — we only fork subprocesses with env set.

WHY SUBPROCESSES: every ``FMHA_*`` knob (KT/DIAG/VCOL/KPAD/VPAD/NWAVES/NBUF/XCD) is read
at IMPORT time, and FlyDSL's module-global ``SmemAllocator`` finalizes exactly ONCE per
process. So a single (level, seqlen) build = one freshly forked Python process whose env
is set BEFORE it imports the kernel. This mirrors ``tests/kernels/sweep_fmha.py`` and CK,
which compiles each instance separately.

Two harness scripts (already in the repo) do the actual work in the child:
  * ``tests/kernels/ck_check.py``     — single-shape parity gate, prints ``OK``/``FAIL``.
  * ``tests/kernels/bench_fmha_fair.py`` — CUDA-graph-replay device-fair TF (the number
    sweep_fmha.py parses; preferred over do_bench which adds ~0.3ms host dispatch).

Usage (from the FlyDSL repo root):
    python3 fmha_research/reproduce_levels.py                         # full ledger
    python3 fmha_research/reproduce_levels.py --levels 13 --seqs 16384
    python3 fmha_research/reproduce_levels.py --levels 7,10,13 --no-perf
    python3 fmha_research/reproduce_levels.py --ck --seqs 1024,16384
    python3 fmha_research/reproduce_levels.py --gpus 0,1,3 --seqs 32768

GPU POLICY: defaults to GPUs 0,1,3,4,5,6,7 and NEVER uses GPU 2. Large seqlens
(16384/32768) may be spread across GPUs in parallel; small seqlens (1024/2048) are
measured ISOLATED + REPEATED on a single GPU (parallel runs are unreliable for small
shapes — the kernel is tiny and noise dominates).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Repo paths (driver lives in <repo>/fmha_research/).
REPO = Path(__file__).resolve().parents[1]
CK_CHECK = str(REPO / "tests" / "kernels" / "ck_check.py")
BENCH = str(REPO / "tests" / "kernels" / "bench_fmha_fair.py")
COMPARE = str(REPO / "tests" / "kernels" / "bench_fmha_compare.py")
RESULTS_TSV = Path(__file__).resolve().parent / "results.tsv"

# Fixed customer shape family: bs=1, nq=8, nk=1, gqa=8, causal, page_size=16, hd=128.
B, NQ, NK, GQA, CAUSAL, PAGE_SIZE = 1, 8, 1, 8, 1, 16
HEADLINE_SEQS = [1024, 2048, 16384, 32768]
# Small shapes must be isolated (one GPU, repeated) — parallel runs are noisy for tiny grids.
SMALL_SEQS = {1024, 2048}
SMALL_REPEAT = 3  # repeat small-seq measurement and keep the best (warm, least-noisy) TF.

# GPU policy: never touch GPU 2. Default usable set:
BANNED_GPUS = {2}
DEFAULT_GPUS = [0, 1, 3, 4, 5, 6, 7]

# CK-Tile fp8 reference (aiter). EXPECTATIONS to confirm on GPU, not ground truth.
# VERIFY-ON-GPU: measure with `--ck`; these are the handoff's reported numbers.
CK_REF_TF = {1024: 30.0, 2048: 62.0, 16384: 141.0, 32768: 146.0}

# Per-seqlen (KT, DIAG) for the L13 dispatcher (matches kernels/fmha_prefill_fp8_dispatch.best_kt_diag).
# VERIFY-ON-GPU: confirm this is still the per-shape optimum after any kernel change.
def best_kt_diag(sq: int) -> tuple[int, int]:
    if sq <= 1024:
        return 64, 0
    if sq <= 2048:
        return 64, 1
    if sq <= 16384:
        return 32, 0
    return 32, 1


@dataclass(frozen=True)
class Level:
    idx: int
    name: str
    family: str  # one of {"baseline","structural","throughput","dispatch"}
    base_module: str  # existing kernels/ module the lever is reproduced from
    env: dict[str, str] = field(default_factory=dict)  # FMHA_* overrides (empty = defaults)
    note: str = ""  # qualitative description + A/B + gotcha
    per_seqlen_env: bool = False  # if True, env is computed per seqlen via best_kt_diag


# ---------------------------------------------------------------------------------------
# The level ledger. Each lever maps to an EXISTING kernels/ module + env overrides; the
# A/B partner (the "before") is the previous level's module. The two FAMILIES of levers
# that have NO clean standalone flag A/B (they are baked into a structural rewrite) are
# documented in `note` and reproduced as the same module — their effect shows up as the
# delta from the prior level, not a flag toggle.
# ---------------------------------------------------------------------------------------
LEVELS: list[Level] = [
    Level(0, "baseline_naive_fp8", "baseline", "fmha_prefill_fp8", {},
          "Correctness-first BM=32, 1 wave (64 thr). LDS P-transpose scratch. The reference "
          "point; everything below is measured as a ratio to the previous level."),
    Level(1, "multiwave_bm128_4wave", "throughput", "fmha_prefill_fp8_8wave",
          {"FMHA_NWAVES": "4"},
          "BM=128, 4 waves/256 thr. A/B: FMHA_NWAVES in {2,8} vs 4 on the SAME module "
          "(NWAVES!=4 is a measured dead-end). Better occupancy/latency-hiding than 256x128/8-wave."),
    Level(2, "coop_kv_to_lds", "structural", "fmha_prefill_fp8_8wave", {"FMHA_NWAVES": "4"},
          "Cooperative K/V -> LDS double-buffered (ping-pong) tile load. EMBEDDED in 8wave; "
          "no clean standalone flag A/B (it is the load path). Documented; reproduced as 8wave."),
    Level(3, "register_P_dsbpermute", "throughput", "fmha_prefill_fp8_8wave", {"FMHA_NWAVES": "4"},
          "Register-resident P, transposed via ds_bpermute (no LDS P scratch). EMBEDDED in 8wave; "
          "the L0 baseline still uses the LDS P-transpose, so the win shows as the L0->L1/L3 delta."),
    Level(4, "fast_exp2", "throughput", "fmha_prefill_fp8_8wave", {"FMHA_NWAVES": "4"},
          "rocdl.exp2 fast softmax exponential (LOG2E pre-scaled scores). EMBEDDED in 8wave; "
          "no flag A/B in this module."),
    Level(5, "causal_loop_split", "structural", "fmha_prefill_fp8_8wave", {"FMHA_NWAVES": "4"},
          "Masked/unmasked causal loop split: interior tiles skip per-element mask VALU "
          "(VALU:MFMA 24->19, +13%). First appears in the CK line; documented here on 8wave."),
    Level(6, "diagonal_pair_tiling", "structural", "fmha_prefill_fp8_v7", {"FMHA_NWAVES": "4"},
          "Diagonal-pair tiling: each CTA does q-tile t AND its causal mirror. A/B: "
          "fmha_prefill_fp8_8wave (one tile/WG) -> fmha_prefill_fp8_v7 (diag), OR FMHA_DIAG=0/1 "
          "on the CK base. +8-24% @ sq>=2048, LOSS @ sq1024 (hence per-shape at L13)."),
    Level(7, "column_v_no_transpose", "structural", "fmha_prefill_fp8_ck", {"FMHA_VCOL": "1"},
          "Column-major V (CK true vec_k_col_v): GEMM2 contraction dim contiguous => delete the "
          "V transpose. A/B: fmha_prefill_fp8_v7 -> fmha_prefill_fp8_ck, OR FMHA_VCOL=0/1. "
          "LDS-wait 54%->18% of busy cycles."),
    Level(8, "lds_row_padding", "throughput", "fmha_prefill_fp8_ck_hk5",
          {"FMHA_KPAD": "8", "FMHA_VPAD": "8"},
          "LDS row padding to break 32-way bank conflicts (68% busy -> spread). A/B: "
          "fmha_prefill_fp8_ck -> fmha_prefill_fp8_ck_hk5, OR FMHA_KPAD/VPAD=0 vs 8 on hk5. "
          "8/8 is the swept optimum (non-monotonic)."),
    Level(9, "kdlds_kdescale_to_lds", "throughput", "fmha_prefill_fp8_combined", {},
          "K-descale staged in LDS (ping-ponged with K/V), off the score-scaling critical path. "
          "A/B: fmha_prefill_fp8_ck_hk5 -> fmha_prefill_fp8_combined."),
    Level(10, "log2e_descale_expbias_hoist", "throughput", "fmha_prefill_fp8_ck_log2dom", {},
          "LOG2E folded into the descale + per-tile exp-bias hoist: whole score domain in log2 "
          "units, removing the per-element *LOG2E. A/B: fmha_prefill_fp8_combined -> "
          "fmha_prefill_fp8_ck_log2dom. THE MEASURED PEAK underlying kernel (layout/reorder are "
          "its readability rewrites)."),
    Level(11, "xcd_chiplet_remap", "dispatch", "fmha_prefill_fp8_ck_log2dom",
          {"FMHA_XCD": "1", "FMHA_XCD_C": "4"},
          "XCD/chiplet block-ID remap (4 XCDs, group C=4 consecutive logical blocks per XCD's L2). "
          "A/B: FMHA_XCD=0 vs 1 (C=4) on hk5/log2dom. Small (VALU-bound): sq16384 110->117, "
          "sq32768 138->140."),
    Level(12, "softmax_valu_fold", "throughput", "fmha_prefill_fp8_ck_log2dom", {},
          "Softmax VALU fold: maxnumf -> v_max3 + p_scale fold. BAKED IN to log2dom; no clean "
          "flag A/B. Documented; reproduced as log2dom (delta vs L10/L11)."),
    Level(13, "per_seqlen_kt_diag_dispatch", "dispatch", "fmha_prefill_fp8_ck_log2dom", {},
          "Per-seqlen (KT,DIAG) dispatch over the log2dom base = CURRENT BEST. Per shape: "
          "sq<=1024 KT64/DIAG0; sq<=2048 KT64/DIAG1; sq<=16384 KT32/DIAG0; else KT32/DIAG1. "
          "NOTE: kernels/fmha_prefill_fp8_dispatch.py currently dispatches over ck_hk5, not "
          "log2dom — use --base to pick the underlying module.",
          per_seqlen_env=True),
]

LEVELS_BY_IDX = {lv.idx: lv for lv in LEVELS}


def resolve_env(level: Level, sq: int, base_override: str | None) -> tuple[str, dict[str, str]]:
    """Return (module, env) for a (level, seqlen). For the per-seqlen dispatch level the
    (KT, DIAG) knobs are computed from `sq`; otherwise the level's static env is used."""
    module = base_override if (base_override and level.per_seqlen_env) else level.base_module
    env = dict(level.env)
    if level.per_seqlen_env:
        kt, diag = best_kt_diag(sq)
        env["FMHA_KT"] = str(kt)
        env["FMHA_DIAG"] = str(diag)
    return module, env


# ---------------------------------------------------------------------------------------
# GPU discovery (mirror sweep_fmha.gpu_list but ALWAYS exclude GPU 2).
# ---------------------------------------------------------------------------------------
def gpu_list(user_gpus: str | None) -> list[int]:
    """Resolve usable GPUs. Explicit --gpus wins (minus banned); else probe idle GPUs via
    rocm-smi; else fall back to DEFAULT_GPUS. GPU 2 is ALWAYS excluded."""
    if user_gpus:
        gpus = [int(g) for g in user_gpus.split(",") if g.strip() != ""]
        return [g for g in gpus if g not in BANNED_GPUS]
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=30
        ).stdout
        # "No KFD" => all GPUs idle. (VERIFY-ON-GPU: rocm-smi output format on this box.)
        if "No KFD" in out:
            return [g for g in range(8) if g not in BANNED_GPUS]
    except Exception:
        pass
    return list(DEFAULT_GPUS)


# ---------------------------------------------------------------------------------------
# Child invocations (one forked process each; env set before the child imports the kernel).
# ---------------------------------------------------------------------------------------
_TF_RE = re.compile(r"/\s*(\d+)\s*TF")  # bench_fmha_fair: "... <ms>ms / <tf>TF (device, graph)"
_MS_RE = re.compile(r"([\d.]+)\s*ms")


def _child_env(env_overrides: dict[str, str], gpu: int) -> dict[str, str]:
    env = {**os.environ, **env_overrides, "HIP_VISIBLE_DEVICES": str(gpu)}
    return env


def run_parity(module: str, sq: int, env_overrides: dict[str, str], gpu: int) -> bool:
    """ck_check.py <module> b sq sk nk gqa causal page_size  -> True iff 'OK' (err<6e-2)."""
    chk = subprocess.run(
        [sys.executable, CK_CHECK, module, str(B), str(sq), str(sq),
         str(NK), str(GQA), str(CAUSAL), str(PAGE_SIZE)],
        capture_output=True, text=True, env=_child_env(env_overrides, gpu),
        cwd=str(REPO), timeout=1800,
    )
    return "OK" in chk.stdout


def run_perf(module: str, sq: int, env_overrides: dict[str, str], gpu: int) -> tuple[float, float]:
    """bench_fmha_fair.py <module> <seqlen> -> (tflops, ms). (0,0) on parse failure."""
    bn = subprocess.run(
        [sys.executable, BENCH, module, str(sq)],
        capture_output=True, text=True, env=_child_env(env_overrides, gpu),
        cwd=str(REPO), timeout=1800,
    )
    mt, mm = _TF_RE.search(bn.stdout), _MS_RE.search(bn.stdout)
    if not mt:
        return 0.0, 0.0
    return float(mt.group(1)), (float(mm.group(1)) if mm else 0.0)


def run_ck_ref(sq: int, gpu: int) -> float:
    """CK-Tile fp8 reference TF via bench_fmha_compare.py --ck (needs aiter on PYTHONPATH).
    Parses the last '<ms>ms/<tf>TF' on the row (the CK-Tile column). VERIFY-ON-GPU."""
    bn = subprocess.run(
        [sys.executable, COMPARE, "--kernels", "fmha_prefill_fp8_ck_log2dom",
         "--seqs", str(sq), "--no-pyisa", "--ck"],
        capture_output=True, text=True, env=_child_env({}, gpu),
        cwd=str(REPO), timeout=1800,
    )
    cells = re.findall(r"([\d.]+)ms/(\d+)TF", bn.stdout)
    if not cells:
        return CK_REF_TF.get(sq, 0.0)  # fall back to the documented expectation
    return float(cells[-1][1])  # last column = CK-Tile


@dataclass
class Cell:
    status: str = "SKIP"  # OK / FAIL / BENCHERR / SKIP
    tflops: float = 0.0
    ms: float = 0.0


def measure_one(level: Level, sq: int, gpu: int, no_perf: bool,
                base_override: str | None, repeat: int = 1) -> Cell:
    """Parity gate then (optionally) device-fair perf for one (level, seqlen) on one GPU.
    `repeat` > 1 keeps the BEST TF (used for small, isolated shapes)."""
    module, env = resolve_env(level, sq, base_override)
    if not run_parity(module, sq, env, gpu):
        return Cell(status="FAIL")
    if no_perf:
        return Cell(status="OK")
    best_tf, best_ms = 0.0, 0.0
    for _ in range(max(1, repeat)):
        tf, ms = run_perf(module, sq, env, gpu)
        if tf > best_tf:
            best_tf, best_ms = tf, ms
    if best_tf <= 0.0:
        return Cell(status="BENCHERR")
    return Cell(status="OK", tflops=best_tf, ms=best_ms)


# ---------------------------------------------------------------------------------------
# Orchestration: small shapes serial+isolated+repeated; large shapes parallel across GPUs.
# ---------------------------------------------------------------------------------------
def run_grid(levels: list[Level], seqs: list[int], gpus: list[int], no_perf: bool,
             base_override: str | None) -> dict[tuple[int, int], Cell]:
    results: dict[tuple[int, int], Cell] = {}

    # Small shapes: ONE GPU, serial, repeated (parallel is unreliable for tiny grids).
    small = [s for s in seqs if s in SMALL_SEQS]
    iso_gpu = gpus[0]
    for sq in small:
        for lv in levels:
            cell = measure_one(lv, sq, iso_gpu, no_perf, base_override, repeat=SMALL_REPEAT)
            results[(lv.idx, sq)] = cell
            print(f"  [iso gpu{iso_gpu}] L{lv.idx} {lv.name} sq{sq}: "
                  f"{cell.status} {cell.tflops:.0f}TF", flush=True)

    # Large shapes: spread (level, seq) jobs across GPUs in parallel.
    large = [s for s in seqs if s not in SMALL_SEQS]
    jobs = [(lv, sq) for sq in large for lv in levels]
    if jobs:
        with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
            futs = {}
            for i, (lv, sq) in enumerate(jobs):
                gpu = gpus[i % len(gpus)]
                futs[ex.submit(measure_one, lv, sq, gpu, no_perf, base_override, 1)] = (lv, sq)
            for fut in as_completed(futs):
                lv, sq = futs[fut]
                cell = fut.result()
                results[(lv.idx, sq)] = cell
                print(f"  [par] L{lv.idx} {lv.name} sq{sq}: "
                      f"{cell.status} {cell.tflops:.0f}TF", flush=True)
    return results


# ---------------------------------------------------------------------------------------
# Ledger output (numeric table + results.tsv append). Mirrors the MoE driver's style.
# ---------------------------------------------------------------------------------------
def ratio(a: float, b: float) -> float:
    return (a / b) if (a > 0 and b > 0) else 0.0


def print_ledger(levels: list[Level], seqs: list[int], results: dict[tuple[int, int], Cell],
                 ck_ref: dict[int, float], no_perf: bool) -> None:
    seq_cols = "".join(f"{('sq'+str(s)):>12}" for s in seqs)
    hdr = f"{'#':>3} {'lever':<30} {'family':<11}{seq_cols}{'x_prev':>9}{'x_ck':>8}"
    print("\n" + "=" * len(hdr))
    print("FMHA PREFILL LEVEL LEDGER (TF = device-fair graph-replay)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    prev_idx: int | None = None
    for lv in levels:
        cells = "".join(
            f"{(f'{results[(lv.idx, s)].tflops:.0f}' if results.get((lv.idx, s), Cell()).status == 'OK' else results.get((lv.idx, s), Cell()).status):>12}"
            for s in seqs
        )
        # ratio-to-prev and ratio-to-CK at the LARGEST measured seqlen (most stable signal).
        ref_seq = seqs[-1]
        cur = results.get((lv.idx, ref_seq), Cell()).tflops
        rp = ratio(cur, results.get((prev_idx, ref_seq), Cell()).tflops) if prev_idx is not None else 0.0
        rc = ratio(cur, ck_ref.get(ref_seq, 0.0))
        rp_s = f"{rp:.2f}x" if rp > 0 else "-"
        rc_s = f"{rc:.2f}x" if rc > 0 else "-"
        print(f"{lv.idx:>3} {lv.name:<30} {lv.family:<11}{cells}{rp_s:>9}{rc_s:>8}")
        prev_idx = lv.idx
    if not no_perf:
        ck_cols = "".join(f"{(f'{ck_ref.get(s, 0.0):.0f}' if ck_ref.get(s) else '-'):>12}" for s in seqs)
        print("-" * len(hdr))
        print(f"{'':>3} {'CK-Tile fp8 (reference)':<30} {'reference':<11}{ck_cols}")
    print("=" * len(hdr))


def append_tsv(levels: list[Level], seqs: list[int], results: dict[tuple[int, int], Cell],
               ck_ref: dict[int, float]) -> None:
    header = "level\tname\tfamily\tseqlen\ttflops\tms\tstatus\tratio_prev\tratio_ck\tdescription\n"
    new = not RESULTS_TSV.exists() or RESULTS_TSV.stat().st_size == 0
    with open(RESULTS_TSV, "a") as f:
        if new:
            f.write(header)
        for i, lv in enumerate(levels):
            prev = levels[i - 1] if i > 0 else None
            for sq in seqs:
                c = results.get((lv.idx, sq), Cell())
                rp = ratio(c.tflops, results.get((prev.idx, sq), Cell()).tflops) if prev else 0.0
                rc = ratio(c.tflops, ck_ref.get(sq, 0.0))
                desc = lv.note.replace("\t", " ").replace("\n", " ")
                f.write(f"{lv.idx}\t{lv.name}\t{lv.family}\t{sq}\t{c.tflops:.1f}\t{c.ms:.4f}\t"
                        f"{c.status}\t{rp:.3f}\t{rc:.3f}\t{desc}\n")
    print(f"\nappended {sum(1 for _ in levels) * len(seqs)} rows to {RESULTS_TSV}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="", help="comma list of level idxs (default: all)")
    ap.add_argument("--seqs", default=",".join(str(s) for s in HEADLINE_SEQS),
                    help="comma list of seqlens")
    ap.add_argument("--no-perf", action="store_true", help="parity gate only, no perf bench")
    ap.add_argument("--ck", action="store_true", help="also measure the CK-Tile fp8 reference row")
    ap.add_argument("--gpus", default="", help="comma list of GPU ids (default: idle set; GPU2 banned)")
    ap.add_argument("--base", default="", help="override the L13 dispatch underlying module "
                                               "(default: fmha_prefill_fp8_ck_log2dom)")
    args = ap.parse_args()

    if args.levels.strip():
        idxs = [int(x) for x in args.levels.split(",")]
        levels = [LEVELS_BY_IDX[i] for i in idxs]
    else:
        levels = list(LEVELS)
    seqs = [int(s) for s in args.seqs.split(",")]
    gpus = gpu_list(args.gpus)
    assert gpus, "no usable GPUs (GPU 2 is always excluded)"
    base_override = args.base.strip() or None

    print(f"levels={[lv.idx for lv in levels]} seqs={seqs} gpus={gpus} "
          f"no_perf={args.no_perf} ck={args.ck}", flush=True)

    results = run_grid(levels, seqs, gpus, args.no_perf, base_override)

    # CK reference row.
    ck_ref = dict(CK_REF_TF)
    if args.ck and not args.no_perf:
        for sq in seqs:
            ck_ref[sq] = run_ck_ref(sq, gpus[0])

    print_ledger(levels, seqs, results, ck_ref, args.no_perf)
    if not args.no_perf:
        append_tsv(levels, seqs, results, ck_ref)


if __name__ == "__main__":
    main()
