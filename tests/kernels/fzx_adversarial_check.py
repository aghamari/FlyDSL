# SPDX-License-Identifier: Apache-2.0
"""Adversarial correctness check for the max-freeze + rollback FMHA kernel (hk_fzx).

The default ck_check.py uses iid randn inputs whose interior-tile softmax probabilities P
almost never exceed the fp8-pack ceiling, so they do NOT exercise the rare rollback path -- the
whole risk of the freeze. This crafts a high-variance causal input whose scores INCREASE strongly
with the kv index, so the seed tile (the first, lowest-kv interior tile) is the SMALLEST and a
later interior tile's frozen P = 2^(scale*(S - FA_max_seed)) blows far past 240 (e4m3 FNUZ max),
forcing the per-tile overflow detect + reconstruct-from-P rollback on most interior tiles.

Usage (one shape per process, free GPU):
  HIP_VISIBLE_DEVICES=<g> FMHA_FEXP=2 FMHA_FREEZE=1 FMHA_KT=32 FMHA_DIAG=0 \
    python3 tests/kernels/fzx_adversarial_check.py <module> <sq>
A correct rollback => ERR < 6e-2. Run the same on the broken probe (hk_freeze) to confirm the
input genuinely overflows (it FAILs there).
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import importlib

import torch

import fmha_prefill_fp8_ref as R


def main():
    mod = sys.argv[1]
    sq = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    sk = sq
    b, nk, gqa, causal, ps = 1, 1, 8, 1, 16
    nq = nk * gqa
    K = importlib.import_module(mod)
    HD = K.HD
    sm = 1.0 / HD**0.5

    torch.manual_seed(0)
    # A shared direction u; scores grow with kv so the seed (kv~0) underestimates the true max and
    # later interior tiles overflow the fp8 P-pack. ramp peaks ~ q.u * 40 in the raw dot product;
    # sm_scale ~ 0.088 => tens of nats of spread => P >> 240 on the seeded interior tiles.
    u = torch.randn(HD)
    u = u / u.norm()
    q = u[None, None, None, :] * 6.0 + torch.randn(b, sq, nq, HD) * 0.5
    ramp = (torch.arange(sk, dtype=torch.float32) / sk) * 40.0  # [sk]
    k = u[None, None, None, :] * ramp[None, :, None, None] + torch.randn(b, sk, nk, HD) * 0.5
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
    K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    torch.cuda.synchronize()

    ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))

    # Diagnostic: peak frozen P that a single-seed (tile-0 max) interior softmax would produce, to
    # confirm the input really crosses the 240 fp8 ceiling (i.e. the rollback path is exercised).
    qd_p = qd.permute(0, 2, 1).unsqueeze(-1)
    kd_p = kd.permute(0, 2, 1).unsqueeze(-1)
    qdq = qf.float() * qd_p
    kdq = (kf.float() * kd_p).repeat_interleave(gqa, dim=2)
    sc = torch.einsum("bqhd,bkhd->bhqk", qdq, kdq) * sm
    qpos = torch.arange(sq).unsqueeze(1)
    kpos = torch.arange(sk).unsqueeze(0)
    cmask = (qpos + (sk - sq) < kpos)
    sc = sc.masked_fill(cmask[None, None], float("-inf"))
    KT = getattr(K, "KT", 32)
    seed_max = sc[..., :KT].amax(dim=-1, keepdim=True)  # first interior tile's max as the frozen seed
    peakP = (sc - seed_max).clamp(max=80).exp().nan_to_num(0.0).amax().item()

    err = (Og.float().cpu() - ref.float()).abs().max().item()
    ok = err < 6e-2
    print(
        f"[ADVERSARIAL] {mod} sq{sq} KT={KT} FREEZE={getattr(K,'_FREEZE','?')} FEXP={getattr(K,'_FEXP','?')} "
        f"-> peak frozen-P(vs tile0 seed)={peakP:.3g} (>240 => overflow path exercised); "
        f"ERR {err:.4f} {'OK' if ok else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
