import sys, os, importlib
sys.path.insert(0, "tests/kernels")
sys.path.insert(0, "kernels")
import torch
import fmha_prefill_fp8_ref as R

MOD = os.environ.get("FMHA_MOD", "fmha_prefill_fp8_pingpong")
b, sq, sk, nk, gqa, causal, ps, pscale = (
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]),
)
K = importlib.import_module(MOD)
torch.manual_seed(0)
HD = K.HD
nq = nk * gqa
sm = 1.0 / HD**0.5
q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
qf, qd = R.quantize_per_token_head(q)
kf, kd = R.quantize_per_token_head(k)
vf, vd = R.quantize_per_head(v)
cache = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
args = [qf.to("cuda"), cache.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        cache.v_pool.view(torch.float8_e4m3fnuz).to("cuda"), qd.to("cuda"), kd.to("cuda"),
        vd.to("cuda"), cache.page_ids.to("cuda"), cache.kv_indptr.to("cuda"),
        torch.full((b * nq,), pscale, device="cuda", dtype=torch.float32)]
Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
grid = b * nq * ((sq + K.BM - 1) // K.BM)
K.run_attn(*args, Og, sq, sk, nq, nk, ps, cache.k_page_stride, cache.v_page_stride, sm, causal, grid)
torch.cuda.synchronize()
ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))
err = (Og.float().cpu() - ref.float()).abs().max().item()
print(f"{MOD} sq{sq} causal={causal}: ERR {err:.5f}  ->  {'PASS' if err < 6e-2 else 'FAIL'}")
sys.exit(0 if err < 6e-2 else 1)
