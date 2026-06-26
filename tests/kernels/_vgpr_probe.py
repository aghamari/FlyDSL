import sys, os, glob, re, importlib
sys.path.insert(0, "tests/kernels"); sys.path.insert(0, "kernels")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
import torch
import fmha_prefill_fp8_ref as R

MOD = os.environ.get("FMHA_MOD", "fmha_prefill_fp8_layout_small")
dump = f"/tmp/vg_{MOD}_{os.environ.get('FMHA_NWAVES','4')}"
os.system(f"rm -rf {dump}")
os.environ["FLYDSL_DUMP_IR"] = "1"; os.environ["FLYDSL_DUMP_DIR"] = dump
K = importlib.import_module(MOD)
b, sq, nk, gqa, ps = 1, 2048, 1, 8, 16
nq = nk * gqa; sm = 1.0 / 128 ** 0.5
torch.manual_seed(0)
q = torch.randn(b, sq, nq, 128); k = torch.randn(b, sq, nk, 128); v = torch.randn(b, sq, nk, 128)
qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
args = [qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"), c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"), kd.to("cuda"), vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda")]
Og = torch.zeros(b, sq, nq, 128, device="cuda", dtype=torch.bfloat16)
grid = b * nq * ((sq + K.BM - 1) // K.BM)
K.run_attn(*args, Og, sq, sq, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, 1, grid)
torch.cuda.synchronize()
out = {}
for f in glob.glob(f"{dump}/**/21_final_isa.s", recursive=True) + glob.glob(f"{dump}/**/19_gpu_module_to_binary.mlir", recursive=True):
    txt = open(f, errors="ignore").read()
    for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "group_segment_fixed_size"):
        m = re.search(rf"{key}\D+(\d+)", txt)
        if m: out[key] = int(m.group(1))
print(f"{MOD} NWAVES={os.environ.get('FMHA_NWAVES','4')}: {out}")
