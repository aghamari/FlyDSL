# SPDX-License-Identifier: Apache-2.0
"""The idiomatic FlyDSL register-resident GEMM, stripped to its essence.

ONE wavefront computes ONE MFMA-shaped output tile, looping over K — the same
thing lesson_02 does by hand, but in the layout-algebra style of
`examples/03-tiledMma.py`: no `.ir_value()`, no `buffer_ops`, no manual
lane↔element math. `make_tiled_mma` derives the A/B/C fragment layouts; tiled
copy atoms move global↔register; the MFMA is the one raw opcode.

Swap `SHAPE` between '16x16x16' and '32x32x8' — that one line (plus its matching
opcode/acc-width in MFMA below) is the entire "change the MFMA instruction".

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/minimal_tiled_gemm.py
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

SHAPE = "32x32x8"   # or "16x16x16"
K = 64              # contraction length (a few MFMA K-steps)

MFMA = {
    "16x16x16": ((16, 16, 16), fx.rocdl.mfma_f32_16x16x16bf16_1k, 4),
    "32x32x8":  ((32, 32, 8),  fx.rocdl.mfma_f32_32x32x8bf16_1k, 16),
}
(m, n, k), opcode, acc_n = MFMA[SHAPE]


@flyc.kernel(known_block_size=[64, 1, 1])
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    gA_k = fx.flat_divide(A, (m, k))[None, None, 0, None]   # (m, k, nK)
    gB_k = fx.flat_divide(B, (n, k))[None, None, 0, None]   # (n, k, nK)
    gC = fx.flat_divide(C, (m, n))[None, None, 0, 0]        # (m, n)

    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(m, n, k, fx.BFloat16)),
        fx.make_layout((1, 1, 1), (0, 0, 0)),
    )
    thr_mma = tiled_mma.thr_slice(tid)

    cp_ab = fx.make_copy_atom(fx.rocdl.BufferCopy((m * k // 64) * 16), fx.BFloat16)
    tcA = fx.make_tiled_copy_A(cp_ab, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(cp_ab, tiled_mma).get_slice(tid)

    thr_gA = tcA.partition_S(gA_k)
    thr_gB = tcB.partition_S(gB_k)
    frag_A = thr_mma.make_fragment_A(gA_k[None, None, 0])
    frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0)

    f32xacc = fx.typing.T.vec(acc_n, fx.typing.T.f32)
    for kt in fx.range_constexpr(K // k):
        fx.copy(cp_ab, thr_gA[None, None, None, kt], tcA.retile(frag_A))
        fx.copy(cp_ab, thr_gB[None, None, None, kt], tcB.retile(frag_B))
        a = frag_A.load().bitcast(fx.Int16)
        b = frag_B.load().bitcast(fx.Int16)
        frag_C.store(fx.Vector(opcode(f32xacc, [a, b, frag_C.load()])))

    cp_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcC = fx.make_tiled_copy_C(cp_c, tiled_mma).get_slice(tid)
    fx.copy(cp_c, tcC.retile(frag_C), tcC.partition_S(gC))


@flyc.jit
def run(A, B, C, stream: fx.Stream = fx.Stream(None)):
    gemm(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(n, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    run(A, B, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    err = (C - A.float() @ B.float().T).abs().max().item()
    print(f"{SHAPE}: max abs err = {err:.5f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
