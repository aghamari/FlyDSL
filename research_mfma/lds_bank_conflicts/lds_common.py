# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the LDS bank-conflict scenarios (FlyDSL port of CK-Tile
`tutorial_14_bank_conflict_scenarios`).

Every scenario does the SAME job — transpose X[M,K] -> Y[K,M] through LDS — and
changes ONLY the LDS layout. The whole kernel is written with layout algebra; there
is no hand-rolled address arithmetic:

  * global tiles come from `flat_divide` + a dynamic block-coordinate slice, then are
    indexed directly (`gX[m, k]`, `gY[yr, yc]`) — no `(bm*TM+m)*K+...`.
  * the LDS is a layout-bearing tensor `make_view(get_dyn_shared(f32), lds_layout)`;
    indexing `s[m, k]` applies the layout (incl. swizzle), so padding / XOR / order
    live entirely in the layout — no `m*stride+col`, no manual `^`.
  * linear thread/block ids become 2-D coordinates via `idx2crd` — no `//` / `%`.

Correctness holds for ANY layout: a layout is a bijection used identically on the
write and the transposed read; only the bank-conflict behaviour differs.

LDS banks (gfx942): 32 x 4 B. For f32, bank(addr_elems) = addr_elems % 32. A wave64
access spans 64 addresses over 32 banks, so the conflict-free floor is 2-way.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# ── Problem + tile config ────────────────────────────────────────────────────
M = 4096
K = 4096
TM = 64                         # tile rows (m)
TK = 64                         # tile cols (k)
BLOCK = 256
ITERS = (TM * TK) // BLOCK      # 16 elements per thread
NMT = M // TM                   # block tiles along M
NKT = K // TK                   # block tiles along K
NGRID = NMT * NKT


def build(name, lds_elems, make_lds_layout):
    """Build a transpose kernel whose LDS layout is supplied as a (traced) callable."""

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def k_transpose(X: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # block id -> 2-D tile coordinate (idx2crd, not // and %)
        bcrd = fx.idx2crd(fx.Int32(bid), fx.make_ordered_layout((NMT, NKT), (1, 0)))
        bm = fx.Int32(fx.get(bcrd, 0))
        bk = fx.Int32(fx.get(bcrd, 1))

        # this block's input / output tiles (layout algebra, no address math)
        gX = fx.flat_divide(fx.rocdl.make_buffer_tensor(X), (TM, TK))[None, None, bm, bk]
        gY = fx.flat_divide(fx.rocdl.make_buffer_tensor(Y), (TK, TM))[None, None, bk, bm]

        # LDS as a layout-bearing tensor: the chosen layout (order / padding / swizzle)
        # IS the bank-conflict knob; s[i,j] applies it automatically.
        s = fx.Tensor(fx.make_view(fx.get_dyn_shared(fx.Float32), make_lds_layout()))

        # this thread's it-th tile element, as a 2-D coordinate (no // or %)
        coord = fx.make_ordered_layout((TM, TK), (1, 0))

        def elem(it):
            c = fx.idx2crd(fx.Int32(tid) + fx.Int32(it * BLOCK), coord)
            return fx.Int32(fx.get(c, 0)), fx.Int32(fx.get(c, 1))

        for it in fx.range_constexpr(ITERS):        # write: global tile -> LDS
            i, j = elem(it)
            s[i, j] = gX[i, j]
        fx.gpu.barrier()
        for it in fx.range_constexpr(ITERS):        # read transposed: LDS -> global tile
            i, j = elem(it)
            gY[i, j] = s[j, i]

    @flyc.jit
    def run(X, Y, stream: fx.Stream = fx.Stream(None)):
        k_transpose(X, Y).launch(
            grid=(NGRID, 1, 1), block=(BLOCK, 1, 1), stream=stream, smem=lds_elems * 4
        )

    return run


def run_case(name, title, lds_elems, make_lds_layout, *, expect):
    """Build, verify against torch, and benchmark one LDS layout scenario."""
    run = build(name, lds_elems, make_lds_layout)

    torch.manual_seed(0)
    X = torch.randn(M, K, dtype=torch.float32, device="cuda")
    Y = torch.zeros(K, M, dtype=torch.float32, device="cuda")
    s = torch.cuda.current_stream()

    run(X, Y, stream=s)
    torch.cuda.synchronize()
    err = (Y - X.t().contiguous()).abs().max().item()
    ok = err < 1e-6

    for _ in range(10):
        run(X, Y, stream=s)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    rep = 100
    for _ in range(rep):
        run(X, Y, stream=s)
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) / rep * 1e3
    gbps = 2 * M * K * 4 / 1e9 / (us / 1e6)

    print(f"\n{title}")
    print(f"  X[{M},{K}] f32 -> Y[{K},{M}]   tile {TM}x{TK}, LDS {lds_elems} elems ({lds_elems*4} B)")
    print(f"  correctness : {'PASS' if ok else 'FAIL'}  (max abs err {err:.6g})")
    print(f"  time        : {us:8.2f} us   |  {gbps:6.0f} GB/s")
    print(f"  expect      : {expect}")
    return us, gbps, ok
