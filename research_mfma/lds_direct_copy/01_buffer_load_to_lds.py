# SPDX-License-Identifier: Apache-2.0
"""Direct global -> LDS DMA via `buffer_load_to_lds` (CDNA3 / gfx942).

Unlike the `lds_bank_conflicts/` examples (global -> register -> LDS), this uses the
AMD `buffer_load_dword ... lds` path: the load lands DIRECTLY in LDS, never touching a
VGPR. This is CK's "global_to_lds" path. It saves registers and a copy, at the cost of
a manual LDS destination pointer + an explicit `s_waitcnt vmcnt(0)` before the barrier.

Hardware addressing (validated by tests/kernels/lds_dma_probe.py):
  - The per-lane LDS destination pointer is IGNORED; hw writes LDS[M0 + lane*4], where
    M0 is the *uniform* base pointer (per wave). So for multi-wave blocks each wave must
    pass its own base M0 = base + wave*64*4; the per-lane spread comes from `voffset`.
  - `voffset` is the per-lane BYTE offset into the global buffer.

Demo: copy N f32 from global IN -> LDS -> global OUT (identity roundtrip), verify.

Run:  HIP_VISIBLE_DEVICES=2 python3 research_mfma/lds_direct_copy/01_buffer_load_to_lds.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

NTHREADS = 256
WARP = 64
N = NTHREADS                       # one f32 per thread
VMCNT0 = 0x3F70                    # s_waitcnt vmcnt(0): wait for the LDS DMA to land

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="lds_direct_dma_smem")
_alloc.ptr = N * 4


@flyc.kernel(known_block_size=[NTHREADS, 1, 1])
def k_dma(IN: fx.Tensor, OUT: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    rin = fx.buffer_ops.create_buffer_resource(IN)
    rout = fx.buffer_ops.create_buffer_resource(OUT)

    # LDS scratch + a raw LDS pointer (address space 3) to its base.
    lds = SmemPtr(_alloc.get_base(), 0, fx.typing.T.f32, shape=(N,)).get()
    lds_base = memref.extract_aligned_pointer_as_index(lds)
    lds_ptr_base = fx.buffer_ops.create_llvm_ptr(
        arith.index_cast(fx.typing.T.i64, lds_base), address_space=3
    )

    # multi-wave: each wave's uniform LDS base is M0 = base + wave*64*4
    wave = tid // fx.Int32(WARP)
    lds_ptr = fx.buffer_ops.get_element_ptr(lds_ptr_base, byte_offset=fx.Index(wave * fx.Int32(WARP * 4)))

    # direct DMA: global[voffset] -> LDS[M0 + lane*4], no VGPR
    fx.rocdl.buffer_load_to_lds(rin, lds_ptr, tid * fx.Int32(4), size_bytes=4)
    fx.rocdl.s_waitcnt(VMCNT0)
    fx.gpu.barrier()

    val = memref.load(lds, [fx.Index(tid).ir_value()])
    fx.buffer_ops.buffer_store(val, rout, tid)


@flyc.jit
def run(IN, OUT, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    k_dma(IN, OUT).launch(grid=(1, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    IN = torch.randn(N, dtype=torch.float32, device="cuda")
    OUT = torch.zeros(N, dtype=torch.float32, device="cuda")
    run(IN, OUT, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    err = (OUT - IN).abs().max().item()
    print(f"\nbuffer_load_to_lds (direct global->LDS DMA)")
    print(f"  N={N} f32 roundtrip   max abs err = {err:.6g}  ->  {'PASS' if err < 1e-6 else 'FAIL'}")
