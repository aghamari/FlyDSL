# SPDX-License-Identifier: Apache-2.0
"""A tiny, runnable, plain-Python simulation of what `retile` does.

No GPU, no FlyDSL — just lists and index functions, so you can run it and read
the output:   python3 research_mfma/retile_demo.py

THE SETUP (one thread, 4 registers)
-----------------------------------
A thread owns 4 physical registers. The SAME 4 registers are addressed two
different ways by two different consumers:

  * the MMA  (fx.gemm)  reads them in its FRAGMENT layout
  * the COPY (fx.copy)  writes them in its TILED-COPY layout

These two layouts are NOT the same. `retile(frag)` builds a *view* of the
fragment in the copy's coordinates, so that when fx.copy writes value `v`, it
lands in the exact physical register that fx.gemm will later read.

We model the fragment's 4 values as a 2x2 block to make the mismatch visible:
  * the MMA reads the 2x2 COLUMN-MAJOR
  * the source data arrives ROW-MAJOR (the copy's order)
"""


# The MMA fragment layout: (row, col) -> which physical register.
# Column-major over the 2x2 block:  reg = row + 2*col
def mma_layout(row, col):
    return row + 2 * col


# `retile` composes the copy's flat coordinate v=0..3 with the fragment layout:
#   v -> logical (row, col) [row-major] -> mma_layout -> physical register
# THIS composition is exactly what retile(frag) builds for you.
def retile_copy_to_physical(v):
    row, col = v // 2, v % 2
    return mma_layout(row, col)


class View:
    """A view over shared storage: __setitem__ writes via an index function."""
    def __init__(self, storage, index_fn):
        self.storage = storage
        self.index_fn = index_fn

    def __setitem__(self, v, value):
        self.storage[self.index_fn(v)] = value


def run(index_fn, label):
    regs = [None, None, None, None]          # this thread's physical registers

    # Global source, delivered in the COPY's (row-major) order, v = 0..3:
    #   v=0 ->(0,0)  v=1 ->(0,1)  v=2 ->(1,0)  v=3 ->(1,1)
    src = ["a00", "a01", "a10", "a11"]

    dst = View(regs, index_fn)               # the (possibly retiled) destination
    for v in range(4):                       # this loop == fx.copy
        dst[v] = src[v]

    print(f"\n{label}")
    print(f"  physical regs after copy: {regs}")
    # fx.gemm reads the fragment in MMA (column-major) coordinates:
    ok = True
    for row in range(2):
        for col in range(2):
            got = regs[mma_layout(row, col)]
            want = f"a{row}{col}"
            mark = "" if got == want else "   <-- WRONG"
            if got != want:
                ok = False
            print(f"  mma frag[{row},{col}] = regs[{mma_layout(row,col)}] = {got}{mark}")
    print(f"  => {'CORRECT' if ok else 'SCRAMBLED'}")


if __name__ == "__main__":
    # WITH retile: copy coord -> logical -> physical, so MMA reads correct data.
    run(retile_copy_to_physical, "WITH retile  (copy writes through the fragment's layout)")

    # WITHOUT retile: copy writes flat v straight into regs[v], ignoring the
    # layout mismatch -> the MMA then reads scrambled values.
    run(lambda v: v, "WITHOUT retile (copy writes regs[v] directly)")
