# Ladder 06 — Cooperative K/V into LDS (Lesson 10)

One change vs 05: the `NWAVES` waves all attend the same kv range, so per-wave global K/V loads are
redundant. The workgroup **cooperatively stages** the K tile `[kv,hd]` and V tile `[d,kv]` into LDS
once per kv-tile (all `NWAVES*64` threads), `barrier`, then every wave reads the shared tile. Q stays
per-wave.

This removes the redundant global traffic; the cost is a workgroup barrier per kv-tile. Measured
win at sq=2048 (single head): **6.56 TF vs 5.24 TF (+25%)** over rung 05. This is the full structural
kernel (multiwave + column-V + register-P + cooperative-LDS) — the layout-algebra sibling of
`lesson_22b`, with `fx.gemm` driving the MFMAs.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/06_cooperative_lds.py`
