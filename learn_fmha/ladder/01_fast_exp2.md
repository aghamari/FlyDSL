# Ladder 01 — Fast exp2 (Lesson 13)

One change vs 00: the softmax exp uses hardware `rocdl.exp2` (1 VALU) instead of
`Float32.exp2()` (which also emits a `v_ldexp` range-reduction pair, ~3 VALU). Math identical
(`exp(x)=2^(x*log2e)`; softmax evaluates `exp(s-m)` with `s-m<=0`, so the fast path is range-safe).

A pure VALU micro-optimization: ~neutral on this occupancy-starved single-wave shape (~29.6 us),
but it removes 2 VALU ops per exp in a kernel that is VALU-heavy once scaled up.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/01_fast_exp2.py`
