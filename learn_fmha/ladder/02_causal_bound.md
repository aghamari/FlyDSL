# Ladder 02 — Causal kv-bound (Lesson 14)

One change vs 01: cap the kv-tile loop at the causal limit. A q-tile at rows
`[blk*BM, blk*BM+BM)` can only attend to `kv <= (blk*BM+BM-1) + (sk-sq)`, so we set the loop count
`n_kv` to the last tile with any valid kv instead of looping all `ceil(sk/BN)` and masking the tail.

Cuts ~half the kv work on the causal triangle — but cutting *work* doesn't help an *under-occupied*
machine, so wall time barely moves here (~28.3 us). It becomes a real win once the grid fills the CUs.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/02_causal_bound.py`
