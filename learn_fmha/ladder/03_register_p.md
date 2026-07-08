# Ladder 03 — Register-resident P (Lesson 12)

One change vs 02: the P-transpose (P is produced as `S[kv,q]` but GEMM2 needs `P[q,kv]`) is done
**across lanes with `ds_bpermute`** instead of storing P to LDS, barrier, and reloading. This kernel
uses **no LDS at all**.

The decisive lesson: `ds_bpermute` is itself an LDS-unit op, so register-P *moves* the transpose
traffic rather than deleting it — roughly neutral (~28 us). It sets up rung 04 (column-V), which
*deletes* the transpose. The 32x32x16 ds_bpermute pattern (2 kv sub-groups x 2 packed dwords x 2
half destinations) is the verified pattern from `lesson_22b`.

Run: `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/ladder/03_register_p.py`
