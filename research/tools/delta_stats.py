"""Cross-token structure of the activations of verification passes (GGML_SRV_DUMP).

Hypothesis: the M tokens of one pass multiply the same weights; if x_t - x_ref is small or sparse at int8 precision,
W x_t = W x_ref + W (x_t - x_ref) could cost much less than M full products.
usage: python research/tools/delta_stats.py ~/act_dump.bin
"""
import sys
from collections import defaultdict

import numpy as np

NAMES = {0: "iq4_xs", 1: "iq3_s", 2: "iq3_xxs", 3: "q3_k", 7: "q6_k", 8: "q4_k", 9: "q5_k"}


def records(path):
    raw = open(path, "rb").read()
    off = 0
    while off + 12 <= len(raw):
        M, K, ty = np.frombuffer(raw, np.int32, 3, off)
        off += 12
        n = int(M) * int(K)
        if off + 4 * n > len(raw):
            break
        X = np.frombuffer(raw, np.float32, n, off).reshape(int(M), int(K)).astype(np.float64)
        off += 4 * n
        yield int(ty), X


def main():
    st = defaultdict(lambda: defaultdict(list))
    for ty, X in records(sys.argv[1]):
        M, K = X.shape
        name = NAMES.get(ty, str(ty))
        s = st[name]
        nrm = np.linalg.norm(X, axis=1)
        # 1. relative size of the difference to the first token and to the mean
        for ref_name, ref in (("first", X[0]), ("mean", X.mean(axis=0))):
            d = np.linalg.norm(X[1:] - ref, axis=1) / nrm[1:]
            s[f"rel_{ref_name}"].append(d.mean())
        s["rel_prev"].append((np.linalg.norm(X[1:] - X[:-1], axis=1) / nrm[1:]).mean())
        # 2. int8 with a scale shared by all tokens per 256 (max over tokens): delta to the mean, per 32-block
        B = X.reshape(M, K // 256, 256)
        sc = np.abs(B).max(axis=(0, 2), keepdims=True) / 127 + 1e-30
        q = np.rint(B / sc).astype(np.int32).reshape(M, K // 32, 32)
        ref = np.rint(q.mean(axis=0)).astype(np.int32)
        dq = q - ref
        mx = np.abs(dq).max(axis=2)  # max |delta| per 32-block, per token
        s["zero_blocks"].append((mx == 0).mean())
        s["fits_2bit"].append((mx <= 1).mean())
        s["fits_4bit"].append((mx <= 7).mean())
        s["mean_abs_q"].append(np.abs(q).mean())
        s["mean_abs_dq"].append(np.abs(dq).mean())
        # 3. effective rank: energy of the top singular directions of the M x K batch
        sv = np.linalg.svd(X, compute_uv=False) ** 2
        s["energy_top1"].append(sv[0] / sv.sum())
        s["energy_top2"].append(sv[:2].sum() / sv.sum())
        s["M"].append(M)
    for name, s in st.items():
        n = len(s["M"])
        print(f"== {name}: {n} calls, M mean {np.mean(s['M']):.1f}")
        print(f"   ||x_t - x_0|| / ||x_t||  {np.mean(s['rel_first']):.2f}   vs mean {np.mean(s['rel_mean']):.2f}   vs previous token {np.mean(s['rel_prev']):.2f}")
        print(f"   int8, shared scale, delta to the mean per 32-block: zero {100 * np.mean(s['zero_blocks']):.1f} %   |d|<=1 {100 * np.mean(s['fits_2bit']):.1f} %   |d|<=7 {100 * np.mean(s['fits_4bit']):.1f} %")
        print(f"   mean |q| {np.mean(s['mean_abs_q']):.2f}  mean |delta| {np.mean(s['mean_abs_dq']):.2f}")
        print(f"   energy in top-1 / top-2 directions of the batch: {100 * np.mean(s['energy_top1']):.1f} % / {100 * np.mean(s['energy_top2']):.1f} %")


if __name__ == "__main__":
    main()
