"""Are the activations of the verification matmuls dominated by a few fixed outlier channels?

For each dumped call (GGML_SRV_DUMP): channels with |x| > 8 x the token's RMS, whether they are the same across the
tokens of the pass and across calls of the same shape, and the int8 quantization error per 256-block with the
current scheme (8 bits, scale = block max) against 7 bits with the outlier channels kept aside.
usage: python research/tools/outlier_stats.py ~/act_dump.bin
"""
import sys
from collections import Counter, defaultdict

import numpy as np

from research.tools.delta_stats import NAMES, records


def qerr(x, bits, mask=None):
    """relative L2 error of per-256 max-scaled symmetric quantization; masked channels are kept exact."""
    lim = 2 ** (bits - 1) - 1
    y = x.copy()
    keep = np.zeros_like(x, dtype=bool) if mask is None else mask
    xb = np.where(keep, 0.0, x).reshape(-1, 256)
    sc = np.abs(xb).max(axis=1, keepdims=True) / lim + 1e-30
    qb = np.clip(np.rint(xb / sc), -lim, lim) * sc
    y = np.where(keep, x, qb.reshape(x.shape))
    return np.linalg.norm(x - y) / np.linalg.norm(x)


def main():
    per = defaultdict(lambda: {"n_out": [], "shared": [], "e8": [], "e7": [], "e7o": [], "e8o": [], "chan": Counter(), "calls": 0})
    for ty, X in records(sys.argv[1]):
        M, K = X.shape
        key = (NAMES.get(ty, str(ty)), K)
        s = per[key]
        s["calls"] += 1
        rms = np.sqrt((X ** 2).mean(axis=1, keepdims=True))
        out = np.abs(X) > 8 * rms
        s["n_out"].append(out.sum(axis=1).mean())
        common = out.all(axis=0)
        s["shared"].append(common.sum() / max(1, out.any(axis=0).sum()))
        for c in np.nonzero(out.any(axis=0))[0]:
            s["chan"][int(c)] += 1
        for t in range(M):
            s["e8"].append(qerr(X[t], 8))
            s["e7"].append(qerr(X[t], 7))
            s["e7o"].append(qerr(X[t], 7, out[t]))
            s["e8o"].append(qerr(X[t], 8, out[t]))
    for (name, K), s in sorted(per.items()):
        top = s["chan"].most_common(8)
        stable = sum(c for _, c in top) / max(1, s["calls"] * len(top))
        print(f"== {name} K={K}: {s['calls']} calls")
        print(f"   outlier channels per token (|x| > 8 rms): {np.mean(s['n_out']):.1f}   shared by all tokens of the pass: {100 * np.mean(s['shared']):.0f} %")
        print(f"   most frequent channels {[c for c, _ in top]}  present in {100 * stable:.0f} % of the calls")
        print(f"   quantization error: 8 bit {np.mean(s['e8']):.4f}   7 bit {np.mean(s['e7']):.4f}   "
              f"7 bit + outliers aside {np.mean(s['e7o']):.4f}   8 bit + outliers aside {np.mean(s['e8o']):.4f}")


if __name__ == "__main__":
    main()
