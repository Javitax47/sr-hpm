"""Order-0 entropy of the quantized weight fields of a GGUF (how far a lossless re-encoding could shrink them).

usage: python research/tools/entropy.py <model.gguf> [first_layer]
IQ4_XS: 4-bit codes, 6-bit block scales. IQ3_S: 9-bit grid indices (4 weights each), sign bytes, 4-bit scales.
"""
import math
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402


def H(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    path, first = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    r = GGUFReader(path)
    acc = defaultdict(lambda: {"bytes": 0, "bits": 0.0})
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if not m or int(m.group(1)) < first:
            continue
        ty = t.tensor_type.name
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        if ty == "IQ4_XS":
            b = raw.reshape(-1, 136)
            qs = b[:, 8:]
            codes = np.concatenate([qs & 15, qs >> 4], axis=None)
            hq = H(np.bincount(codes, minlength=16))
            # 6-bit scales: low nibbles in scales_l (4 bytes), high 2 bits in scales_h (2 bytes)
            sl = b[:, 4:8]
            sh = b[:, 2].astype(np.uint16) | (b[:, 3].astype(np.uint16) << 8)
            sc = []
            for j in range(8):
                lo = (sl[:, j // 2] >> (4 * (j % 2))) & 15
                hi = (sh >> (2 * j)) & 3
                sc.append(lo | (hi << 4))
            hs = H(np.bincount(np.concatenate(sc), minlength=64))
            n = b.shape[0]
            bits = n * (256 * hq + 8 * hs + 16)
        elif ty == "IQ3_S":
            b = raw.reshape(-1, 110)
            qs, qh, signs, scales = b[:, 2:66], b[:, 66:74], b[:, 74:106], b[:, 106:110]
            idx = []
            for k in range(8):
                hb = ((qh[:, k:k + 1] >> np.arange(8)) & 1).astype(np.uint16)
                idx.append(qs[:, 8 * k:8 * k + 8].astype(np.uint16) | (hb << 8))
            hi = H(np.bincount(np.concatenate(idx, axis=None), minlength=512))
            sbits = np.unpackbits(signs.reshape(-1))
            hsg = H(np.bincount(sbits, minlength=2))
            hsc = H(np.bincount(np.concatenate([scales & 15, scales >> 4], axis=None), minlength=16))
            n = b.shape[0]
            bits = n * (64 * hi + 256 * hsg + 8 * hsc + 16)
        else:
            continue
        acc[ty]["bytes"] += raw.size
        acc[ty]["bits"] += bits
        acc[ty].setdefault("h", []).append((hq if ty == "IQ4_XS" else hi))
    tot_b, tot_e = 0, 0.0
    for ty, a in acc.items():
        e = a["bits"] / 8
        tot_b += a["bytes"]
        tot_e += e
        print(f"{ty:8s} {a['bytes'] / 1e9:5.2f} GB -> entropy bound {e / 1e9:5.2f} GB ({100 * e / a['bytes']:.1f} %)"
              f"   code entropy {np.mean(a['h']):.3f} bits")
    print(f"total    {tot_b / 1e9:5.2f} GB -> {tot_e / 1e9:5.2f} GB ({100 * tot_e / tot_b:.1f} %)")


if __name__ == "__main__":
    main()
