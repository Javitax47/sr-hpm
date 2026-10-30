"""How much does each CPU layer change the hidden state? (candidates for layer dropping, ShortGPT / Gromov et al.)

From per-pass dumps of every CPU layer's input (GGML_SRV_DUMP_NAMES with blk.L.attn_qkv / attn_q, = attn_norm(h_L)),
h_L / rms(h_L) is recovered by dividing out the norm weight, and the angular distance arccos(cos(h_L, h_{L+n})) / pi is
averaged over tokens. Small distance = the block of n layers barely changes the residual stream.

usage: python -m research.tools.layer_redundancy ~/dump4.bin
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

from research.tools.lens_test import read_dump  # noqa: E402


def main():
    recs = read_dump(sys.argv[1])
    T = {t.name: t for t in GGUFReader(str(Path.home() / "f0_gguf/q_speed1.gguf")).tensors}
    layers = sorted(recs)
    n = min(len(recs[l]) for l in layers)
    H = {}
    for l in layers:
        w = np.asarray(T[f"blk.{l}.attn_norm.weight"].data, np.float32)
        H[l] = [x / w for x in recs[l][:n]]
    def dist(a, b):
        ds = []
        for xa, xb in zip(H[a], H[b]):
            if xa.shape != xb.shape:
                continue
            c = (xa * xb).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1))
            ds.extend(np.arccos(np.clip(c, -1, 1)) / np.pi)
        return float(np.mean(ds))
    print(f"{n} passes per layer")
    print("single layers (input L -> input L+1):")
    single = [(l, dist(l, l + 1)) for l in layers if l + 1 in H]
    for l, d in single:
        print(f"  layer {l:2d}: {d:.4f}")
    for nb in (2, 4, 6, 8):
        best = min(((l, dist(l, l + nb)) for l in layers if l + nb in H), key=lambda t: t[1])
        print(f"most redundant block of {nb}: layers {best[0]}..{best[0] + nb - 1}, distance {best[1]:.4f}")


if __name__ == "__main__":
    main()
