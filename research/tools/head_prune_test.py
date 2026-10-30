"""Refutation test: exact argmax of the output head over a small part of the vocabulary.

If the 248k rows of lm_head are grouped (k-means) and each group j gets an upper bound on its logits
    max_{w in j} w.h <= c_j.h + r_j * ||h||          (centroid c_j, radius r_j = max ||w - c_j||)
then only the groups whose bound reaches the best logit found so far need to be read, and the argmax stays exact.
If that is a small fraction, the head could live on the CPU (reading a few % of it) and free its VRAM for more layers.
Inputs: per-pass dumps of the head input (GGML_SRV_DUMP_NAMES=<file>,output.weight with the head on the CPU).

usage: python -m research.tools.head_prune_test ~/dump5.bin
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from research.tools.lens_test import read_dump  # noqa: E402


def main():
    recs = read_dump(sys.argv[1])
    H = torch.tensor(np.concatenate([x for v in recs.values() for x in v]), device="cuda")  # [n, 5120]
    t = {t.name: t for t in GGUFReader(str(Path.home() / "f0_gguf/q_speed1.gguf")).tensors}["output.weight"]
    rows = t.data.shape[0]
    W = torch.empty(rows, 5120, dtype=torch.float16, device="cuda")
    for i in range(0, rows, 16384):
        W[i:i + 16384] = torch.tensor(dequantize(t.data[i:i + 16384], t.tensor_type), dtype=torch.float16, device="cuda")
    print(f"{H.shape[0]} head inputs, lm_head {tuple(W.shape)}", flush=True)
    Hh = H.half()
    L = (Hh @ W.T).float()  # [n, vocab]
    top2 = L.topk(2, dim=1).values
    print(f"top-1 logit mean {float(top2[:, 0].mean()):.2f}, gap top1-top2 mean {float((top2[:, 0] - top2[:, 1]).mean()):.2f}, "
          f"||h|| mean {float(H.norm(dim=1).mean()):.1f}, ||w|| mean {float(W.float().norm(dim=1).mean()):.3f}", flush=True)
    hn = H.norm(dim=1)
    for K in (1024, 4096, 16384):
        # k-means (Euclidean), a few Lloyd iterations from random rows
        g = torch.Generator(device="cuda").manual_seed(0)
        C = W[torch.randperm(rows, generator=g, device="cuda")[:K]].float()
        for _ in range(8):
            a = torch.empty(rows, dtype=torch.long, device="cuda")
            for i in range(0, rows, 32768):
                w = W[i:i + 32768].float()
                d = (w * w).sum(1, keepdim=True) - 2 * w @ C.T + (C * C).sum(1)
                a[i:i + 32768] = d.argmin(1)
            Cn = torch.zeros_like(C).index_add_(0, a, W.float())
            cnt = torch.bincount(a, minlength=K).clamp(min=1).unsqueeze(1).float()
            C = Cn / cnt
        r = torch.zeros(K, device="cuda")
        for i in range(0, rows, 32768):
            w = W[i:i + 32768].float()
            r.scatter_reduce_(0, a[i:i + 32768], (w - C[a[i:i + 32768]]).norm(dim=1), reduce="amax")
        bound = H @ C.T + r.unsqueeze(0) * hn.unsqueeze(1)  # [n, K]
        need = (bound >= top2[:, :1]).float()  # groups that must be read for an exact argmax
        sizes = torch.bincount(a, minlength=K).float()
        frac_rows = float(((need * sizes).sum(1) / rows).mean())
        print(f"K={K:5d}: mean radius {float(r.mean()):.3f}, groups to read {float(need.mean()):.1%}, "
              f"rows to read {frac_rows:.1%} of the head", flush=True)


if __name__ == "__main__":
    main()
