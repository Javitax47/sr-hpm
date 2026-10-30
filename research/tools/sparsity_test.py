"""Refutation test: activation sparsity with one mask shared by the whole verification pass.

TEAL-style sparsity (skip the weight columns whose input activation is ~0) is defined per token; a verification pass
multiplies 2-8 tokens by the same weights, so the columns that can be skipped are those small for EVERY token of the
pass. Blocks of 32 or 256 inputs (the kernels' sub-block / super-block) are dropped by the smallest score
    s_j = sum_tokens sum_{k in block j} x[t, k]^2 * ||W[:, k]||^2
and the relative output error ||W x_masked - W x||^2 / ||W x||^2 is compared with what the IQ4_XS quantization of the
same matrix already costs. Inputs: the per-pass dumps of GGML_SRV_DUMP_NAMES (attn_norm output = input of attn_qkv).

usage: python -m research.tools.sparsity_test ~/dump2.bin [attn_qkv | ffn_down]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from research.tools.lens_test import read_dump  # noqa: E402

G = Path.home() / "f0_gguf"
KEEPS = (0.9, 0.8, 0.7, 0.6, 0.5)


def main():
    recs = read_dump(sys.argv[1])
    q8 = {t.name: t for t in GGUFReader(str(G / "Qwen3.8-27B-Q8_0.gguf")).tensors}
    iq4 = {t.name: t for t in GGUFReader(str(G / "q_cpu_iq4.gguf")).tensors}
    rng = np.random.default_rng(0)
    for layer in sorted(recs):
        name = f"blk.{layer}.{sys.argv[2] if len(sys.argv) > 2 else 'attn_qkv'}.weight"
        W = dequantize(q8[name].data, q8[name].tensor_type).astype(np.float32)  # [out, in]
        Wq = dequantize(iq4[name].data, iq4[name].tensor_type).astype(np.float32)
        c = (W.astype(np.float64) ** 2).sum(axis=0)  # column norms^2, [in]
        rows = [x for x in recs[layer] if x.shape[0] >= 2]
        pick = [rows[i] for i in rng.choice(len(rows), size=min(60, len(rows)), replace=False)]
        Ms = [x.shape[0] for x in pick]
        # quantization error of the same product, for scale
        eq = np.mean([float(((x @ (Wq - W).T) ** 2).sum() / ((x @ W.T) ** 2).sum()) for x in pick])
        print(f"layer {layer}: {len(pick)} passes (M {min(Ms)}-{max(Ms)}), IQ4_XS error of the product {eq:.2e}")
        for B in (32, 256):
            nb = W.shape[1] // B
            for mode in ("per-token", "shared"):
                errs = {k: [] for k in KEEPS}
                for x in pick:
                    y = x @ W.T
                    ny = float((y ** 2).sum())
                    e = (x.astype(np.float64) ** 2) * c  # [M, in]
                    eb = e.reshape(x.shape[0], nb, B).sum(axis=2)  # [M, nb]
                    for keep in KEEPS:
                        kk = int(round(keep * nb))
                        if mode == "shared":
                            order = np.argsort(-eb.sum(axis=0))
                            mask = np.zeros(nb, bool)
                            mask[order[:kk]] = True
                            xm = x * np.repeat(mask, B)[None, :]
                        else:
                            order = np.argsort(-eb, axis=1)
                            mask = np.zeros_like(eb, bool)
                            np.put_along_axis(mask, order[:, :kk], True, axis=1)
                            xm = x * np.repeat(mask, B, axis=1)
                        errs[keep].append(float(((xm @ W.T - y) ** 2).sum()) / ny)
                print(f"  blocks of {B:3d}, {mode:9s}: " + "  ".join(f"keep {k:.0%} {np.mean(v):.2e}" for k, v in errs.items()),
                      flush=True)


if __name__ == "__main__":
    main()
