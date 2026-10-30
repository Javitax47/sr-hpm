"""Refutation test: can a low-rank correction held in VRAM (computed by the idle GPU) recover what 2-bit CPU weights lose?

For real CPU-layer matrices: residual R = W_q8 - W_lowbit, weighted by sqrt(activation energy) per input column
(imatrix), so that ||R_w||^2 is the output error energy. The best rank-r correction removes the top r eigenvalues of
R_w^T R_w (Eckart-Young). Compared with the error energy of the types used today (IQ4_XS from q_cpu_iq4, Q3_K from
q_speed1 / q_cpu_q3k), and the VRAM a rank-r fp16 correction costs, (rows + cols) * r * 2 bytes.

usage: python -m research.tools.lowrank_test
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

G = Path.home() / "f0_gguf"
RANKS = (16, 32, 64, 128, 256)


def tensors(p):
    return {t.name: t for t in GGUFReader(str(p)).tensors}


def deq(t):
    return dequantize(t.data, t.tensor_type).astype(np.float32)


def main():
    q8, lo, iq4, s1 = (tensors(G / f) for f in ("Qwen3.8-27B-Q8_0.gguf", "Qwen3.8-27B-UD-IQ1_S.gguf", "q_cpu_iq4.gguf",
                                               "q_speed1.gguf"))
    imx = tensors(G / "imatrix_unsloth.gguf")
    picks = []
    for layer in (30, 45, 58):
        for kind in ("ffn_up", "ffn_gate", "ffn_down", "attn_qkv", "ssm_out"):
            n = f"blk.{layer}.{kind}.weight"
            if n in lo and n in q8 and n + ".in_sum2" in imx:
                picks.append(n)
    print(f"{'tensor':28s} {'low':8s} {'bpw':>5s} {'err_low':>9s} " + " ".join(f"+r{r:<4d}" for r in RANKS)
          + f" {'IQ4_XS':>8s} {'speed1':>8s}   (error energy relative to the weighted signal; VRAM MB at r=64)")
    for n in picks:
        w8 = deq(q8[n])
        imp = np.asarray(imx[n + ".in_sum2"].data, dtype=np.float64) / max(float(np.asarray(imx[n + ".counts"].data)[0]), 1.0)
        sw = np.sqrt(imp).astype(np.float32)
        sig = float(((w8 * sw) ** 2).sum())
        r_w = (w8 - deq(lo[n])) * sw  # [rows, cols]
        e_low = float((r_w.astype(np.float64) ** 2).sum())
        # eigenvalues of the smaller Gram matrix = squared singular values
        g = (r_w.T @ r_w) if r_w.shape[1] <= r_w.shape[0] else (r_w @ r_w.T)
        ev = np.sort(np.linalg.eigvalsh(g.astype(np.float64)))[::-1]
        rem = [max(e_low - float(ev[:r].sum()), 0.0) / sig for r in RANKS]
        e4 = float((((w8 - deq(iq4[n])) * sw).astype(np.float64) ** 2).sum()) / sig
        e1 = float((((w8 - deq(s1[n])) * sw).astype(np.float64) ** 2).sum()) / sig
        bpw = lo[n].n_bytes * 8 / w8.size
        vram = (w8.shape[0] + w8.shape[1]) * 64 * 2 / 1e6
        print(f"{n:28s} {lo[n].tensor_type.name:8s} {bpw:5.2f} {e_low / sig:9.2e} " + " ".join(f"{x:.2e}" for x in rem)
              + f" {e4:8.2e} {e1:8.2e} [{s1[n].tensor_type.name}] {vram:5.1f} MB", flush=True)


if __name__ == "__main__":
    main()
