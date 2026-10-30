"""Sensitivity-guided type mix for the CPU layers, without re-quantizing per candidate.

For each CPU-layer tensor that can be IQ4_XS or Q3_K, the error of each option against Q8_0 is measured on the weights
themselves, weighted by the activation energy of each input column (imatrix in_sum2 / counts):
    E(tensor, type) = sum_ij imp_j * (W_type[i, j] - W_q8[i, j])^2     (output error energy of that product)
The options come from GGUFs already on disk (one with every candidate at IQ4_XS, one with every candidate at Q3_K).
The proxy is calibrated against measured KLDs of earlier mixes, then a greedy knapsack picks which tensors go to Q3_K:
least error for a byte budget, or fewest bytes for an error budget.

usage: python -m research.tools.sens_mix [--budget-gb 8.4] [--write types.txt]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

G = Path.home() / "f0_gguf"
FIRST_CPU = 20
BPW = {"IQ4_XS": 4.25, "Q3_K": 3.4375}
CACHE = Path.home() / "sens_cache.json"


def tensors(path):
    return {t.name: t for t in GGUFReader(str(path)).tensors}


def measure():
    q8, iq4, q3, s1, ud = (tensors(G / f) for f in ("Qwen3.8-27B-Q8_0.gguf", "q_cpu_iq4.gguf", "q_allq3k.gguf",
                                                     "q_speed1.gguf", "Qwen3.8-27B-UD-Q3_K_XL.gguf"))
    imx = tensors(G / "imatrix_unsloth.gguf")
    rows = []
    for name, t in s1.items():
        m = re.match(r"blk\.(\d+)\.", name)
        if not m or int(m.group(1)) < FIRST_CPU or t.tensor_type.name not in BPW:
            continue
        if iq4[name].tensor_type.name != "IQ4_XS" or q3[name].tensor_type.name != "Q3_K":
            continue
        w8 = dequantize(q8[name].data, q8[name].tensor_type).astype(np.float32)
        s2, cnt = imx.get(name + ".in_sum2"), imx.get(name + ".counts")
        imp = (np.asarray(s2.data, dtype=np.float64) / max(float(np.asarray(cnt.data)[0]), 1.0)) if s2 is not None else np.ones(w8.shape[1])
        err = {}
        for ty, src in (("IQ4_XS", iq4), ("Q3_K", q3)):
            d = dequantize(src[name].data, src[name].tensor_type).astype(np.float32) - w8
            err[ty] = float(((d.astype(np.float64) ** 2).sum(axis=0) * imp).sum())
        sig = float(((w8.astype(np.float64) ** 2).sum(axis=0) * imp).sum())
        rows.append({"name": name, "n": int(w8.size), "ud": ud[name].tensor_type.name, "s1": t.tensor_type.name,
                     "e_iq4": err["IQ4_XS"], "e_q3k": err["Q3_K"], "sig": sig})
        print(f"{name:40s} ud {rows[-1]['ud']:7s} iq4 {err['IQ4_XS']:.4g} q3k {err['Q3_K']:.4g} rel {err['IQ4_XS'] / sig:.2e}",
              flush=True)
    CACHE.write_text(json.dumps(rows))
    return rows


RES = (".ffn_down", ".ssm_out", ".attn_output")  # products that write straight into the residual stream
W_RES = 1.0


def werr(r, ty):
    w = W_RES if any(k in r["name"] for k in RES) else 1.0
    return w * (r["e_q3k"] if ty == "Q3_K" else r["e_iq4"])


def total(rows, choice):
    e = sum(werr(r, "Q3_K") if choice(r) else werr(r, "IQ4_XS") for r in rows)
    b = sum(r["n"] * BPW["Q3_K" if choice(r) else "IQ4_XS"] / 8 for r in rows)
    return e, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=None, help="CPU bytes of the candidate tensors")
    ap.add_argument("--write", default=None, help="write a llama-quantize type file for the chosen mix")
    ap.add_argument("--res-weight", type=float, default=None, help="weight of the residual-writing products (fitted if unset)")
    a = ap.parse_args()
    global W_RES
    rows = json.loads(CACHE.read_text()) if CACHE.exists() else measure()
    ffn = lambda r: ".ffn_" in r["name"]
    was3 = lambda r: r["ud"] in ("IQ3_S", "IQ3_XXS")
    known = {  # measured KLD vs Q8_0 (README_F6)
        "cpu_iq4": (lambda r: False, 0.0345),
        "speed1": (lambda r: was3(r) and ffn(r), 0.0430),
        "cpu_q3k": (lambda r: was3(r), 0.0483),
        "speed2": (lambda r: (was3(r) and ffn(r)) or (ffn(r) and (".ffn_gate" in r["name"] or ".ffn_up" in r["name"])), 0.0516),
    }
    s3 = Path.home() / "types_speed3_v1.txt"  # first knapsack mix, unweighted proxy: measured KLD 0.0569
    if s3.exists():
        q3 = {l.split("$=")[0][1:].replace(chr(92), "") for l in s3.read_text().splitlines() if l.endswith("=q3_k")}
        known["speed3v1"] = (lambda r, q3=q3: r["name"] in q3, 0.0569)
    if a.res_weight is None:  # least-squares fit of the residual weight over a grid
        best = None
        for w in np.geomspace(0.25, 64, 49):
            W_RES = w
            x = np.array([total(rows, ch)[0] for ch, _ in known.values()])
            y = np.array([k for _, k in known.values()])
            c = np.polyfit(x, y, 1)
            sse = float(((np.polyval(c, x) - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, w)
        W_RES = best[1]
        print(f"fitted residual weight {W_RES:.2f} (sse {best[0]:.2e})")
    else:
        W_RES = a.res_weight
    pts = []
    for k, (ch, kld) in known.items():
        e, b = total(rows, ch)
        pts.append((e, kld))
        print(f"{k:8s} proxy {e:.5g}  bytes {b / 1e9:.3f} GB  KLD {kld}")
    x, y = np.array(pts).T
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    print(f"fit KLD = {icpt:.4f} + {slope:.3e} * proxy; residuals {np.round(y - pred, 4)}")

    # greedy: from all-IQ4_XS, move to Q3_K the tensor with the least added error per byte saved
    order = sorted(rows, key=lambda r: (werr(r, "Q3_K") - werr(r, "IQ4_XS")) / (r["n"] * (BPW["IQ4_XS"] - BPW["Q3_K"]) / 8))
    e, b = total(rows, lambda r: False)
    chosen, curve = set(), []
    for r in order:
        chosen.add(r["name"])
        e += werr(r, "Q3_K") - werr(r, "IQ4_XS")
        b -= r["n"] * (BPW["IQ4_XS"] - BPW["Q3_K"]) / 8
        curve.append((b, e, set(chosen)))
    s1e, s1b = total(rows, known["speed1"][0])
    print(f"speed1: {s1b / 1e9:.3f} GB, predicted KLD {icpt + slope * s1e:.4f}")
    for target in (s1b, s1b * 0.97, s1b * 0.95, s1b * 0.93, s1b * 0.90):
        bb, ee, _ = min((c for c in curve if c[0] <= target), key=lambda c: -c[0])
        print(f"budget {target / 1e9:.3f} GB: greedy {bb / 1e9:.3f} GB, predicted KLD {icpt + slope * ee:.4f}")
    same = [c for c in curve if c[1] <= s1e]
    if same:
        bb, ee, _ = same[-1]
        print(f"same proxy error as speed1: {bb / 1e9:.3f} GB ({100 * (bb / s1b - 1):+.1f} %)")
    if a.budget_gb and a.write:
        bb, ee, ch = min((c for c in curve if c[0] <= a.budget_gb * 1e9), key=lambda c: -c[0])
        ref = GGUFReader(str(G / "q_speed1.gguf"))
        n = 0
        with open(a.write, "w", encoding="ascii") as f:
            for t in ref.tensors:
                ty = t.tensor_type.name
                if ty in ("F32", "F16", "BF16"):
                    continue
                if t.name in {r["name"] for r in rows}:
                    ty = "Q3_K" if t.name in ch else "IQ4_XS"
                    n += t.name in ch
                f.write("^" + re.escape(t.name) + "$=" + ty.lower() + "\n")
        print(f"wrote {a.write}: {n} tensors at Q3_K, {bb / 1e9:.3f} GB, predicted KLD {icpt + slope * ee:.4f}")


if __name__ == "__main__":
    main()
