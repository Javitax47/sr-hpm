"""Cost-aware tensor placement for partially offloaded models (SR-HPM).

llama.cpp's -ngl offloads whole layers by index. VRAM is a byte budget, and what it buys is CPU time:
a tensor whose type is expensive to decode on the CPU (IQ2/IQ3 grids) saves more time per VRAM byte
than a cheap one (IQ4_XS, Q4_K). This script ranks the CPU-side matmul weights of a GGUF by
CPU seconds per byte (measured per type, see CPU_S_PER_GB) and fills the VRAM budget greedily,
then prints the matching `-ot` overrides.

Per-tensor placement splits layers between devices and the extra CPU<->GPU hops cost more than they
save (measured: 5.1 vs 5.7 tok/s). --mode layers (default) ranks whole layers instead and prints the
LLAMA_GPU_LAYERS list understood by the SR-HPM llama.cpp patch (the layer's KV cache and recurrent
state follow it, so no extra hops).

usage: python -m srhpm.placement <model.gguf> --budget-gib 3.0 [--mode layers|tensors] [--device CUDA0]
"""
import argparse
import re
import sys

from srhpm import gguf_reader

# CPU cost (s per GB of weights) for one verification pass of M = 4 tokens, measured with
# GGML_OP_PROF on a Ryzen 7 5800H (16 threads, SR-HPM kernels on, clang build), Qwen3.8-27B UD-Q3_K_XL.
CPU_S_PER_GB = {
    "IQ2_XS": 0.080, "IQ2_XXS": 0.077, "IQ3_XXS": 0.062, "IQ2_S": 0.061, "IQ3_S": 0.057,
    "Q3_K": 0.042, "Q6_K": 0.035, "Q4_K": 0.033, "IQ4_XS": 0.032, "Q5_K": 0.029, "Q2_K": 0.040,
    "IQ4_NL": 0.030, "Q4_0": 0.030, "Q8_0": 0.030,
}
# used only via GET_ROWS (one row per token): never worth VRAM
SKIP = re.compile(r"token_embd")


def plan(path: str, budget_gib: float):
    r = gguf_reader(path)
    cands = []
    for t in r.tensors:
        ty = t.tensor_type.name
        if ty in ("F32", "F16", "BF16") or SKIP.search(t.name) or ty not in CPU_S_PER_GB:
            continue
        cands.append((CPU_S_PER_GB[ty], t.name, ty, int(t.n_bytes)))
    cands.sort(key=lambda c: -c[0])
    budget, used, saved, chosen = budget_gib * 2**30, 0, 0.0, []
    for cost, name, ty, nb in cands:
        if used + nb > budget:
            continue
        chosen.append((name, ty, nb))
        used += nb
        saved += cost * nb / 1e9
    total = sum(c[0] * c[3] / 1e9 for c in cands)
    return chosen, used, saved, total


def plan_layers(path: str, budget_gib: float, max_runs: int = 1, hop_s: float = 0.005):
    r = gguf_reader(path)
    cost, nbytes = {}, {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        ty = t.tensor_type.name
        if not m:
            continue
        il = int(m.group(1))
        nbytes[il] = nbytes.get(il, 0) + int(t.n_bytes)
        cost[il] = cost.get(il, 0.0) + CPU_S_PER_GB.get(ty, 0.0) * int(t.n_bytes) / 1e9
    # Every CPU<->GPU boundary costs a synchronisation per pass (~5 ms measured with WDDM), so the GPU set
    # is limited to max_runs contiguous ranges of layers; each range adds two boundaries.
    L = max(cost) + 1
    pc = [0.0] * (L + 1)
    pb = [0] * (L + 1)
    for il in range(L):
        pc[il + 1] = pc[il] + cost.get(il, 0.0)
        pb[il + 1] = pb[il] + nbytes.get(il, 0)
    budget = budget_gib * 2**30
    wins = [(pc[b] - pc[a] - 2 * hop_s, pb[b] - pb[a], a, b) for a in range(L) for b in range(a + 1, L + 1)
            if pb[b] - pb[a] <= budget]
    wins.sort(key=lambda w: -w[0])
    best = (0.0, 0, [])
    for g, nb, a, b in wins[:1] if wins else []:
        best = (g, nb, [(a, b)])
    if max_runs >= 2:
        top = wins[:400]
        for i, (g1, n1, a1, b1) in enumerate(top):
            for g2, n2, a2, b2 in top[i + 1:]:
                if (b1 <= a2 or b2 <= a1) and n1 + n2 <= budget and g1 + g2 > best[0]:
                    best = (g1 + g2, n1 + n2, [(a1, b1), (a2, b2)])
    gain, used, runs = best
    chosen = sorted(il for a, b in runs for il in range(a, b))
    return chosen, used, sum(cost.get(il, 0.0) for il in chosen), sum(cost.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--budget-gib", type=float, default=3.0)
    ap.add_argument("--device", default="CUDA0")
    ap.add_argument("--mode", choices=["layers", "tensors"], default="layers")
    ap.add_argument("--max-runs", type=int, default=1, help="contiguous GPU layer ranges (layers mode)")
    ap.add_argument("--quiet", action="store_true", help="print only the -ot argument / layer list")
    a = ap.parse_args()
    if a.mode == "layers":
        layers, used, saved, total = plan_layers(a.model, a.budget_gib, a.max_runs)
        if not a.quiet:
            print(f"# {len(layers)} layers, {used / 2**30:.2f} GiB; estimated CPU time per M=4 pass: "
                  f"{total:.3f} s -> {total - saved:.3f} s", file=sys.stderr)
        print(",".join(map(str, layers)))
        return
    chosen, used, saved, total = plan(a.model, a.budget_gib)
    ot = ",".join(n.replace(".", "[.]") + "=" + a.device for n, _, _ in chosen)  # [.] survives shell quoting
    if not a.quiet:
        by = {}
        for _, ty, nb in chosen:
            by[ty] = by.get(ty, 0) + nb
        print(f"# {len(chosen)} tensors, {used / 2**30:.2f} GiB to {a.device}: "
              + ", ".join(f"{k} {v / 2**30:.2f}" for k, v in by.items()), file=sys.stderr)
        print(f"# estimated CPU time per M=4 pass: {total:.3f} s -> {total - saved:.3f} s", file=sys.stderr)
    print(ot)


if __name__ == "__main__":
    main()
