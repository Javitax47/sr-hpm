"""Per-tensor quantization mixes for llama-quantize --tensor-type-file, starting from an existing GGUF's types.

The reference mix is Unsloth's UD-Q3_K_XL. Rules rewrite the type of some tensors; the result is one anchored, exact
pattern per tensor (`^blk\\.20\\.ffn_up\\.weight$=iq4_xs`), so no pattern can hit another tensor.

usage: python -m research.tools.mix --ref <model.gguf> --rule cpu_iq4 --first-cpu-layer 20 --out types.txt
rules:
  repro     the reference types unchanged (to measure what re-quantizing from Q8_0 costs by itself)
  cpu_q3k   in the CPU layers, IQ3_S / IQ3_XXS -> Q3_K (read natively at 3.44 bpw instead of C4 at 4.375)
  cpu_iq4   in the CPU layers, IQ3_S / IQ3_XXS -> IQ4_XS: on the CPU path IQ3 is read through the C4 re-encoding
            (4.375 bpw), more bytes than IQ4_XS (4.25 bpw), so this is more quality for no more memory traffic
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

BPW = {"IQ4_XS": 4.25, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625,
       "Q8_0": 8.5, "IQ2_S": 2.5625, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "F32": 32, "F16": 16, "BF16": 16}
# bytes the SR-HPM CPU kernels read per weight (IQ3 types go through C4)
CPU_BPW = dict(BPW, IQ3_S=4.375, IQ3_XXS=4.375)


def ref_types(path):
    r = GGUFReader(path)
    return {t.name: (t.tensor_type.name, int(t.n_elements)) for t in r.tensors}


def apply(types, rule, first_cpu, lite_layers=(), lite_map=None, head=None):
    out = {}
    for name, (ty, n) in types.items():
        m = re.match(r"blk\.(\d+)\.", name)
        layer = int(m.group(1)) if m else -1
        cpu = layer >= first_cpu
        if "cpu_iq4" in rule and cpu and ty in ("IQ3_S", "IQ3_XXS"):
            ty = "IQ4_XS"
        # cpu_q3k: in the CPU layers, IQ3 (read as C4, 4.375 bpw) -> Q3_K read natively at 3.44 bpw (-21 % bytes)
        if "cpu_q3k" in rule and cpu and ty in ("IQ3_S", "IQ3_XXS"):
            ty = "Q3_K"
        # speed1: IQ3 in the FFN -> Q3_K (bytes), IQ3 in attention / recurrent projections -> IQ4_XS (quality)
        # speed2: speed1, plus ffn_gate / ffn_up IQ4_XS -> Q3_K
        if ("speed1" in rule or "speed2" in rule) and cpu:
            ffn = ".ffn_" in name
            if ty in ("IQ3_S", "IQ3_XXS"):
                ty = "Q3_K" if ffn else "IQ4_XS"
            elif "speed2" in rule and ty == "IQ4_XS" and (".ffn_gate" in name or ".ffn_up" in name):
                ty = "Q3_K"
        # gpu_lite: the layers meant for VRAM get fewer bits, so more of them fit (no C4 inflation on the GPU)
        if "gpu_lite" in rule and layer in lite_layers and lite_map and ty in lite_map:
            ty = lite_map[ty]
        # gq3: in the GPU layers, IQ3_S -> Q3_K (same size; IQ3_XXS would grow 12 % and cost VRAM). CUDA multiplies Q3_K 26-34 % faster than IQ3_S with 4-8
        # columns (tensor-core MMQ instead of grid decoding), at the same size as IQ3_S
        if "gq3" in rule and 0 <= layer < first_cpu and ty == "IQ3_S":
            ty = "Q3_K"
        if head and name == "output.weight":
            ty = head
        out[name] = (ty, n)
    return out


def report(label, types, first_cpu):
    tot = cpu_b = 0.0
    for name, (ty, n) in types.items():
        m = re.match(r"blk\.(\d+)\.", name)
        tot += n * BPW.get(ty, 16) / 8
        if m and int(m.group(1)) >= first_cpu:
            cpu_b += n * CPU_BPW.get(ty, 16) / 8
    c = Counter(ty for ty, _ in types.values())
    print(f"{label}: file {tot / 1e9:.2f} GB, CPU-side bytes read per pass {cpu_b / 1e9:.2f} GB, types {dict(c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--rule", default="repro")
    ap.add_argument("--first-cpu-layer", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lite-layers", default="", help="range a-b of layers to lighten (rule gpu_lite)")
    ap.add_argument("--lite-map", default="IQ4_XS:IQ3_XXS,IQ3_S:IQ3_XXS,Q4_K:Q3_K,Q5_K:Q4_K",
                    help="type replacements in the lightened layers")
    ap.add_argument("--head", default=None, help="type of output.weight")
    a = ap.parse_args()
    ref = ref_types(a.ref)
    lite = range(int(a.lite_layers.split("-")[0]), int(a.lite_layers.split("-")[1]) + 1) if a.lite_layers else ()
    lmap = dict(kv.split(":") for kv in a.lite_map.split(","))
    mix = apply(ref, a.rule, a.first_cpu_layer, lite, lmap, a.head)
    report("reference", ref, a.first_cpu_layer)
    report(a.rule, mix, a.first_cpu_layer)
    with open(a.out, "w", encoding="ascii") as f:
        for name, (ty, _) in mix.items():
            if ty in ("F32", "F16", "BF16"):
                continue
            f.write("^" + re.escape(name) + "$=" + ty.lower() + "\n")


if __name__ == "__main__":
    main()
