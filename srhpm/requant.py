"""Kernel-aware re-quantization of a dense model for this machine (needs a near-lossless source and disk space).

The SR-HPM CPU kernels read IQ3_S / IQ3_XXS through a lossless 4-bit re-encoding (C4, 4.375 bpw), more bytes than
IQ4_XS (4.25 bpw), while Q3_K is read natively at 3.44 bpw with a fast sign-free kernel. So in the layers that run on
the CPU, IQ3 is the worst of both: the bytes of IQ4 and the quality of IQ3. The mixes here start from a published
dynamic quant (the template, e.g. Unsloth's UD-Q3_K_XL) and rewrite only those tensors:

  quality   CPU layers: IQ3 -> IQ4_XS.  Qwen3.8-27B: KLD 0.0427 -> 0.0345 vs Q8_0, same speed (cpu_iq4)
  speed     CPU layers: IQ3 in the FFN -> Q3_K, IQ3 in attention / recurrent projections -> IQ4_XS.
            Qwen3.8-27B: KLD 0.0430 (same as the template), CPU bytes -6.6 %, +5.7 % tok/s (speed1)

Pipeline (llama.cpp tools):
  0. (optional) convert_hf_to_gguf.py  safetensors -> BF16 GGUF, when the source is not a GGUF already
  1. llama-quantize --allow-requantize --imatrix  with the per-tensor type file of the recipe
  2. llama-perplexity --kl-divergence-base on your text with the source (once), then --kl-divergence with the result
The disk check and the command plan always run; execution needs --run.

    python -m srhpm.requant --source Qwen3.8-27B-Q8_0.gguf --template Qwen3.8-27B-UD-Q3_K_XL.gguf \\
        --imatrix imatrix_unsloth.gguf --recipe speed --first-cpu-layer 20 --calib my_texts.txt --out out/ [--run]
"""
import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from srhpm.platform import disk_free_gib

GIB = 1 << 30
# free space for a re-quantization from a GGUF source: the output (~13 GB for a 27B model) and the KLD base (~2 GB)
NEED_GIB = 16.0


def mix(template_types: dict, recipe: str, first_cpu: int) -> dict:
    """tensor name -> type for the recipe, from the template's types (names as in the GGUF)."""
    out = {}
    for name, ty in template_types.items():
        m = re.match(r"blk\.(\d+)\.", name)
        cpu = bool(m) and int(m.group(1)) >= first_cpu
        if cpu and ty in ("IQ3_S", "IQ3_XXS"):
            if recipe == "quality":
                ty = "IQ4_XS"
            elif recipe == "speed":
                ty = "Q3_K" if ".ffn_" in name else "IQ4_XS"
        out[name] = ty
    return out


def write_type_file(types: dict, path: Path) -> int:
    n = 0
    with open(path, "w", encoding="ascii") as f:
        for name, ty in types.items():
            if ty in ("F32", "F16", "BF16"):
                continue
            f.write("^" + re.escape(name) + "$=" + ty.lower() + "\n")
            n += 1
    return n


def template_types(path: str) -> dict:
    from srhpm import gguf_reader
    return {t.name: t.tensor_type.name for t in gguf_reader(path).tensors}


def plan(a, type_file: Path) -> list[list[str]]:
    o = Path(a.out)
    bin_ = Path(a.llama_dir) / "build/bin"
    src = a.source
    steps = []
    if not src.endswith(".gguf"):
        bf16 = o / "model-bf16.gguf"
        steps.append([sys.executable, str(Path(a.llama_dir) / "convert_hf_to_gguf.py"), src, "--outtype", "bf16", "--outfile", str(bf16)])
        src = str(bf16)
    q = o / f"model-srhpm-{a.recipe}.gguf"
    quant = [str(bin_ / "llama-quantize"), "--allow-requantize", "--tensor-type-file", str(type_file)]
    if a.imatrix:
        quant += ["--imatrix", a.imatrix]
    steps.append(quant + [src, str(q), "IQ4_XS"])
    base = o / "kld-base.bin"
    steps.append([str(bin_ / "llama-perplexity"), "-m", src, "-f", a.calib, "-c", "512", "--chunks", "16", "--kl-divergence-base", str(base)])
    steps.append([str(bin_ / "llama-perplexity"), "-m", str(q), "-f", a.calib, "-c", "512", "--chunks", "16",
                  "--kl-divergence-base", str(base), "--kl-divergence"])
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="near-lossless source: a Q8_0 / BF16 GGUF, or a safetensors directory")
    ap.add_argument("--template", required=True, help="published dynamic quant whose per-tensor types are the starting point")
    ap.add_argument("--imatrix", default=None, help="importance matrix (the template's, or one from your own text)")
    ap.add_argument("--recipe", choices=["quality", "speed"], default="speed")
    ap.add_argument("--first-cpu-layer", type=int, required=True, help="first layer that runs on the CPU (from srhpm.launch)")
    ap.add_argument("--calib", required=True, help="text representative of your use, for the KLD check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-dir", default=None, help="SR-HPM llama.cpp checkout (default: found like srhpm.launch)")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.llama_dir is None:
        from srhpm.launch import llama_dir
        a.llama_dir = str(llama_dir())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tt = template_types(a.template)
    types = mix(tt, a.recipe, a.first_cpu_layer)
    changed = sum(1 for k in tt if tt[k] != types[k])
    type_file = out / f"types-{a.recipe}.txt"
    n = write_type_file(types, type_file)
    print(f"# {a.recipe}: {changed} of {n} quantized tensors change type; type file {type_file}", file=sys.stderr)

    src_gib = Path(a.source).stat().st_size / GIB if Path(a.source).is_file() else 60.0
    need = (0 if a.source.endswith(".gguf") else src_gib * 2) + Path(a.template).stat().st_size / GIB * 1.1 + 2
    free = disk_free_gib(str(out))
    print(f"# free space at the destination: {free:.0f} GiB (needs ~{need:.0f} GiB)", file=sys.stderr)
    steps = plan(a, type_file)
    for s in steps:
        print(shlex.join(s))
    if a.run:
        if free < need:
            sys.exit(f"not enough free space ({free:.0f} < {need:.0f} GiB): not running")
        for s in steps:
            print("+", shlex.join(s), file=sys.stderr)
            subprocess.run(s, check=True)


if __name__ == "__main__":
    main()
