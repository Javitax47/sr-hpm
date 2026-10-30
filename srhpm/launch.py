"""SR-HPM launcher: plans and starts llama-server (SR-HPM build) for any GGUF model on the local machine.

    python -m srhpm.launch MODEL.gguf [--draft DRAFT.gguf] [--ctx 8192] [--run]

What it decides (each choice was measured on Qwen3.8-27B, see research/notes/):
- model fits in VRAM          -> everything on the GPU (the CPU kernels do not apply);
- MoE that does not fit       -> experts on the CPU (-ot exps=CPU), the rest on the GPU, no speculation by default
                                 (each verified token activates other experts, so verification is not memory-flat);
- dense model that does not fit -> the SR-HPM path: small-batch CPU kernels (GGML_SRV, C4), the contiguous range of
                                 layers with the highest CPU cost per byte on the GPU (LLAMA_GPU_LAYERS), output head on
                                 the GPU when the drafter shares it (DFlash), recurrent snapshots written in place and
                                 replayed lazily (hybrid models), GPU keep-alive, dGPU co-verification over PCIe, and
                                 adaptive speculative drafting when a drafter is given.

Host-dependent choices come from srhpm/platform.py (large pages, VRAM reserve, VNNI, OS).
"""
import argparse
import os
import platform as _pf
import re
import shlex
import subprocess
import sys
from pathlib import Path

from srhpm import gguf_reader
from srhpm.placement import plan_layers
from srhpm.platform import Host, detect, disk_free_gib
from srhpm.requant import NEED_GIB

GIB = 2**30
EXE = ".exe" if _pf.system() == "Windows" else ""


def llama_dir() -> Path:
    """The SR-HPM llama.cpp checkout: $SRHPM_LLAMA_DIR, ./llama.cpp next to this repo, or ~/src/llama.cpp."""
    cands = [os.environ.get("SRHPM_LLAMA_DIR"), Path(__file__).resolve().parents[1] / "llama.cpp-srhpm",
             Path.home() / "src/llama.cpp"]
    for c in cands:
        if c and (Path(c) / "build/bin" / f"llama-server{EXE}").exists():
            return Path(c)
    return Path(cands[1])


def inspect(path: str) -> dict:
    r = gguf_reader(path)
    arch = r.fields["general.architecture"].contents()
    info = {"arch": arch, "bytes": 0, "layer_bytes": {}, "out_bytes": 0, "types": {}, "moe": False,
            "recurrent": False, "dflash": False}
    for t in r.tensors:
        nb = int(t.n_bytes)
        info["bytes"] += nb
        info["types"][t.tensor_type.name] = info["types"].get(t.tensor_type.name, 0) + nb
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            info["layer_bytes"][int(m.group(1))] = info["layer_bytes"].get(int(m.group(1)), 0) + nb
        if t.name == "output.weight":
            info["out_bytes"] = nb
        if "_exps" in t.name:
            info["moe"] = True
        if "ssm_" in t.name:
            info["recurrent"] = True
        if t.name.startswith("selector_") or t.name == "fc.weight":
            info["dflash"] = True
    return info


def plan(model: str, draft: str | None, ctx: int, host: Host, retrieval: bool = False,
         server: str = "", kv_type: str = "f16", kv_ram: bool = False) -> tuple[dict, list[str], str]:
    vram, ram = host.vram_free_gib, host.ram_gib
    mi = inspect(model)
    di = inspect(draft) if draft else None
    env, args = {}, ["-m", model, "-c", str(ctx), "-fa", "on", "--jinja", "-np", "1"]
    threads = host.cores
    args += ["-t", str(threads), "-tb", str(2 * threads)]
    weights = mi["bytes"] / GIB
    draft_gib = (di["bytes"] / GIB) if di else 0.0
    # KV cache + recurrent state + compute buffers on the GPU (measured: ~0.3 GiB at 4k context for Qwen3.8-27B
    # with 20 GPU layers and a DFlash drafter with --spec-draft-n-ubatch 16). With --kv-ram the KV cache stays in
    # system RAM, so long contexts do not take VRAM away from the model's layers
    kv_scale = {"f16": 1.0, "q8_0": 0.53, "q4_0": 0.28}.get(kv_type, 1.0)
    # (with --kv-ram, keep the margin of an 8k f16 cache: the compute buffers still grow with the batch)
    kv_gib = 0.15 + (0.015 * 8 if kv_ram else 0.015 * kv_scale * ctx / 1024)
    if kv_type != "f16":
        args += ["-ctk", kv_type, "-ctv", kv_type]
    if kv_ram:
        args += ["-nkvo"]
    if weights + draft_gib + kv_gib <= vram:
        args += ["-ngl", "99"]
        why = f"fits in VRAM ({weights:.1f} of {vram:.1f} GiB): everything on the GPU"
    elif mi["moe"]:
        args += ["-ngl", "99", "-ot", "exps=CPU"]
        why = f"MoE of {weights:.1f} GiB: experts on the CPU, the rest on the GPU"
        draft = None
    else:
        budget = vram - draft_gib - kv_gib
        out = mi["out_bytes"] / GIB
        use_out = bool(di and di["dflash"]) and out < budget * 0.5
        if use_out:
            budget -= out
        layers, used, saved, total = plan_layers(model, max(0.0, budget), max_runs=1)
        gpu = ",".join(map(str, layers)) + (",out" if use_out else "")
        env.update({"GGML_SRV": "1", "GGML_CUDA_SRHPM_TUNE": "1", "SRHPM_FA_GQA": "1"})
        # sign-free kernels for the types whose stored code is already w + offset (Q3_K, C4, Q6_K): fewer
        # instructions per token; results differ from the signed path only by float summation order
        env["GGML_SRV_OFF"] = "1"
        if any(t in mi["types"] for t in ("IQ3_S", "IQ3_XXS")):
            env["GGML_SRV_C4"] = "1"
        if mi["recurrent"]:
            # rollback snapshots straight into the cache, and only two full states per pass (the rest replayed from
            # rank-1 updates when needed: +4 % measured, same output)
            env["SRHPM_GDN_DIRECT"] = "1"
            env["SRHPM_GDN_LAZY"] = "1"
        if gpu:
            env["LLAMA_GPU_LAYERS"] = gpu
        if host.large_pages:
            env["GGML_LARGE_PAGES"] = "1"
        if host.vram_total_gib > 0:
            # the GPU idles through the CPU part of every pass and the driver drops its clock in between; a 1-warp
            # spin kernel keeps it in its high-performance state (+0-5 % measured on the laptop)
            env["GGML_CUDA_KEEPALIVE"] = "200"
        # the GPU also computes part of the rows of the CPU layers' products, reading them from RAM over PCIe, in the
        # passes of 6-8 tokens (with 4 the CPU alone already saturates the RAM): -2 to -4 % per pass measured
        if server and any(Path(server).with_name(n).exists() for n in ("srhpm_dgpu.dll", "libsrhpm_dgpu.so")):
            env["GGML_SRV_DGPU"] = "0,0,0,0,0,0.12,0.16,0.2"
        # -ub 1024: prompt processing streams the CPU weights to the GPU once per ubatch (+5-10 % prefill measured;
        # 2048 overflows a 6 GB VRAM)
        args += ["-ngl", str(len(layers) + (1 if use_out else 0)), "--no-host", "-lm", "none", "-ub", "1024", "-b", "2048"]
        why = (f"dense model of {weights:.1f} GiB > VRAM {vram:.1f} GiB: {len(layers)} layers ({used / GIB:.1f} GiB)"
               f"{' + output head' if use_out else ''} on the GPU, the rest on the CPU with the SR-HPM kernels; "
               f"estimated CPU time per pass {total:.2f} -> {total - saved:.2f} s")
        if weights + 2 > ram + vram:
            why += f"  WARNING: the model barely fits in RAM ({ram:.0f} GiB)"
    if draft:
        # n-gram retrievers are tried before the model drafter (llama.cpp priority order); drafts stay <= n_max
        # because the recurrent rollback keeps n_max snapshots
        extra = ",ngram-simple" if retrieval else ""
        if di["dflash"]:
            args += ["--spec-type", "draft-dflash" + extra, "-md", draft, "-ngld", "99", "--spec-draft-n-ubatch", "16"]
        else:
            args += ["--spec-type", "draft-simple" + extra, "-md", draft, "-ngld", "99"]
        if retrieval:
            args += ["--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "3", "--spec-ngram-simple-min-hits", "1"]
        if di["dflash"] and "GGML_SRV" in env:
            # verifying 4 tokens costs about as much as 1 on the CPU path: always keep 3 drafts, and extend up to 7
            # while the DFlash selector is confident (+7-10 % measured: tau 3.25 -> 4.2)
            env.update({"SRHPM_DRAFT_KEEP": "3", "SRHPM_DRAFT_PEXT": "0.9"})
            args += ["--spec-draft-n-max", "7", "--spec-draft-n-min", "1", "--draft-p-min", "0"]
        else:
            args += ["--spec-draft-n-max", "3", "--spec-draft-n-min", "3", "--draft-p-min", "0"]
    return env, args, why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--draft", default=None, help="drafter GGUF (DFlash/DFlash2, or a small model of the same family)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--server", default=None, help="llama-server of the SR-HPM build (default: found automatically)")
    ap.add_argument("--vram-gib", type=float, default=None, help="override the detected usable VRAM")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--retrieval", action="store_true", help="n-gram retrieval from the context before the drafter")
    ap.add_argument("--draft-sampling", choices=["greedy", "probabilistic"], default="greedy",
                    help="probabilistic: the DFlash2 drafter samples and the target verifies by rejection, which keeps "
                         "the sampled distribution exact and accepts more drafts at temperature > 0")
    ap.add_argument("--draft-max", type=int, default=None, help="override the maximum draft length")
    ap.add_argument("--kv-type", choices=["f16", "q8_0", "q4_0"], default="f16",
                    help="KV cache precision (q8_0 halves its memory for long contexts)")
    ap.add_argument("--kv-ram", action="store_true", help="keep the whole KV cache in system RAM (no VRAM for context)")
    ap.add_argument("--relax", type=float, default=0.0,
                    help="lossy: also accept a drafted token within this many nats of the target's best (1.0: +6 %% "
                         "tok/s for ~0.02 nats per token; the 3-4 bit quantization itself costs ~0.04)")
    ap.add_argument("--source", default=None, help="near-lossless weights (Q8_0/BF16 GGUF) for a re-quantization plan")
    ap.add_argument("--run", action="store_true", help="start the server instead of printing the command")
    a = ap.parse_args()
    server = a.server or str(llama_dir() / "build/bin" / f"llama-server{EXE}")
    host = detect()
    if a.vram_gib is not None:
        host.vram_free_gib = a.vram_gib
    env, args, why = plan(a.model, a.draft, a.ctx, host, a.retrieval, server, a.kv_type, a.kv_ram)
    if a.relax > 0 and a.draft:
        env["SRHPM_RELAX"] = str(a.relax)
    if a.draft and a.draft_sampling == "probabilistic":
        args += ["--spec-draft-sampling", "probabilistic"]
    if a.draft and a.draft_max is not None:
        i = args.index("--spec-draft-n-max")
        args[i + 1] = str(a.draft_max)
    args += ["--host", a.host, "--port", str(a.port)]
    print(f"# {why}", file=sys.stderr)
    print(f"# host: {host.os}, usable VRAM {host.vram_free_gib:.1f} GiB, RAM {host.ram_gib:.0f} GiB, {host.cores} cores, "
          f"{host.cpu_name}", file=sys.stderr)
    print(f"# large pages: {'yes' if host.large_pages else 'no'} - {host.large_pages_note}", file=sys.stderr)
    print(f"# CPU {host.vnni_note}", file=sys.stderr)
    for n in host.notes:
        print(f"# {n}", file=sys.stderr)
    if a.source:
        free = disk_free_gib(str(Path(a.model).parent))
        if free >= NEED_GIB:
            print(f"# {free:.0f} GiB free: you can re-quantize for this machine with python -m srhpm.requant "
                  f"--source {a.source} --template {a.model} --imatrix <imatrix.gguf> --recipe speed "
                  f"--first-cpu-layer <first CPU layer> --calib <your_text.txt> --out <dir> --run", file=sys.stderr)
        else:
            print(f"# re-quantization needs ~{NEED_GIB:.0f} GiB free, only {free:.0f} GiB", file=sys.stderr)
    if not Path(server).exists():
        print(f"# llama-server not found at {server}: build it with scripts/install_* or pass --server", file=sys.stderr)
    cmd = " ".join(f"{k}={v}" for k, v in env.items()) + " " + shlex.join([server] + args)
    print(cmd.strip())
    if a.run:
        sys.exit(subprocess.run([server] + args, env={**os.environ, **env}).returncode)


if __name__ == "__main__":
    main()
