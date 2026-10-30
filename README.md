# SR-HPM — fast decoding of dense LLMs that do not fit in your GPU

**SR-HPM** = **S**peculative verification on **R**AM-resident weights for **H**eterogeneous **P**C **M**emory.

SR-HPM is a patch for [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a small Python launcher that make
**dense models larger than the GPU's memory** decode several times faster on ordinary PCs. It splits the model
between GPU and CPU, verifies speculative drafts on the CPU with kernels built for 2–8 tokens, and attacks every
other cost we could measure in that pipeline.

Reference result — **Qwen3.8-27B** (12.4 GB at ~3.5 bpw) on a laptop with an **RTX 3060 Laptop (6 GB)**, a
**Ryzen 7 5800H** and **32 GB DDR4-3200**:

| Configuration | Decode | Prompt processing | Quality vs Q8_0 |
|---|---|---|---|
| stock llama.cpp, `-ngl 17`, DFlash2 drafter | 2.7 tok/s | 220 tok/s | KLD 0.043 |
| **SR-HPM** (exact: greedy output identical to non-speculative decoding) | **9.6–10.2 tok/s** | 132 tok/s | KLD 0.043 |
| SR-HPM + large pages | +0–10 % ([details](#large-pages)) | — | KLD 0.043 |
| SR-HPM + `--relax 1.0` (lossy, opt-in) | ~10.4 tok/s | — | + 0.018 nats/token |

Decode is measured on 10 short prompts, 200 greedy tokens each; prompt processing on a 3.6k-token prompt. SR-HPM
trades some prompt-processing speed (fewer layers' weights stream through the GPU per batch) for 3.6× faster decoding.

📄 **Paper:** [paper/srhpm.pdf](paper/srhpm.pdf) · 🌐 **Site:** https://javitax47.github.io/sr-hpm/

## Quick start

**Windows** (Visual Studio 2022 C++ tools, CMake, Git, Python; optional: CUDA 12.x, LLVM):

```powershell
git clone https://github.com/Javitax47/sr-hpm
cd sr-hpm
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
python -m srhpm.launch path\to\model.gguf --draft path\to\drafter.gguf --run
```

**Linux** (gcc ≥ 11 or clang ≥ 14, CMake, Git, Python; optional: CUDA 12.x):

```bash
git clone https://github.com/Javitax47/sr-hpm && cd sr-hpm
bash scripts/install_linux.sh
python3 -m srhpm.launch path/to/model.gguf --draft path/to/drafter.gguf --run
```

The installer clones llama.cpp at the pinned commit ([patch/UPSTREAM_COMMIT](patch/UPSTREAM_COMMIT)), applies
[patch/srhpm.patch](patch/srhpm.patch) and builds it. The launcher reads the GGUF and the machine (VRAM, RAM, cores,
large pages, OS) and starts `llama-server` with an OpenAI-compatible API on `http://127.0.0.1:8080`. Without `--run`
it only prints the command it would run.

<a id="large-pages"></a>**Large pages** (recommended): on Windows run `scripts\grant_large_pages.ps1` as administrator once, then
sign out and back in; on Linux enable transparent huge pages (`madvise`). The launcher turns them on when available.
On the reference laptop they gave +10 % in one A/B pair and a tie in the other (identical output).

## Using it from your app or pipeline

What SR-HPM runs is a standard `llama-server` with an **OpenAI-compatible API**, so anything that talks to OpenAI can
use it by pointing its base URL at the local server — LangChain, LlamaIndex, Open WebUI, Continue, n8n, or your own
code. The application does not need to know SR-HPM exists; chat, completions, streaming, JSON grammars and tool calls
come from llama-server unchanged, SR-HPM only changes how fast the tokens are computed.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Where it fits, and where it does not:

| | |
|---|---|
| ✅ Long generations from moderate prompts: assistants, code, writing, agents | the decode speedup applies to every generated token |
| ✅ A single user or a single pipeline worker | one request at a time |
| ⚠️ Several concurrent requests | they queue: the rollback machinery assumes one sequence (`-np 1`) |
| ⚠️ Huge prompts with very short answers (classification, RAG with a one-line answer) | prompt processing is 130–170 tok/s vs ~220 in stock llama.cpp on the reference laptop, so the gain shrinks or reverses |
| ⚠️ Conversations or documents beyond ~10k tokens | decode slows down as the context fills (see [context window](#context-window-sampling-and-thinking)) |
| ⚠️ Prose where the drafter is weak (e.g. Spanish summaries) | ~6 tok/s instead of ~10 on the same hardware |
| ➖ Models that fit in VRAM, MoE models, embeddings / reranking | the launcher falls back to the stock llama.cpp path: no gain, no loss |

Other llama.cpp-based programs built from the patched tree (`llama-cli`, language bindings) should pick up the CPU
kernels with the same environment variables, but only the server + launcher path is tested.

## Context window, sampling and thinking

Measured on the reference laptop with Qwen3.8-27B (`q_speed1`), 10 prompts each:

**Context window** (`--ctx`, default 8192). Qwen3.8-27B accepts up to 262,144 tokens; only its 16 attention layers keep
a KV cache (64 KB per token in f16).

| Window | Decode, short prompts | Notes |
|---|---|---|
| 8k (default) | 9.6–9.7 tok/s | |
| 32k, f16 cache on GPU (`--ctx 32768`) | 8.7 tok/s (−10 %) | |
| 128k, q8 cache in RAM (`--ctx 131072 --kv-type q8_0 --kv-ram`) | 8.8 tok/s (−10 %) | ~4.4 GB of RAM for the cache |
| 262k, q8 cache in RAM | 6.6 tok/s (−32 %) | the drafter's own cache then fills the VRAM |

A q8 cache costs nothing measurable in quality (KLD to Q8_0: 0.0428 with q8 vs 0.0430 with f16), so use
`--kv-type q8_0` whenever the window is large. **What the window costs is the text actually in it**: decode slows down as
the conversation or document grows, because every verified token attends to the whole context on the CPU.

With a quantized cache, llama.cpp's CPU attention handled each (token, query head) row on its own and dequantized the
whole cache for every row — 24× the traffic for 4 verified tokens and 6 query heads per KV head. SR-HPM's grouped
attention (`SRHPM_FA_GQA`) now covers quantized caches: each K row is read once per KV head and multiplied in int8
(the same arithmetic as the stock path, so the output is identical), and each V row is dequantized once:

| Tokens in context | Time to first token | Decode before | **Decode now** |
|---|---|---|---|
| 9.7k | 57 s | 3.4 tok/s | **5.2 tok/s** |
| 36k | 3.6 min | 1.5 tok/s | **4.1 tok/s** |
| 63k | 6.6 min | 0.7 tok/s | **≥ 2.5 tok/s** |
| 100k | 11 min | 0.5 tok/s | **2.2 tok/s** |
| 190k (262k window) | 29 min | 0.24 tok/s | not re-measured |

(long-context runs are summaries of Spanish text, where the drafter is weak; prompt processing stays at 110–170 tok/s;
the 63k figure is from an intermediate version that dequantized K.) In practice: windows up to 32–64k are comfortable;
beyond that, time to first token is what limits.

**Sampling.** Benchmarks are greedy. With temperature 1 (top-k 20, top-p 0.95, the web UI's values) decode drops ~9 %
(9.6 → 8.8 tok/s) because sampled tokens match the drafter less often. `--draft-sampling probabilistic` (the drafter
samples and the target verifies by rejection, which keeps the distribution exact) does not change that: 8.8–9.0 tok/s.

**Thinking.** With the model's reasoning mode on (the 💡 toggle in the web UI) and temperature 1: 7.5 tok/s, 14 %
fewer tokens per second than with it off at the same temperature, because reasoning text is harder to draft
(acceptance 0.53 vs 0.63), on top of the extra tokens it writes. `--thinking off` turns it off server-wide.

## Shorter answers: the biggest lever on the time you wait

What you wait for is tokens ÷ speed. Asking the model to be brief cuts the tokens far more than it costs in speed.
Same 10 prompts, complete answers (up to 2000 tokens), greedy, thinking off, two rounds:

| System prompt | Tokens per answer | Decode | **Seconds per answer** |
|---|---|---|---|
| none | 563 | 10.2–10.6 tok/s | 56–58 s |
| concise¹ | 286 (−49 %) | 9.9–10.6 tok/s | **30–32 s (−46 %)** |
| caveman² | 194 (−66 %) | 9.8–10.2 tok/s | **22–23 s (−61 %)** |

¹ `Answer concisely: no preamble, no filler, no repetition, no closing summary. Keep code, numbers and technical terms exact.`
² `Respond terse like smart caveman. Drop articles, filler, pleasantries and hedging. Fragments OK. Short synonyms. Technical terms and code exact.`

The answers stayed correct in this set (arithmetic, proofs, code with docstrings and tests); what goes is the preamble,
the step-by-step decoration and the closing summary. Telegraphic text is slightly harder to draft (acceptance 0.71 vs
0.76), which costs 2–4 % of speed. Put one of these lines in your app's system prompt; it works with any client.

## Which models

Any **dense** GGUF model that llama.cpp supports. The launcher picks the path:

| Situation | What the launcher does |
|---|---|
| model fits in VRAM | everything on the GPU (SR-HPM adds nothing) |
| MoE that does not fit | experts on the CPU, the rest on the GPU (stock llama.cpp path) |
| **dense model that does not fit** | **the SR-HPM path** |

Speculative decoding needs a **drafter** for the same tokenizer: a DFlash/DFlash2 drafter for the model (best), or a
small model of the same family (`--draft`). Without a drafter SR-HPM still places layers and uses its kernels, but the
large gains come from verifying several drafted tokens per pass.

Quantization types read by the fast CPU kernels: IQ4_XS, IQ3_S, IQ3_XXS (losslessly re-encoded to 4-bit codes),
Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS/XS/S. Other types fall back to stock llama.cpp kernels.
`python -m srhpm.requant` re-quantizes a model for this layout (CPU layers read as few bytes as possible at the same
quality) when you have a near-lossless source (Q8_0/BF16) and ~16 GB free.

## How it works (short version)

Decoding a dense model that does not fit in VRAM is bound by reading the CPU-side weights from RAM once per token.
Speculative decoding verifies several drafted tokens in one pass, so the same bytes serve several tokens — but only
if verifying *M* tokens costs about as much as verifying one. SR-HPM makes that true and removes what else stood in
the way:

1. **Small-batch CPU kernels (`GGML_SRV`).** Each weight block is decoded once and applied to all *M* tokens
   (vpmaddubsw/vpmaddwd), with sign-free paths for offset-coded types and a lossless 4-bit re-encoding (C4) of IQ3.
   Verifying 4 tokens costs ~1.1–1.2× one token instead of 2–3× in stock llama.cpp.
2. **Cost-aware layer placement.** The contiguous range of layers with the highest CPU cost per byte goes to the GPU,
   plus the output head when the drafter shares it.
3. **Adaptive drafting.** Keep 3 drafted tokens, extend up to 7 while the drafter is confident.
4. **Lazy rollback snapshots for hybrid (recurrent) models.** Two full states per pass plus exact rank-1 replay
   instead of one 64 KB state per head and token (+4 %).
5. **dGPU co-verification over PCIe.** The GPU is idle while the CPU layers run; it computes a share of their rows
   reading a pinned copy straight from RAM (zero copy). Only from 6 tokens per pass, where the CPU is compute-bound.
6. **GPU keep-alive, large pages, split prompt processing** (the recurrent op runs on the GPU in chunks while the
   CPU replays only the first token), and kernel-aware re-quantization.

The full story — including the ~25 ideas that did **not** work and why — is in the [paper](paper/srhpm.pdf) and in
[research/](research/).

## Repository layout

| Path | Contents |
|---|---|
| [patch/](patch/) | `srhpm.patch` for llama.cpp and the upstream commit it applies to |
| [srhpm/](srhpm/) | launcher (`launch.py`), host detection, layer placement, re-quantization |
| [scripts/](scripts/) | installers for Windows / Linux, large-page helper |
| [dgpu/](dgpu/) | CUDA kernels of the dGPU co-verifier (`srhpm_dgpu.dll`) and their tests |
| [bench/](bench/) | decode / context / prefill benchmarks used for every number in the paper |
| [research/](research/) | experiment scripts and lab notes (Spanish), including refuted hypotheses |
| [paper/](paper/) | the paper (LaTeX + PDF) |
| [docs/](docs/) | the project site (GitHub Pages) |

## Environment variables (set by the launcher)

| Variable | Effect |
|---|---|
| `GGML_SRV=1` | small-batch CPU kernels |
| `GGML_SRV_C4=1` | lossless 4-bit re-encoding of IQ3 weights |
| `GGML_SRV_OFF=1` | sign-free kernels for offset-coded types |
| `GGML_SRV_DGPU=f1,...,f8` | share of rows the dGPU verifies for 1..8 tokens (Windows + CUDA) |
| `LLAMA_GPU_LAYERS=a,b,...,out` | exact set of layers (and output head) on the GPU |
| `GGML_CUDA_SRHPM_TUNE=1` | op offload only for large batches; per-type MMVQ/MMQ switch |
| `GGML_CUDA_KEEPALIVE=200` | 1-warp spin kernel that keeps the GPU clocked up |
| `GGML_LARGE_PAGES=1` | CPU weights on 2 MB pages |
| `SRHPM_FA_GQA=1` | grouped small-batch attention on the CPU |
| `SRHPM_GDN_DIRECT=1`, `SRHPM_GDN_LAZY=1` | in-place / lazy recurrent snapshots (hybrid models) |
| `SRHPM_DRAFT_KEEP=3`, `SRHPM_DRAFT_PEXT=0.9` | adaptive draft length |
| `SRHPM_RELAX=δ` | lossy relaxed acceptance (opt-in, `--relax`) |

Launcher options beyond the model and drafter: `--ctx`, `--kv-type {f16,q8_0,q4_0}`, `--kv-ram`, `--thinking {auto,on,off}`,
`--draft-sampling {greedy,probabilistic}`, `--draft-max`, `--relax`, `--retrieval`, `--vram-gib`, `--host`, `--port`.

Instrumentation for research: `GGML_SRV_PROF`, `GGML_OP_PROF`, `SRHPM_SPEC_PROF`, `SRHPM_PASS_LOG`,
`SRHPM_LATTICE_LOG`, `SRHPM_ACCEPT_LOG`, `GGML_SRV_DUMP_NAMES`.

## Limits

- x86-64 with AVX2 for the fast CPU kernels; ARM falls back to stock llama.cpp.
- One sequence at a time (the rollback machinery assumes `-np 1`).
- The dGPU co-verifier is Windows + CUDA only; on Linux it is simply not used.
- Gains depend on the drafter: on prose where the drafter is weak (e.g. Spanish summaries) decode drops to ~6 tok/s
  with the same hardware.

## About the name

The project started from a proposal called *Speculative Residual · Hierarchical Precision Memory* (kept in
[research/history/](research/history/)). Its audit refuted several of its pieces (a "holographic" context memory among
them); what survived and became the tool is speculative verification on the CPU over weights kept in RAM, on a
GPU + CPU memory hierarchy. The acronym was kept and re-read accordingly.

## License

MIT (see [LICENSE](LICENSE)). The patch modifies llama.cpp (MIT); the vendored OpenCL headers are Apache-2.0
(Khronos).
