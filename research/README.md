# Research log

Everything SR-HPM ships was kept only after an A/B measurement on the reference laptop (RTX 3060 Laptop 6 GB,
Ryzen 7 5800H, 32 GB DDR4-3200, Windows 11, Qwen3.8-27B + DFlash2). Every idea was first attacked with the cheapest
experiment that could refute it. The lab notes are in Spanish ([notes/](notes/)); this page indexes them.

## Kept (measured gains)

| Idea | Gain | Where |
|---|---|---|
| Small-batch CPU verification kernels (decode each block once for all tokens) | 2.70 → 5.72 tok/s | notes/README_F5.md |
| Cost-aware GPU layer range + output head on the GPU | → 6.77 | notes/README_F5.md |
| Op offload only for large batches; per-type MMVQ/MMQ switch | → 7.23 | notes/README_F5.md |
| C4: lossless 4-bit re-encoding of IQ3 (one `pshufb` instead of grid lookups) | → 8.13 | notes/README_F5.md |
| Recurrent snapshots written in place | → 8.62 | notes/README_F5.md |
| Q4_K/Q5_K min-term fix (5× slower with ≥4 tokens before) | +4–6 % | notes/README_F6.md |
| Adaptive drafting (keep 3, extend to 7 while confident) | +7–10 % | notes/README_F6.md |
| IQ4_XS paired layout; vectorized scale unpacking | +5–7 % | notes/README_F6.md |
| GPU keep-alive kernel | +0–5 % | notes/README_F6.md |
| Lazy GDN snapshots (2 full states + rank-1 replay) | +4 % | notes/README_F6.md |
| Kernel-aware re-quantization (`speed1`: same KLD, −6.6 % CPU bytes) | +5.7 % | notes/README_F6.md, srhpm/requant.py |
| Split prompt processing for rollback snapshots (conv tail + GPU-chunked GDN) | prefill +35 % | notes/README_F6.md |
| dGPU co-verification over PCIe for 6–8-token passes | −2–4 % per pass | notes/README_F6.md, dgpu/ |
| Large pages for the CPU weights | see the paper | notes/README_F6.md |
| Relaxed acceptance (opt-in, lossy) | +6 % at δ = 1 | notes/README_F6.md |

## Refuted (each with its test)

| Hypothesis | Test | Why it fails |
|---|---|---|
| Lossless entropy coding of weights | tools/entropy.py | IQ4_XS codes carry 3.91 of 4 bits: ≤3 % |
| Delta verification (W·(x_t − x_0)) | tools/delta_stats.py | deltas are 50–95 % of the norm |
| Helper SMT threads prefetching weights | tools/srv_bench.cpp `helper` | the kernel is compute-bound with 4+ tokens |
| Thermal headroom | tools/thermo.ps1 | CPU stays at 106–108 % of nominal |
| iGPU as a second verifier (OpenCL) | tools/igpu_*.cpp | 38 GB/s alone, 5.5× slower under CPU load (APU power budget) |
| MoE-ification of the dense model | analysis | union of experts over 4–8 tokens ≈ whole FFN |
| Draft trees / two-chain verification | tools/tree_sim.py | ≤ +3 % with the real cost curve; chains on a recurrent model duplicate the prefix |
| Expected-acceptance path over the lattice | tools/tree_sim.py `dp chain` | +0.3 % τ |
| Online reranker of the drafter's candidates | tools/rerank_test.py | τ 4.098 vs 4.105, although a perfect reranker would give τ 6.33 |
| Logit lens of the GPU half-model as reranker | tools/lens_test.py | layer 20 picks the right candidate 10.9 % of the time |
| 2-bit CPU layers | quantize + KLD + A/B | KLD 0.094 and slower (grid decode is compute-bound) |
| Low-rank GPU correction of 2-bit error | tools/lowrank_test.py | the residual is nearly white: rank 256 removes 10–20 % |
| Sensitivity-guided mix (imatrix-weighted error knapsack) | tools/sens_mix.py | predicted KLD 0.042, measured 0.057; with a fitted class weight only −1.8 % bytes |
| Activation sparsity with a pass-shared mask (TEAL-style) | tools/sparsity_test.py | at block granularity, skipping 10 % already costs 5–10× the quantization error |
| Layer dropping (ShortGPT) | layer distance + KLD | dropping 2 layers: KLD 0.043 → 0.108 |
| Exact output head over a vocabulary subset | tools/head_prune_test.py | the bound adds 60–95 logits: 97–100 % of the head must be read |
| Small ops on one thread without barriers | A/B + op profile | identical speed; the cost was cache misses, not barriers |
| N-gram / context retrieval drafting | A/B on chat and long summaries | never triggers (the model paraphrases) |
| More GPU layers in the spare VRAM | A/B | below ~0.4 GB free WDDM slows everything down |
| Probabilistic drafting (rejection verification) at temperature 1 | A/B, 10 prompts | 8.8–9.0 vs 8.8 tok/s with greedy drafting |

## Context window, sampling, KV cache (30 oct. 2026)

Unattended queue `queue_ctx_sampling.sh` (lab), 20 runs:

| Measurement | Result |
|---|---|
| Temperature 1 (top-k 20, top-p 0.95) vs greedy | 8.8 vs 9.6 tok/s (−9 %): sampled tokens match the drafter less often |
| Thinking on, temperature 1 | 7.5 tok/s |
| Window 32k f16 / 128k q8 in RAM / 262k q8 in RAM, short prompts | 8.7 / 8.8 / 6.6 tok/s (8k: 9.7); at 262k the drafter's cache fills the VRAM |
| KV cache q8_0 vs f16, KLD to Q8_0 | 0.0428 vs 0.0430: no measurable loss |
| Decode with 9.7k / 36k / 63k / 100k / 190k tokens in context | 3.4 / 1.5 / 0.7 / 0.5 / 0.24 tok/s; prompt processing 110–170 tok/s: CPU attention over the cache becomes the bottleneck |

**Follow-up — grouped attention for quantized caches (kept).** llama.cpp's CPU attention for a q8 cache handled each
(verified token, query head) row on its own and dequantized the whole cache for each: 24× the traffic. The grouped
small-batch path (`SRHPM_FA_GQA`) was f16/f32-only; it now covers quantized caches, first dequantizing each K/V row once
per KV head, then multiplying K in int8 against query rows quantized once per tile (the stock arithmetic: the 36k output
hash equals the stock run's). `test-backend-ops -o FLASH_ATTN_EXT -b CPU`: 5379/5379 against the reference.

| Tokens in context | Stock | Grouped, K dequantized | Grouped, int8 Q·K |
|---|---|---|---|
| 9.7k | 3.4 | 4.9 | 5.2 tok/s |
| 36k | 1.5 | 4.1 | 4.1 tok/s |
| 63k | 0.71 | 2.5 | — |
| 100k | 0.50 | 1.9 | 2.2 tok/s |

## Tools

`tools/` holds the scripts and micro-benchmarks behind these numbers. They were written for the reference machine
(paths, model names); read them as executable lab notes rather than polished tools. The benchmarks that are meant to
be rerun anywhere are in [../bench/](../bench/).

`history/` keeps the original proposals and the audit that started the project (Spanish).

## Shorter answers (30 oct. 2026)

`bench/dec_bench.py --system ... --max-tokens 2000`, 10 prompts, greedy, thinking off, two rounds:

| System prompt | Tokens per answer | Decode | Seconds per answer | Acceptance |
|---|---|---|---|---|
| none | 563 | 10.2–10.6 tok/s | 56–58 | 0.76 |
| concise | 286 | 9.9–10.6 tok/s | 30–32 | 0.76 |
| caveman | 194 | 9.8–10.2 tok/s | 22–23 | 0.71 |

Answers checked by hand on arithmetic, the irrationality proof and the sieve code: all correct.

