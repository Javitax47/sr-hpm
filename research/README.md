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

## Tools

`tools/` holds the scripts and micro-benchmarks behind these numbers. They were written for the reference machine
(paths, model names); read them as executable lab notes rather than polished tools. The benchmarks that are meant to
be rerun anywhere are in [../bench/](../bench/).

`history/` keeps the original proposals and the audit that started the project (Spanish).
