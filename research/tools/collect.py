"""Generates many greedy completions against a running llama-server, to log draft lattices and verified tokens
(SRHPM_LATTICE_LOG / SRHPM_ACCEPT_LOG on the server) for offline drafter studies. Prompts: random slices of the local
corpus with a varied instruction. Prompt seeds start at --seed, so two runs with different seeds do not overlap.

usage: python -m research.tools.collect --n 150 --seed 0
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

TASKS = ["Continúa el texto.", "Resume en dos frases el texto anterior.", "Explica qué hace este fragmento.",
         "Continue the text.", "Summarize the previous text in two sentences.", "Escribe una pregunta sobre el texto y respóndela.",
         "Rewrite the previous passage more formally.", "Haz una lista de las ideas principales."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    text = (Path.home() / "f0_corpus_full.txt").read_text(encoding="utf-8", errors="ignore")
    rng = random.Random(a.seed)
    n_tok = 0
    for i in range(a.n):
        off = rng.randrange(0, max(1, len(text) - 2000))
        chunk = text[off:off + rng.randrange(300, 1500)]
        prompt = chunk + "\n\n" + rng.choice(TASKS)
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": 0.0,
                "top_k": 1, "seed": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.loads(r.read())
        n_tok += res["timings"]["predicted_n"]
        if (i + 1) % 10 == 0:
            print(f"[collect] {i + 1}/{a.n} prompts, {n_tok} tokens", flush=True)


if __name__ == "__main__":
    main()
