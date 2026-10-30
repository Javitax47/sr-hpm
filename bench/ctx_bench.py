"""Prefill (time to first token) and decode speed versus context length on a running llama-server.

usage: python -m bench.ctx_bench --text book.txt --port 8080 --lengths 500,2000,8000 [--out ctx.json]
The prompt is a slice of a text file (--text) followed by a summary request.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--text", required=True, help="any long text file (the prompt is its beginning)")
    ap.add_argument("--lengths", default="500,2000,8000")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    text = Path(a.text).read_text(encoding="utf-8", errors="ignore")
    rows = []
    for n in map(int, a.lengths.split(",")):
        # ~3.5 characters per token for mixed Spanish/English/code text
        chunk = text[: int(n * 3.5)]
        prompt = chunk + "\n\nResume en tres frases el texto anterior."
        t0 = time.time()
        r = post(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
            "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        wall = time.time() - t0
        t = r["timings"]
        row = {"target_tokens": n, "prompt_n": t["prompt_n"], "prompt_s": t["prompt_ms"] / 1e3,
               "prompt_tok_s": t["prompt_per_second"], "gen_n": t["predicted_n"], "gen_tok_s": t["predicted_per_second"],
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "wall_s": wall}
        rows.append(row)
        print(f"[ctx] prompt {row['prompt_n']:6d} tok: prefill {row['prompt_tok_s']:7.1f} tok/s ({row['prompt_s']:6.1f} s to first token)"
              f"  decode {row['gen_tok_s']:5.2f} tok/s  hash {hashlib.md5(r['choices'][0]['message']['content'].encode()).hexdigest()[:10]}",
              flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
