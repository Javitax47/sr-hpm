"""Decode speed on a running llama-server over a fixed set of 10 mixed prompts (greedy, thinking off), with a hash
of the outputs so that configurations that must be exact can be compared. tau = generated / verification passes.

usage: python -m bench.dec_bench --port 8080 --n 10 --max-tokens 200 --label base
"""
import argparse
import hashlib
import json
import time
import urllib.request

# the prompt set of every decode number in the paper: code, math, prose and translation, in Spanish and English
PROMPTS = [
    "Escribe una función en Python que devuelva los n primeros números primos usando la criba de Eratóstenes, con docstring y tests.",
    "Explica paso a paso cómo resolver la ecuación 3x^2 - 12x + 9 = 0 y comprueba las soluciones.",
    "Resume en un párrafo las causas principales de la Revolución Francesa.",
    "Write a SQL query that returns, for each customer, their total spend in 2025 and their rank by spend.",
    "A train leaves at 14:35 and travels 312 km at 96 km/h. At what time does it arrive? Show your reasoning.",
    "Translate into English and then explain the grammar: 'Si hubiera sabido que venías, habría preparado la cena.'",
    "Devuelve un objeto JSON con los campos nombre, edad, ciudad e intereses (lista) para una persona ficticia, sin texto adicional.",
    "Describe the difference between a mutex and a semaphore, with a short C example of each.",
    "Escribe un correo formal pidiendo un aplazamiento de una entrega de proyecto una semana.",
    "Prove that the square root of 2 is irrational.",
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    gen, secs, dn, da, h = 0, 0.0, 0, 0, hashlib.sha1()
    per = []
    for i in range(a.first, a.first + a.n):
        r = post(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": PROMPTS[i]}], "max_tokens": a.max_tokens,
            "temperature": 0.0, "top_k": 1, "seed": 0, "chat_template_kwargs": {"enable_thinking": False},
        })
        t = r["timings"]
        gen += t["predicted_n"]
        secs += t["predicted_ms"] / 1e3
        dn += t.get("draft_n") or 0
        da += t.get("draft_n_accepted") or 0
        h.update(r["choices"][0]["message"]["content"].encode())
        per.append(t["predicted_per_second"])
    acc = da / dn if dn else 0.0
    # each verification pass yields its accepted drafts + 1 token: passes = generated - accepted
    tau = gen / (gen - da) if gen > da else 0.0
    print(f"[dec] {a.label:24s} {gen / secs:5.2f} tok/s  acc {acc:.3f}  tau {tau:.2f}  per-prompt "
          + " ".join(f"{x:4.1f}" for x in per) + f"  hash {h.hexdigest()[:10]}", flush=True)


if __name__ == "__main__":
    main()
