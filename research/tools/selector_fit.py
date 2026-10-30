"""Target-specific adaptation of the DFlash2 selector from verification feedback (no drafter retraining).

The selector scores candidate c at a block position after predecessor token p as
    s(p, c) = unary[c] + E_next[c] . (E_prev[p] * gate),     gate = W_hidden h   (rank 256)
(llama.cpp src/models/dflash.cpp). With SRHPM_LATTICE_EXTRA=1 the lattice log carries gate and unary per position, and
SRHPM_ACCEPT_LOG what the target produced, so the scores can be rebuilt offline and refitted to THIS target:
    s'(p, c) = a_i * unary[c] + b_i * E_next[c] . (E_prev[p] * (gate + A gate))
(A: 256 x 256, a_i / b_i per block position), cross-entropy over the 16 candidates at the positions of the true prefix.
Fitted on the first part of the log, evaluated on the rest (later prompts): accepted tokens of the greedy chain.

usage: python -m research.tools.selector_fit ~/lat3.jsonl ~/acc3.jsonl [--train 0.7]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

MAXPOS = 7


def load(lat_path, acc_path):
    lats = [json.loads(l) for l in open(lat_path, encoding="utf-8")]
    accs = [json.loads(l) for l in open(acc_path, encoding="utf-8")]
    n = min(len(lats), len(accs))
    lats, accs = lats[:n], accs[:n]
    passes, stream = [], []
    for k, (L, A) in enumerate(zip(lats, accs)):
        if k == 0 or L["anchor"] != accs[k - 1][-1]:
            stream.append([L["anchor"]])
        seg = stream[-1]
        passes.append((L, len(stream) - 1, len(seg) - 1))
        seg.extend(A)
    return [(L, stream[s][j + 1:j + 1 + 16]) for L, s, j in passes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lat")
    ap.add_argument("acc")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    data = [(L, t) for L, t in load(a.lat, a.acc) if "gate" in L["pos"][0]]
    dr = {t.name: t for t in GGUFReader(str(Path.home() / "f0_gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf")).tensors}
    toks = set()
    for L, truth in data:
        toks.add(L["anchor"])
        for p in L["pos"]:
            toks.update(p["cand"])
    toks = sorted(toks)
    idx = {t: i for i, t in enumerate(toks)}

    def rows(name):
        t = dr[name]
        return torch.tensor(np.concatenate([dequantize(t.data[i:i + 1], t.tensor_type) for i in toks]).astype(np.float32))

    Ep, En = rows("selector_predecessor.weight"), rows("selector_successor.weight")  # [n_tok, 256]
    print(f"{len(data)} passes, {len(toks)} distinct tokens", flush=True)

    # check: rebuild the logged scores (position 2, predecessor index 0)
    errs = []
    for L, _ in data[:50]:
        if len(L["pos"]) < 2:
            continue
        p0, p1 = L["pos"][0], L["pos"][1]
        g = torch.tensor(p1["gate"])
        prev = idx[p0["cand"][0]]
        s = torch.tensor(p1["unary"]) + (En[[idx[c] for c in p1["cand"]]] * (Ep[prev] * g)).sum(1)
        errs.append(float((s - torch.tensor(p1["score"][:16])).abs().max()))
    print(f"rebuilt vs logged scores: max abs diff {max(errs):.3f} (logged with 3 decimals; Q4_K tables)", flush=True)

    # training samples along the true prefix
    def samples(part):
        S = []
        for L, truth in part:
            for i in range(min(MAXPOS, len(L["pos"]), len(truth))):
                p = L["pos"][i]
                if truth[i] not in p["cand"]:
                    break
                prev = L["anchor"] if i == 0 else truth[i - 1]
                S.append((i, idx[prev], [idx[c] for c in p["cand"]], p["cand"].index(truth[i]), p["gate"], p["unary"]))
        return S

    ntr = int(len(data) * a.train)
    tr = samples(data[:ntr])
    P = torch.tensor([s[0] for s in tr])
    PR = torch.tensor([s[1] for s in tr])
    C = torch.tensor([s[2] for s in tr])
    Y = torch.tensor([s[3] for s in tr])
    G = torch.tensor([s[4] for s in tr])
    U = torch.tensor([s[5] for s in tr])
    A = torch.zeros(256, 256, requires_grad=True)
    al = torch.ones(MAXPOS, requires_grad=True)
    be = torch.ones(MAXPOS, requires_grad=True)
    opt = torch.optim.Adam([A, al, be], lr=3e-3)

    def score(pos, prev_idx, cand_idx, gate, unary, adapted=True):
        g = gate + gate @ A.T if adapted else gate
        tr_ = (En[cand_idx] * (Ep[prev_idx] * g).unsqueeze(-2)).sum(-1)
        if not adapted:
            return unary + tr_
        return al[pos].unsqueeze(-1) * unary + be[pos].unsqueeze(-1) * tr_

    for ep in range(a.epochs):
        opt.zero_grad()
        s = score(P, PR, C, G, U)
        loss = torch.nn.functional.cross_entropy(s, Y) + 1e-3 * (A ** 2).sum()
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            acc = float((s.argmax(1) == Y).float().mean())
            print(f"epoch {ep}: loss {float(loss):.4f}, train top-1 {acc:.1%}", flush=True)

    # held-out: greedy chain of MAXPOS with original vs adapted scores
    with torch.no_grad():
        for adapted in (False, True):
            tok = []
            for L, truth in data[ntr:]:
                prev, n = idx[L["anchor"]], 0
                for i in range(min(MAXPOS, len(L["pos"]))):
                    p = L["pos"][i]
                    s = score(torch.tensor([i]), torch.tensor([prev]), torch.tensor([[idx[c] for c in p["cand"]]]),
                              torch.tensor([p["gate"]]), torch.tensor([p["unary"]]), adapted)[0]
                    k = int(s.argmax())
                    if i >= len(truth) or p["cand"][k] != truth[i]:
                        break
                    n += 1
                    prev = idx[p["cand"][k]]
                tok.append(1 + n)
            print(f"{'adapted' if adapted else 'original'} selector, held-out chain of {MAXPOS}: tau {np.mean(tok):.3f} "
                  f"({len(tok)} passes)", flush=True)
        print("per-position scales a_i:", np.round(al.detach().numpy(), 3), "b_i:", np.round(be.detach().numpy(), 3))


if __name__ == "__main__":
    main()
