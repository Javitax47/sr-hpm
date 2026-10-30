"""Refutation test: can the drafter be corrected online from what verification reveals?

Every verification pass tells which token the target wanted at each drafted position: a free label. A small
reranker over the DFlash2 lattice's 16 candidates (features: selector score / probability / rank, draft position,
whether the candidate already occurs in the recent output, how recently) is trained on the first half of the logged
passes and evaluated on the second, as a stand-in for learning while decoding. Metric: the drafter's chain rebuilt with
the reranker, accepted tokens per pass vs the greedy chain.

usage: python -m research.tools.rerank_test ~/lattice.jsonl ~/accept.jsonl
"""
import math
import sys

import numpy as np

from research.tools.tree_sim import load, matched

MAXPOS = 7


def feats(pos_i, i, pred, hist, cand_idx):
    cand, P = pos_i
    p = P[pred]
    order = sorted(range(16), key=lambda c: -p[c])
    rank = {c: r for r, c in enumerate(order)}
    tok = cand[cand_idx]
    last = len(hist) - 1 - hist[::-1].index(tok) if tok in hist else -1
    rec = (len(hist) - last) if last >= 0 else 999
    return [math.log(p[cand_idx] + 1e-9), p[cand_idx], 1.0 if rank[cand_idx] == 0 else 0.0, rank[cand_idx] / 15,
            i / MAXPOS, 1.0 if last >= 0 else 0.0, 1.0 / rec if last >= 0 else 0.0,
            # follows the same continuation as an earlier occurrence of the previous token (copy / induction)
            0.0]


def chain(pos, hist, w, length=MAXPOS):
    ch, pred, node, h = {}, 0, 0, list(hist)
    for i in range(min(length, len(pos))):
        cand, _ = pos[i]
        if w is None:
            k = max(range(16), key=lambda c: pos[i][1][pred][c])
        else:
            prev = h[-1] if h else -1
            best, k = -1e30, 0
            for c in range(16):
                f = feats(pos[i], i, pred, h[-256:], c)
                # induction: candidate equals the token that followed `prev` last time
                for j in range(len(h) - 2, max(-1, len(h) - 256), -1):
                    if h[j] == prev:
                        f[7] = 1.0 if h[j + 1] == cand[c] else 0.0
                        break
                s = float(np.dot(w, f))
                if s > best:
                    best, k = s, c
        ch.setdefault(node, {})[cand[k]] = node + 1
        node, pred = node + 1, k
        h.append(cand[k])
    return ch


def main():
    data = load(sys.argv[1], sys.argv[2])
    # history per pass: the stream up to the pass's anchor
    hists, stream = [], []
    for L, truth, pos in data:
        if not stream or L["anchor"] != stream[-1]:
            stream = [L["anchor"]]
        hists.append(list(stream))
        stream = stream + truth[:0]
    # rebuild history from truth: tree_sim stores the true continuation, so the stream is anchor + truth prefix
    hists = []
    s = []
    for k, (L, truth, pos) in enumerate(data):
        if k == 0 or L["anchor"] != (s[-1] if s else None):
            s = [L["anchor"]]
        hists.append(list(s))
        n = 1 + matched({i: {t: i + 1} for i, t in enumerate(L["draft"])}, truth)
        s = s + truth[:n]
    half = len(data) // 2
    # training set: at each position along the TRUE prefix, which candidate is the true token
    X, y = [], []
    for (L, truth, pos), h in zip(data[:half], hists[:half]):
        pred, hh = 0, list(h)
        for i in range(min(MAXPOS, len(pos), len(truth))):
            cand, _ = pos[i]
            if truth[i] not in cand:
                break
            for c in range(16):
                f = feats(pos[i], i, pred, hh[-256:], c)
                prev = hh[-1]
                for j in range(len(hh) - 2, max(-1, len(hh) - 256), -1):
                    if hh[j] == prev:
                        f[7] = 1.0 if hh[j + 1] == cand[c] else 0.0
                        break
                X.append(f)
                y.append(1.0 if cand[c] == truth[i] else 0.0)
            pred = cand.index(truth[i])
            hh.append(truth[i])
    X, y = np.array(X), np.array(y)
    # logistic regression by gradient descent (pairwise-free; ranking within a position through the score)
    w = np.zeros(X.shape[1])
    for _ in range(3000):
        z = 1 / (1 + np.exp(-(X @ w)))
        w -= 0.5 * X.T @ (z - y) / len(y)
    print("weights:", np.round(w, 3), f"({len(y) // 16} labelled positions)")
    for name, ww in (("greedy", None), ("reranked", w)):
        tok = [1 + matched(chain(pos, h, ww), truth) for (L, truth, pos), h in zip(data[half:], hists[half:])]
        print(f"{name:9s} chain of {MAXPOS}: tau {np.mean(tok):.3f} on {len(tok)} held-out passes")


if __name__ == "__main__":
    main()
