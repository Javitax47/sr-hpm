"""Refutation test for "the GPU half-model reranks the draft": does the target's hidden state after its GPU layers
(the input of the first CPU layer) tell which of the drafter's 16 candidates is right?

Logs (one verification pass each): SRHPM_LATTICE_LOG (candidates and scores), SRHPM_ACCEPT_LOG (what the target
produced), GGML_SRV_DUMP_NAMES (input rows of blk.<L>.attn_qkv = attn_norm(h_<L-1>), one row per verified token).
Logit lens restricted to the candidates: score(c) = (x / attn_norm_w * output_norm_w) . W_out[c].
Rows are valid only along the true prefix (row j's input is draft token j, true iff the first j drafts were accepted),
which covers the position where the chain broke: the case a reranker would rescue.

usage: python -m research.tools.lens_test ~/lat2.jsonl ~/acc2.jsonl ~/dump2.bin
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from research.tools.tree_sim import load, matched  # noqa: E402


def read_dump(path):
    recs = {}
    b = Path(path).read_bytes()
    o = 0
    while o + 12 <= len(b):
        layer, M, K = np.frombuffer(b, np.int32, 3, o)
        o += 12
        x = np.frombuffer(b, np.float32, M * K, o).reshape(M, K)
        o += 4 * M * K
        recs.setdefault(int(layer), []).append(x)
    return recs


def main():
    data = load(sys.argv[1], sys.argv[2])
    recs = read_dump(sys.argv[3])
    r = GGUFReader(str(Path.home() / "f0_gguf/q_speed1.gguf"))
    T = {t.name: t for t in r.tensors}
    onorm = np.asarray(T["output_norm.weight"].data, np.float32)
    wout = T["output.weight"]
    cache = {}

    def wrow(tok):
        if tok not in cache:
            cache[tok] = dequantize(wout.data[tok:tok + 1], wout.tensor_type).astype(np.float32)[0]
        return cache[tok]

    for layer in sorted(recs):
        anorm = np.asarray(T[f"blk.{layer}.attn_norm.weight"].data, np.float32)
        rows = [x for x in recs[layer] if x.shape[0] >= 2]
        # align: the k-th pass verifies len(draft) + 1 tokens
        k = 0
        pairs = []
        for L, truth, pos in data:
            m = len(L["draft"]) + 1
            while k < len(rows) and rows[k].shape[0] != m:
                k += 1
            if k == len(rows):
                break
            pairs.append((L, truth, pos, rows[k]))
            k += 1
        # samples: positions along the true prefix with the truth among the 16 candidates
        S = []
        for L, truth, pos, X in pairs:
            a = matched({i: {t: i + 1} for i, t in enumerate(L["draft"])}, truth)
            pred = 0
            for i in range(min(a + 1, len(pos), len(truth), X.shape[0])):
                cand, P = pos[i]
                if truth[i] not in cand:
                    break
                z = X[i] / anorm * onorm
                lens = np.array([float(z @ wrow(c)) for c in cand])
                lp = np.log(np.array(P[pred]) + 1e-9)
                S.append((lp, lens, cand.index(truth[i]), i == a))
                pred = cand.index(truth[i])
        half = len(S) // 2

        def acc(beta, part, only_break=False):
            ok = n = 0
            for lp, lens, t, brk in part:
                if only_break and not brk:
                    continue
                ls = lens - lens.max()
                ls = ls - math.log(np.exp(ls).sum())
                ok += int(np.argmax(lp + beta * ls) == t)
                n += 1
            return ok / max(n, 1), n

        betas = [0, 0.1, 0.25, 0.5, 1, 2, 4, 1e3]
        best = max(betas, key=lambda b: acc(b, S[:half])[0])
        a0, n = acc(0, S[half:])
        ab, _ = acc(best, S[half:])
        al, _ = acc(1e3, S[half:])
        b0, nb = acc(0, S[half:], True)
        bb, _ = acc(best, S[half:], True)
        print(f"layer {layer:2d} input ({len(pairs)} passes, {n} held-out positions): top-1 among the 16 - drafter {a0:.1%}, "
              f"lens alone {al:.1%}, drafter + {best} x lens {ab:.1%};  at the breaking position ({nb}): "
              f"drafter {b0:.1%} -> combined {bb:.1%}", flush=True)


if __name__ == "__main__":
    main()
