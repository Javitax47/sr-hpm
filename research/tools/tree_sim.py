"""Offline simulation of draft-tree policies on logged DFlash2 lattices (SRHPM_LATTICE_LOG / SRHPM_ACCEPT_LOG).

For every verification pass the lattice holds, per draft position, 16 candidate tokens and a 16 x 16 table of scores
(previous candidate x candidate). The tokens the target really produced come from the accepted-token log. A policy
picks the tokens to verify (a chain or a tree); it yields 1 + the depth of the longest tree path matching the target,
and costs the measured pass time T(M) for M = verified tokens + 1. Speed is relative to the measured chain policy.

usage: python -m research.tools.tree_sim ~/lattice.jsonl ~/accept.jsonl
"""
import heapq
import json
import math
import os
import sys

# measured pass time (s) vs tokens per pass, Qwen3.8-27B hybrid, llama-bench pp<M> (28 oct. 2026)
T_PASS = {1: 0.338, 2: 0.314, 3: 0.328, 4: 0.356, 5: 0.445, 6: 0.495, 7: 0.549, 8: 0.605, 10: 0.797, 12: 0.884, 16: 1.134}
# SRHPM_T_PASS="2:0.31,4:0.36,..." replaces it (e.g. the in-server curve from SRHPM_PASS_LOG, in seconds)
if os.environ.get("SRHPM_T_PASS"):
    T_PASS = {int(k): float(v) for k, v in (kv.split(":") for kv in os.environ["SRHPM_T_PASS"].split(","))}


def t_pass(m):
    if m in T_PASS:
        return T_PASS[m]
    ks = sorted(T_PASS)
    lo = max(k for k in ks if k <= m)
    hi = min(k for k in ks if k >= m)
    return T_PASS[lo] + (T_PASS[hi] - T_PASS[lo]) * (m - lo) / (hi - lo)


def softmax(v):
    mx = max(v)
    e = [math.exp(x - mx) for x in v]
    s = sum(e)
    return [x / s for x in e]


def load(lat_path, acc_path):
    lats = [json.loads(l) for l in open(lat_path, encoding="utf-8")]
    accs = [json.loads(l) for l in open(acc_path, encoding="utf-8")]
    assert len(lats) == len(accs), (len(lats), len(accs))
    # token stream per request: a pass's anchor is the last token the previous pass produced
    passes, stream, starts = [], [], []
    for k, (L, A) in enumerate(zip(lats, accs)):
        if k == 0 or L["anchor"] != accs[k - 1][-1]:
            stream.append([L["anchor"]])  # new request
        seg = stream[-1]
        passes.append((L, len(stream) - 1, len(seg) - 1))
        seg.extend(A)
    out = []
    for L, s, j in passes:
        truth = stream[s][j + 1:j + 1 + 16]
        pos = [(p["cand"], [softmax(p["score"][16 * a:16 * a + 16]) for a in range(16)]) for p in L["pos"]]
        out.append((L, truth, pos))
    return out


def matched(tree_children, truth):
    """tree_children: dict node -> {token: child}; root = 0. Returns accepted draft tokens."""
    node, n = 0, 0
    for tok in truth:
        nxt = tree_children.get(node, {}).get(tok)
        if nxt is None:
            break
        node, n = nxt, n + 1
    return n


def chain_tree(pos, length):
    ch, pred, node = {}, 0, 0
    for i in range(min(length, len(pos))):
        cand, P = pos[i]
        k = max(range(16), key=lambda c: P[pred][c])
        ch.setdefault(node, {})[cand[k]] = node + 1
        node, pred = node + 1, k
    return ch, node


def dp_chain(pos, length):
    """Chain that maximizes the expected number of accepted drafts, sum_k prod_{j<=k} p_j, by backward dynamic
    programming over the lattice (greedy maximizes each step alone)."""
    n = min(length, len(pos))
    V = [0.0] * 16  # value after position i, per candidate index at i
    nxt = []
    for i in range(n - 1, -1, -1):
        _, P = pos[i]
        Vi, arg = [], []
        for a in range(16):  # predecessor candidate index (position i-1)
            row = P[a]
            best_c = max(range(16), key=lambda c: row[c] * (1 + V[c]))
            Vi.append(row[best_c] * (1 + V[best_c]))
            arg.append(best_c)
        nxt.append(arg)
        V = Vi
    nxt.reverse()
    ch, pred, node = {}, 0, 0
    for i in range(n):
        k = nxt[i][pred]
        ch.setdefault(node, {})[pos[i][0][k]] = node + 1
        node, pred = node + 1, k
    return ch, node


def best_tree(pos, budget):
    """Best-first tree: the `budget` nodes of highest path probability (EAGLE-2 / DDTree style)."""
    ch, nid = {}, 0
    heap = []  # (-p, depth, parent node, parent candidate index, candidate index)
    cand, P = pos[0]
    for k in range(16):
        heapq.heappush(heap, (-P[0][k], 0, 0, k))
    used = 0
    while heap and used < budget:
        negp, depth, parent, k = heapq.heappop(heap)
        nid += 1
        used += 1
        ch.setdefault(parent, {})[pos[depth][0][k]] = nid
        if depth + 1 < len(pos):
            _, Pn = pos[depth + 1]
            for c in range(16):
                heapq.heappush(heap, (negp * Pn[k][c], depth + 1, nid, c))
    return ch, used


def main():
    data = load(sys.argv[1], sys.argv[2])
    n = len(data)
    # the policy that ran: the logged draft as a chain
    tok = sum(1 + matched({i: {t: i + 1} for i, t in enumerate(L["draft"])}, truth) for L, truth, _ in data)
    tim = sum(t_pass(len(L["draft"]) + 1) for L, _, _ in data)
    base = tok / tim
    print(f"{n} passes. Logged policy (3 + confidence extension): tau {tok / n:.2f}, mean M {sum(len(L['draft']) + 1 for L, _, _ in data) / n:.2f}, relative speed 1.00")
    rows = []
    for length in range(1, 8):
        tok = sum(1 + matched(chain_tree(pos, length)[0], truth) for _, truth, pos in data)
        rows.append((f"chain {length}", tok / n, length + 1, tok / (n * t_pass(length + 1)) / base))
        tok = sum(1 + matched(dp_chain(pos, length)[0], truth) for _, truth, pos in data)
        rows.append((f"dp chain {length}", tok / n, length + 1, tok / (n * t_pass(length + 1)) / base))
    for budget in (3, 4, 5, 6, 7, 9, 11, 15):
        tok = sum(1 + matched(best_tree(pos, budget)[0], truth) for _, truth, pos in data)
        rows.append((f"tree {budget} nodes", tok / n, budget + 1, tok / (n * t_pass(budget + 1)) / base))
    # two chains as two sequences (what a recurrent/hybrid target can verify): chain A of length la, chain B forks at
    # the least confident of the first 3 positions with the runner-up candidate and continues greedily for lb tokens.
    # B repeats the shared prefix in its own sequence and needs a copy of the recurrent state (SRHPM_T_COPY seconds)
    t_copy = float(os.environ.get("SRHPM_T_COPY", "0.010"))
    for la, lb in ((3, 1), (3, 2), (3, 3), (4, 2), (2, 2)):
        tok = tim = 0.0
        for _, truth, pos in data:
            ch, used = chain_tree(pos, la)
            # confidences along A
            pred, conf, ks = 0, [], []
            for i in range(min(la, len(pos))):
                P = pos[i][1][pred]
                k = max(range(16), key=lambda c: P[c])
                conf.append(P[k])
                ks.append((pred, k))
                pred = k
            p = min(range(min(3, len(conf))), key=lambda i: conf[i])
            a_pred, a_k = ks[p]
            P = pos[p][1][a_pred]
            k2 = sorted(range(16), key=lambda c: -P[c])[1]
            node, nid = p, used  # fork from A's node at depth p
            ch.setdefault(node, {})[pos[p][0][k2]] = nid + 1
            nid += 1
            cur, pred = nid, k2
            for i in range(p + 1, min(p + lb, len(pos))):
                P = pos[i][1][pred]
                k = max(range(16), key=lambda c: P[c])
                ch.setdefault(cur, {})[pos[i][0][k]] = nid + 1
                nid += 1
                cur, pred = nid, k
            m = (1 + used) + (1 + p + min(lb, len(pos) - p))
            tok += 1 + matched(ch, truth)
            tim += t_pass(m) + t_copy
        rows.append((f"2 chains {la}+{lb}", tok / n, round((1 + la) + (1 + lb)), tok / tim / base))
    # adaptive: per pass, the size that maximizes expected tokens / pass time, from the lattice's own probabilities
    for kind in ("chain", "tree"):
        for temp in (1.0,):
            tok = tim = 0.0
            ms = 0
            for _, truth, pos in data:
                best = None
                for size in range(1, 16 if kind == "tree" else 8):
                    if kind == "chain":
                        ch, used = chain_tree(pos, size)
                        # expected accepted = sum of prefix probabilities along the chain
                        e, p, pred, node = 0.0, 1.0, 0, 0
                        for i in range(used):
                            cand, P = pos[i]
                            k = max(range(16), key=lambda c: P[pred][c])
                            p *= P[pred][k]
                            e += p
                            pred = k
                    else:
                        ch, used, e = best_tree_e(pos, size)
                    v = (1 + e) / t_pass(used + 1)
                    if best is None or v > best[0]:
                        best = (v, ch, used)
                _, ch, used = best
                tok += 1 + matched(ch, truth)
                tim += t_pass(used + 1)
                ms += used + 1
            rows.append((f"adaptive {kind}", tok / n, round(ms / n), tok / tim / base))
    # oracle chain: the drafter's own argmax chain, stopped exactly where it first goes wrong (+1 token)
    tok = tim = 0.0
    for _, truth, pos in data:
        full, _ = chain_tree(pos, 7)
        a = matched(full, truth)
        L = min(7, a + 1)
        tok += 1 + min(a, L)
        tim += t_pass(L + 1)
    rows.append(("oracle chain", tok / n, 0, tok / tim / base))
    print(f"{'policy':16s} {'tau':>5s} {'M':>3s} {'speed vs logged':>16s}")
    for name, tau, m, sp in rows:
        print(f"{name:16s} {tau:5.2f} {m:3d} {sp:16.3f}")


def best_tree_e(pos, budget):
    """best_tree plus the expected number of accepted drafts (sum of the chosen nodes' path probabilities)."""
    ch, nid, e = {}, 0, 0.0
    heap = []
    _, P = pos[0]
    for k in range(16):
        heapq.heappush(heap, (-P[0][k], 0, 0, k))
    used = 0
    while heap and used < budget:
        negp, depth, parent, k = heapq.heappop(heap)
        nid += 1
        used += 1
        e += -negp
        ch.setdefault(parent, {})[pos[depth][0][k]] = nid
        if depth + 1 < len(pos):
            _, Pn = pos[depth + 1]
            for c in range(16):
                heapq.heappush(heap, (negp * Pn[k][c], depth + 1, nid, c))
    return ch, used, e


if __name__ == "__main__":
    main()
