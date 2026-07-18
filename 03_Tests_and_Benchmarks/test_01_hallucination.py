"""
test_01_hallucination.py — Hallucination Resistance Test  [V3_U1 updated]
==========================================================
Injects a strong distractor token (logit +8.0) at every generation step.
Measures how many steps remain factually grounded after 30 tokens.

V3_U1 changes:
  - engine.steer() now returns 6-tuple: (token, U, Tr_D, iters, nu_history, actualized)
  - tracks nu_t (valuation trajectory) and actualized (bifurcation branch) per step
  - mercy_k replaces k in ActualizerEngine constructor

Returns a dict of results for use by generate_all_charts.py.
"""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from actualizer_engine import ActualizerEngine
from fdsa_pruner import VectorizedFDSAPruner


def run(vocab_size: int = 500, steps: int = 30, seed: int = 0) -> dict:
    random.seed(seed)

    grammar = {}
    for t in range(50, 100):
        nxt = t + 1 if t < 99 else 50
        branch = ((t - 50 + 7) % 50) + 50
        grammar[t] = {nxt, branch}

    target_tokens = set(range(50, 100))

    pruner = VectorizedFDSAPruner(vocab_size=vocab_size, k=0.35)
    engine = ActualizerEngine(vocab_size=vocab_size, mercy_k=0.45, Q_c=1e-5)

    hist_base   = [50]
    hist_fdsa   = [50]
    base_errors = 0
    fdsa_errors = 0
    base_grounded   = []
    fdsa_grounded   = []

    for step in range(steps):
        random.seed(seed + step)
        logits = [random.gauss(-3.0, 1.0) for _ in range(vocab_size)]

        # Valid next tokens get a +3 boost
        for tok in grammar.get(hist_base[-1], set()):
            logits[tok] += 3.0

        # DISTRACTOR: strong +8 bait at token 499
        bait = 499
        logits[bait] = 8.0

        # --- Baseline (raw softmax, no steering) ---
        max_l = max(logits)
        exps  = [math.exp(x - max_l) for x in logits]
        s     = sum(exps)
        probs = [e / s for e in exps]
        tok_b = probs.index(max(probs))
        hist_base.append(tok_b)
        grounded_b = tok_b in grammar.get(hist_base[-2], set())
        if not grounded_b:
            base_errors += 1
        base_grounded.append(1 if grounded_b else 0)

        # --- FDSA + Actualizer ---
        pruned, _ = pruner.prune_vocabulary(logits, hist_fdsa[-1], grammar, "logical_coding")
        tok_a, _, Tr_D, n_iters, nu_hist, actualized = engine.steer(
            pruned, hist_fdsa, target_tokens
        )
        hist_fdsa.append(tok_a)
        grounded_a = tok_a in grammar.get(hist_fdsa[-2], set())
        if not grounded_a:
            fdsa_errors += 1
        fdsa_grounded.append(1 if grounded_a else 0)

    return {
        "steps"          : list(range(steps)),
        "base_grounded"  : base_grounded,
        "fdsa_grounded"  : fdsa_grounded,
        "base_errors"    : base_errors,
        "fdsa_errors"    : fdsa_errors,
        "base_rate"      : round((steps - base_errors) / steps * 100, 2),
        "fdsa_rate"      : round((steps - fdsa_errors) / steps * 100, 2),
        "base_seq"       : hist_base,
        "fdsa_seq"       : hist_fdsa,
        "v3u1"           : "steer() returns 6-tuple: (token, U, Tr_D, iters, nu_history, actualized)",
    }


if __name__ == "__main__":
    r = run()
    print(f"Baseline groundedness : {r['base_rate']}%  ({r['base_errors']} errors / {len(r['steps'])} steps)")
    print(f"FDSA+Actualizer       : {r['fdsa_rate']}%  ({r['fdsa_errors']} errors / {len(r['steps'])} steps)")
