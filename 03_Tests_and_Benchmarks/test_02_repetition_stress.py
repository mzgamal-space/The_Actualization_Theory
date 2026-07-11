"""
test_02_repetition_stress.py — Repetition Loop Suppression Test
=================================================================
Forces the recent token history to be filled with one repeated token.
Measures whether the Actualizer's Order Prime successfully suppresses
the repeating token and redirects probability mass to valid alternatives.

Returns a dict of results for use by generate_all_charts.py.
"""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from actualizer_engine import ActualizerEngine
from fdsa_pruner import VectorizedFDSAPruner


def run(vocab_size: int = 300, steps: int = 40, seed: int = 42) -> dict:
    random.seed(seed)

    # Factual token set: 100..149
    target_tokens = set(range(100, 150))
    # Grammar: each token in [100,148] can go to next or +5
    grammar = {}
    for t in range(100, 149):
        grammar[t] = {t + 1, ((t - 100 + 5) % 49) + 100}
    grammar[149] = {100, 105}

    pruner = VectorizedFDSAPruner(vocab_size=vocab_size, k=0.35)
    engine = ActualizerEngine(vocab_size=vocab_size, k=0.45, Q_c=1e-5,
                               repetition_penalty=3.0)

    repeated_token = 100   # the token that will be forced into history

    base_repeat_counts  = []
    fdsa_repeat_counts  = []
    base_diversity      = []  # unique tokens in last 10 steps
    fdsa_diversity      = []

    hist_base = [100, 100, 100, 100, 100]   # pre-filled with repeats
    hist_fdsa = [100, 100, 100, 100, 100]

    for step in range(steps):
        random.seed(seed + step)
        logits = [random.gauss(-3.0, 1.0) for _ in range(vocab_size)]

        # The repeated token gets a moderate self-boost (simulating inertia)
        logits[repeated_token] += 4.5

        # Valid transitions also get a small boost
        for tok in grammar.get(hist_base[-1], set()):
            logits[tok] += 2.0

        # --- Baseline ---
        max_l = max(logits)
        exps  = [math.exp(x - max_l) for x in logits]
        s     = sum(exps)
        probs = [e / s for e in exps]
        tok_b = probs.index(max(probs))
        hist_base.append(tok_b)
        base_repeat_counts.append(1 if tok_b == repeated_token else 0)

        # --- FDSA + Actualizer ---
        pruned, _ = pruner.prune_vocabulary(logits, hist_fdsa[-1], grammar, "logical_coding")
        tok_a, _, _, _ = engine.steer(pruned, hist_fdsa, target_tokens)
        hist_fdsa.append(tok_a)
        fdsa_repeat_counts.append(1 if tok_a == repeated_token else 0)

    # Compute diversity: unique tokens in rolling 10-step window
    for i in range(steps):
        window_b = hist_base[max(0, i-9):i+1]
        window_a = hist_fdsa[max(0, i-9):i+1]
        base_diversity.append(len(set(window_b)))
        fdsa_diversity.append(len(set(window_a)))

    base_repeat_rate = sum(base_repeat_counts) / steps * 100
    fdsa_repeat_rate = sum(fdsa_repeat_counts) / steps * 100

    return {
        "steps"             : list(range(steps)),
        "base_repeat_counts": base_repeat_counts,
        "fdsa_repeat_counts": fdsa_repeat_counts,
        "base_diversity"    : base_diversity,
        "fdsa_diversity"    : fdsa_diversity,
        "base_repeat_rate"  : round(base_repeat_rate, 2),
        "fdsa_repeat_rate"  : round(fdsa_repeat_rate, 2),
        "base_seq"          : hist_base,
        "fdsa_seq"          : hist_fdsa,
    }


if __name__ == "__main__":
    r = run()
    print(f"Baseline repeat rate   : {r['base_repeat_rate']}%")
    print(f"FDSA+Actualizer repeat : {r['fdsa_repeat_rate']}%")
    print(f"Baseline avg diversity : {sum(r['base_diversity'])/len(r['base_diversity']):.2f} unique/10 steps")
    print(f"FDSA avg diversity     : {sum(r['fdsa_diversity'])/len(r['fdsa_diversity']):.2f} unique/10 steps")
