"""
test_03_pre_inference_speed.py — Pre-Inference Speed Sweep
===========================================================
Sweeps vocabulary sizes V = 1k, 5k, 10k, 30k, 50k, 100k.
At each V, runs 50 timing trials and records:
  - Baseline softmax latency (NumPy, full V)
  - FDSA-pruned softmax latency (NumPy, pruned subset)
  - Active vocab size after pruning
  - Speedup factor
  - Pruning rate %

Returns a dict of results for use by generate_all_charts.py.
"""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from fdsa_pruner import VectorizedFDSAPruner


def _baseline_softmax(logits: np.ndarray) -> int:
    """Standard full-vocabulary softmax + argmax."""
    shifted = logits - logits.max()
    exp_l   = np.exp(shifted)
    probs   = exp_l / exp_l.sum()
    return int(np.argmax(probs))


def _pruned_softmax(logits: np.ndarray) -> int:
    """Softmax over valid (non -inf) entries only."""
    mask   = np.isfinite(logits)
    if not mask.any():
        return 0
    valid  = logits[mask]
    shifted = valid - valid.max()
    exp_l  = np.exp(shifted)
    probs  = exp_l / exp_l.sum()
    indices = np.where(mask)[0]
    return int(indices[np.argmax(probs)])


def run(
    vocab_sizes = (1_000, 5_000, 10_000, 30_000, 50_000, 100_000),
    trials      : int = 50,
    seed        : int = 7,
) -> dict:
    np.random.seed(seed)

    results = {
        "vocab_sizes"      : list(vocab_sizes),
        "baseline_ms"      : [],
        "fdsa_ms"          : [],
        "speedup"          : [],
        "active_vocab"     : [],
        "pruning_rate_pct" : [],
    }

    for V in vocab_sizes:
        pruner  = VectorizedFDSAPruner(vocab_size=V, k=0.35)

        # Simple grammar: anchor token is V//2; it can go to V//2+1 or V//2+3
        anchor = V // 2
        grammar = {anchor: {anchor + 1, anchor + 3}}

        base_times, fdsa_times = [], []
        active_sizes = []

        for t in range(trials):
            np.random.seed(seed + t)
            logits = np.random.normal(-3.0, 1.0, size=(V,))
            # Inject a valid transition boost
            logits[anchor + 1] += 3.0
            # Inject distractor bait
            logits[V - 1] = 6.0

            # --- Baseline timing ---
            t0 = time.perf_counter()
            _baseline_softmax(logits)
            base_times.append((time.perf_counter() - t0) * 1000)

            # --- FDSA timing ---
            t0 = time.perf_counter()
            pruned_logits, active = pruner.prune_numpy(
                logits, anchor, grammar, "logical_coding"
            )
            _pruned_softmax(pruned_logits)
            fdsa_times.append((time.perf_counter() - t0) * 1000)
            active_sizes.append(active)

        base_ms = float(np.median(base_times))
        fdsa_ms = float(np.median(fdsa_times))
        avg_active = float(np.mean(active_sizes))
        speedup = base_ms / fdsa_ms if fdsa_ms > 0 else 0.0
        pruning = (1.0 - avg_active / V) * 100.0

        results["baseline_ms"].append(round(base_ms, 4))
        results["fdsa_ms"].append(round(fdsa_ms, 4))
        results["speedup"].append(round(speedup, 2))
        results["active_vocab"].append(round(avg_active, 1))
        results["pruning_rate_pct"].append(round(pruning, 4))

        print(f"  V={V:>7,} | Base {base_ms:.4f} ms | FDSA {fdsa_ms:.4f} ms | "
              f"{speedup:.2f}× speedup | {pruning:.2f}% pruned")

    return results


if __name__ == "__main__":
    print("Running pre-inference speed sweep (V = 1k → 100k)…\n")
    r = run()
    print("\n── Summary ──")
    for i, V in enumerate(r["vocab_sizes"]):
        print(f"  V={V:>7,}: {r['speedup'][i]}× speedup, {r['pruning_rate_pct'][i]}% pruned")
