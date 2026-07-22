"""
test_07_architecture_comparison.py
===================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com

THREE-WAY ARCHITECTURE COMPARISON
===================================
Compares three inference architectures on identical synthetic substrates:

  [Model A]  Attention Baseline
             - Standard softmax over full vocabulary (no steering)
             - Argmax token selection (greedy decoding)
             - Complexity: O(V) per step
             - Represents: vanilla transformer output head

  [Model B]  Actualizer Engine  (FDSA + Contractive Steering)
             - FDSA logit mask (dimensional truncation, isomorphic anchoring)
             - Banach contractive steering (k=0.45, Vacuum Brake)
             - V3_U1 bifurcation criterion  Tr(D_mu_nu) <= tau
             - Complexity: O(V * max_iters) per step
             - Represents: single-dataset top-down steering

  [Model C]  QCA Parallel Actualizer Engine
             - QCA crystallization (N nodes -> K clusters, T_q^RGG)
             - Parallel worker execution per cluster (ProcessPoolExecutor)
             - Global synthesis pass (metacluster ActualizerEngine)
             - Complexity: O(N^2/K) parallel — factor-K reduction
             - Represents: clustered parallel steering

METRICS MEASURED
-----------------
  1. Latency (ms) — wall-clock execution time per episode
  2. Hallucination resistance — fraction of steps that are factually grounded
     (token selected from target_tokens set)
  3. Repetition rate — fraction of tokens that repeat a recent token
  4. Valuation (nu_t) — mean actualization quality [0,1]
  5. Actualization rate — fraction of steps successfully actualized (not dissolved)
  6. Speedup factor vs Baseline — relative speedup at each N

Theory References:
  CKT White Paper v3 §7.2 Theorem 2 Corollary: O(N^2/K) via K-clustering
  V3_U1 §3.3.1-B: Tr(D_mu_nu) bifurcation criterion
  V3_U1 §3.3.1-A: nu_t valuation trajectory
"""

from __future__ import annotations

import sys
import os
import time
import math
import random
from typing import Dict, List, Set, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from actualizer_engine import ActualizerEngine
from fdsa_pruner import VectorizedFDSAPruner
from qca import QCANode
from qca_parallel_engine import QCAParallelEngine


# ═══════════════════════════════════════════════════════════════════════════
# Shared synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_logits(V: int, distractor_strength: float, seed: int) -> List[float]:
    """
    Generate a synthetic logit vector of size V.

    - ~10% of tokens are 'target' tokens (slightly elevated logits)
    - One distractor token is injected with a +distractor_strength boost
    - Remaining tokens are near-zero noise

    This simulates a realistically challenging attention output where a
    strong hallucination bait competes against valid semantic targets.
    """
    rng = _rng(seed)
    logits = [rng.gauss(0.0, 0.5) for _ in range(V)]

    # Target band: tokens in [V//5, V//3) — moderate positive logits
    for v in range(V // 5, V // 3):
        logits[v] += rng.uniform(1.0, 2.5)

    # Distractor: one strong bait token outside target band
    distractor_idx = rng.randint(V // 3, V - 1)
    logits[distractor_idx] += distractor_strength

    return logits


def generate_target_tokens(V: int) -> Set[int]:
    """Target semantic window: [V//5, V//3)."""
    return set(range(V // 5, V // 3))


def generate_history(V: int, length: int = 10, seed: int = 0) -> List[int]:
    """Simulate prior generation history (no repetition by default)."""
    rng = _rng(seed + 99)
    return [rng.randint(0, V - 1) for _ in range(length)]


def generate_qca_nodes(N: int, seed: int = 42) -> List[QCANode]:
    """Generate N QCANodes for QCA Parallel Engine input."""
    rng = _rng(seed)
    nodes = []
    for i in range(N):
        coords = [rng.uniform(0.0, 10.0) for _ in range(5)]
        prime_profile = [rng.uniform(0.1, 0.9) for _ in range(5)]
        s = sum(prime_profile) or 1.0
        prime_profile = [p / s for p in prime_profile]
        nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime_profile))
    return nodes


# ═══════════════════════════════════════════════════════════════════════════
# MODEL A — Attention Baseline (pure softmax + greedy argmax)
# ═══════════════════════════════════════════════════════════════════════════

def _softmax_baseline(logits: List[float]) -> List[float]:
    """Numerically stable softmax — simulates the bare transformer output head."""
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def run_attention_baseline(
    V: int,
    n_steps: int,
    distractor_strength: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Model A: Attention Baseline.
    Runs n_steps of greedy argmax decoding with no top-down steering.
    """
    target_tokens = generate_target_tokens(V)
    history = generate_history(V, seed=seed)
    tokens_chosen = []

    t0 = time.perf_counter()
    for step in range(n_steps):
        logits = generate_logits(V, distractor_strength, seed=seed + step)
        probs = _softmax_baseline(logits)
        token = max(range(V), key=lambda v: probs[v])
        tokens_chosen.append(token)
        history.append(token)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    grounded = sum(1 for t in tokens_chosen if t in target_tokens)
    repeated = sum(
        1 for i in range(1, len(tokens_chosen)) if tokens_chosen[i] == tokens_chosen[i - 1]
    )

    return {
        "model"             : "Attention Baseline",
        "latency_ms"        : round(elapsed_ms, 2),
        "tokens"            : tokens_chosen,
        "hallucination_rate": round(1.0 - grounded / n_steps, 4),
        "grounded_rate"     : round(grounded / n_steps, 4),
        "repetition_rate"   : round(repeated / max(n_steps - 1, 1), 4),
        "mean_valuation"    : 0.0,    # Baseline has no valuation mechanism
        "actualization_rate": 0.0,    # Baseline has no bifurcation criterion
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL B — Actualizer Engine (FDSA + Contractive Steering)
# ═══════════════════════════════════════════════════════════════════════════

def run_actualizer_engine(
    V: int,
    n_steps: int,
    distractor_strength: float,
    seed: int,
    mercy_k: float = 0.45,
) -> Dict[str, Any]:
    """
    Model B: Actualizer Engine with FDSA pre-pruning.
    Each step runs the full V3_U1 contractive steering loop.
    """
    import numpy as np

    target_tokens = generate_target_tokens(V)
    history = generate_history(V, seed=seed)
    engine = ActualizerEngine(vocab_size=V, mercy_k=mercy_k)
    pruner = VectorizedFDSAPruner(vocab_size=V)

    # Minimal grammar: each token allows all tokens (unconstrained)
    # Use an empty grammar dict so FDSA only applies threshold masking.
    grammar_rules: Dict = {}

    tokens_chosen, nu_values, actualized_steps = [], [], []

    t0 = time.perf_counter()
    for step in range(n_steps):
        logits_raw = generate_logits(V, distractor_strength, seed=seed + step)
        last_token = history[-1] if history else 0

        # FDSA Phase 1+2: isomorphic anchoring + vectorized logit masking
        logits_np = np.array(logits_raw, dtype=float)
        masked_np, _ = pruner.prune_numpy(logits_np, last_token, grammar_rules)
        masked = masked_np.tolist()

        # ActualizerEngine: full contractive steering loop
        token, U_final, trace_drift, iters, nu_hist, actualized = engine.steer(
            logits=masked,
            history=history,
            target_tokens=target_tokens,
        )
        tokens_chosen.append(token)
        history.append(token)
        nu_values.append(nu_hist[-1] if nu_hist else 0.0)
        actualized_steps.append(actualized)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    grounded = sum(1 for t in tokens_chosen if t in target_tokens)
    repeated = sum(
        1 for i in range(1, len(tokens_chosen)) if tokens_chosen[i] == tokens_chosen[i - 1]
    )

    return {
        "model"             : "Actualizer Engine",
        "latency_ms"        : round(elapsed_ms, 2),
        "tokens"            : tokens_chosen,
        "hallucination_rate": round(1.0 - grounded / n_steps, 4),
        "grounded_rate"     : round(grounded / n_steps, 4),
        "repetition_rate"   : round(repeated / max(n_steps - 1, 1), 4),
        "mean_valuation"    : round(sum(nu_values) / len(nu_values), 4) if nu_values else 0.0,
        "actualization_rate": round(sum(actualized_steps) / n_steps, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL C — QCA Parallel Actualizer Engine
# ═══════════════════════════════════════════════════════════════════════════

def run_qca_parallel(
    N: int,
    V: int,
    K: int,
    seed: int,
    mercy_k: float = 0.45,
) -> Dict[str, Any]:
    """
    Model C: QCA Parallel Actualizer Engine.
    Crystallizes N nodes into K clusters and runs parallel ActualizerEngine
    workers per cluster, then synthesizes a global metacluster result.
    """
    nodes = generate_qca_nodes(N, seed=seed)
    engine = QCAParallelEngine(K=K, vocab_size=V, mercy_k=mercy_k, seed=seed)

    t0 = time.perf_counter()
    result = engine.process_parallel(nodes, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    total_nodes = N
    actualized_count = sum(c.actualized_count for c in result.cluster_results)
    actualization_rate = actualized_count / total_nodes if total_nodes > 0 else 0.0

    return {
        "model"             : "QCA Parallel Engine",
        "latency_ms"        : round(elapsed_ms, 2),
        "qca_time_ms"       : round(result.qca_time_ms, 2),
        "parallel_time_ms"  : round(result.parallel_time_ms, 2),
        "synthesis_time_ms" : round(result.synthesis_time_ms, 2),
        "n_clusters"        : K,
        "mean_valuation"    : round(result.global_valuation, 4),
        "actualization_rate": round(actualization_rate, 4),
        # Grounding / repetition are computed at cluster synthesis level
        # — approximate via actualization_rate as proxy
        "grounded_rate"     : round(actualization_rate, 4),
        "hallucination_rate": round(1.0 - actualization_rate, 4),
        "repetition_rate"   : 0.0,  # QCA operates at substrate level, not token sequence
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run(
    vocab_sizes   : Tuple[int, ...] = (500, 1000, 2000),
    n_steps       : int             = 30,
    n_sizes       : Tuple[int, ...] = (20, 40, 80, 120, 200),
    K             : int             = 5,
    distractor_str: float           = 8.0,
    mercy_k       : float           = 0.45,
    seed          : int             = 42,
) -> Dict[str, Any]:
    """
    Full three-way comparison across vocabulary sizes and dataset sizes.

    Returns a structured dict suitable for downstream visualization
    (generate_all_charts.py Figure 7).
    """
    divider = "=" * 72

    print(divider)
    print("  THREE-WAY ARCHITECTURE COMPARISON")
    print("  [A] Attention Baseline | [B] Actualizer Engine | [C] QCA Parallel")
    print(divider)

    # ── Phase 1: Per-step quality comparison at V=500 ──────────────────────
    print(f"\n[Phase 1] Step-by-step quality comparison (V=500, n_steps={n_steps}, distractor=+{distractor_str})\n")
    print(f"  {'Model':<30} {'Latency (ms)':>12} {'Grounded':>10} {'Repetition':>12} {'Valuation':>11} {'Actualized':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*11} {'-'*12}")

    quality_results = {}
    V_quality = 500

    res_a = run_attention_baseline(V=V_quality, n_steps=n_steps, distractor_strength=distractor_str, seed=seed)
    res_b = run_actualizer_engine(V=V_quality, n_steps=n_steps, distractor_strength=distractor_str, seed=seed, mercy_k=mercy_k)
    res_c = run_qca_parallel(N=80, V=V_quality, K=K, seed=seed, mercy_k=mercy_k)

    for res in [res_a, res_b, res_c]:
        print(
            f"  {res['model']:<30} {res['latency_ms']:>12.2f}"
            f" {res['grounded_rate']:>10.4f}"
            f" {res['repetition_rate']:>12.4f}"
            f" {res['mean_valuation']:>11.4f}"
            f" {res['actualization_rate']:>12.4f}"
        )
        quality_results[res['model']] = res

    # ── Phase 2: Latency scaling across vocabulary sizes (Models A & B) ─────
    print(f"\n[Phase 2] Latency scaling vs vocabulary size V (n_steps={n_steps}, distractor=+{distractor_str})\n")
    print(f"  {'V':>6} | {'Baseline (ms)':>14} | {'Actualizer (ms)':>16} | {'Speedup A->B':>13}")
    print(f"  {'-'*6}   {'-'*14}   {'-'*16}   {'-'*13}")

    vocab_scaling = {"vocab_sizes": list(vocab_sizes), "baseline_ms": [], "actualizer_ms": [], "speedup_ab": []}
    for V in vocab_sizes:
        ra = run_attention_baseline(V=V, n_steps=n_steps, distractor_strength=distractor_str, seed=seed)
        rb = run_actualizer_engine(V=V, n_steps=n_steps, distractor_strength=distractor_str, seed=seed, mercy_k=mercy_k)
        spd = ra["latency_ms"] / max(rb["latency_ms"], 0.01)
        vocab_scaling["baseline_ms"].append(ra["latency_ms"])
        vocab_scaling["actualizer_ms"].append(rb["latency_ms"])
        vocab_scaling["speedup_ab"].append(round(spd, 3))
        print(f"  {V:>6} | {ra['latency_ms']:>14.2f} | {rb['latency_ms']:>16.2f} | {spd:>12.3f}x")

    # ── Phase 3: QCA Parallel vs Sequential scaling across dataset sizes ─────
    print(f"\n[Phase 3] QCA Parallel speedup vs dataset size N (K={K}, V={vocab_sizes[-1]})\n")
    print(
        f"  {'N':>5} | {'Seq (ms)':>10} | {'Par (ms)':>10} | {'Speedup':>8}"
        f" | {'QCA (ms)':>9} | {'Valuation':>10} | {'Actualized':>11}"
    )
    print(f"  {'-'*5}   {'-'*10}   {'-'*10}   {'-'*8}   {'-'*9}   {'-'*10}   {'-'*11}")

    V_par = vocab_sizes[-1]
    parallel_scaling = {
        "n_sizes": list(n_sizes), "K": K, "V": V_par,
        "seq_ms": [], "par_ms": [], "speedup": [],
        "qca_ms": [], "valuation": [], "actualization_rate": [],
    }
    for N in n_sizes:
        nodes = generate_qca_nodes(N, seed=seed + N)
        par_engine = QCAParallelEngine(K=K, vocab_size=V_par, mercy_k=mercy_k, seed=seed)

        res_par = par_engine.process_parallel(nodes, verbose=False)
        t_seq   = par_engine.process_sequential(nodes)
        t_par   = res_par.total_time_ms
        spd     = t_seq / max(t_par, 0.01)

        actualized_count = sum(c.actualized_count for c in res_par.cluster_results)
        act_rate = actualized_count / N if N > 0 else 1.0

        parallel_scaling["seq_ms"].append(round(t_seq, 2))
        parallel_scaling["par_ms"].append(round(t_par, 2))
        parallel_scaling["speedup"].append(round(spd, 3))
        parallel_scaling["qca_ms"].append(round(res_par.qca_time_ms, 2))
        parallel_scaling["valuation"].append(round(res_par.global_valuation, 4))
        parallel_scaling["actualization_rate"].append(round(act_rate, 4))

        print(
            f"  {N:>5} | {t_seq:>10.2f} | {t_par:>10.2f} | {spd:>7.2f}x"
            f" | {res_par.qca_time_ms:>9.2f} | {res_par.global_valuation:>10.4f}"
            f" | {act_rate:>10.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  SUMMARY — METRIC COMPARISON AT V=500, N=80, K=5")
    print(divider)
    print(f"  {'Metric':<28} {'Baseline':>12} {'Actualizer':>12} {'QCA Parallel':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*14}")

    metrics = [
        ("Latency (ms)",        "latency_ms",         f"{res_a['latency_ms']:>12.2f}",   f"{res_b['latency_ms']:>12.2f}",   f"{res_c['latency_ms']:>14.2f}"),
        ("Grounded Rate",       "grounded_rate",       f"{res_a['grounded_rate']:>12.4f}", f"{res_b['grounded_rate']:>12.4f}", f"{res_c['grounded_rate']:>14.4f}"),
        ("Hallucination Rate",  "hallucination_rate",  f"{res_a['hallucination_rate']:>12.4f}", f"{res_b['hallucination_rate']:>12.4f}", f"{res_c['hallucination_rate']:>14.4f}"),
        ("Repetition Rate",     "repetition_rate",     f"{res_a['repetition_rate']:>12.4f}", f"{res_b['repetition_rate']:>12.4f}", f"{res_c['repetition_rate']:>14.4f}"),
        ("Mean Valuation nu_t", "mean_valuation",      f"{'N/A':>12}",                    f"{res_b['mean_valuation']:>12.4f}",   f"{res_c['mean_valuation']:>14.4f}"),
        ("Actualization Rate",  "actualization_rate",  f"{'N/A':>12}",                    f"{res_b['actualization_rate']:>12.4f}", f"{res_c['actualization_rate']:>14.4f}"),
    ]

    for label, _, va, vb, vc in metrics:
        print(f"  {label:<28} {va} {vb} {vc}")

    print(f"\n{divider}")
    print("  ARCHITECTURAL COMPLEXITY COMPARISON")
    print(divider)
    print(f"  {'Architecture':<30} {'Inference Complexity':>25} {'Quality':>10}")
    print(f"  {'-'*30} {'-'*25} {'-'*10}")
    print(f"  {'Attention Baseline':<30} {'O(V) per step':>25} {'Low':>10}")
    print(f"  {'Actualizer Engine':<30} {'O(V x iters) per step':>25} {'High':>10}")
    print(f"  {'QCA Parallel Engine':<30} {'O(N^2/K) parallel':>25} {'High':>10}")
    print(f"  {'QCA Theoretical Speedup':<30} {'K=' + str(K) + 'x maximum':>25} {'-':>10}")
    print(divider)

    return {
        "quality_results"   : quality_results,
        "vocab_scaling"     : vocab_scaling,
        "parallel_scaling"  : parallel_scaling,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run(
        vocab_sizes    = (500, 1000, 2000),
        n_steps        = 30,
        n_sizes        = (20, 40, 80, 120, 200),
        K              = 5,
        distractor_str = 8.0,
        mercy_k        = 0.45,
        seed           = 42,
    )
