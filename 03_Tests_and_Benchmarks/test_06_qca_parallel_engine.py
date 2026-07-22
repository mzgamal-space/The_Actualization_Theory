"""
test_06_qca_parallel_engine.py — QCA Parallel Engine Benchmark
===================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153

Benchmarks the QCAParallelEngine against single-dataset sequential execution.

Theory (CKT White Paper v3, §7.2 Theorem 2 Corollary):
  Partitioning N nodes into K parallel clusters reduces steering & search
  complexity from O(N²) down to K · O((N/K)²) = O(N²/K).
  This test empirically measures:
    1. Sequential vs Parallel execution time (ms)
    2. Speedup factor (S = T_seq / T_par)
    3. Break-down of QCA clustering, parallel execution, and final synthesis times
    4. Solution quality & valuation trace consistency (ν_t)

Returns a dict of results for use by generate_all_charts.py (Figure 6).
"""

from __future__ import annotations

import sys
import os
import time
import random
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent folders to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from qca import QCANode
from qca_parallel_engine import QCAParallelEngine


def generate_synthetic_dataset(N: int, dim: int = 5, seed: int = 42) -> List[QCANode]:
    """Generate reproducible problem dataset of N QCANodes."""
    rng = random.Random(seed)
    nodes = []
    for i in range(N):
        coords = [rng.uniform(0.0, 10.0) for _ in range(dim)]
        prime_profile = [rng.uniform(0.1, 0.9) for _ in range(5)]
        # Normalize prime profile
        s = sum(prime_profile) or 1.0
        prime_profile = [p / s for p in prime_profile]
        nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime_profile))
    return nodes


def run(
    n_sizes: tuple = (20, 40, 80, 120, 200),
    K: int = 5,
    vocab_size: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full speedup & scaling benchmark for QCA Parallel Engine.
    """
    print(f"[test_06_qca_parallel_engine] Running benchmark for N={n_sizes}, K={K}, V={vocab_size}...")

    results = {
        "n_sizes"           : list(n_sizes),
        "K"                 : K,
        "vocab_size"        : vocab_size,
        "sequential_ms"     : [],
        "parallel_ms"       : [],
        "speedup"           : [],
        "qca_ms"            : [],
        "parallel_worker_ms": [],
        "synthesis_ms"      : [],
        "valuations_par"    : [],
        "actualized_rates"  : [],
    }

    for N in n_sizes:
        dataset = generate_synthetic_dataset(N=N, seed=seed + N)
        engine = QCAParallelEngine(
            K=K,
            vocab_size=vocab_size,
            mercy_k=0.45,
            seed=seed,
        )

        # 1. Parallel execution
        res_par = engine.process_parallel(dataset, verbose=False)

        # 2. Sequential baseline execution
        t_seq = engine.process_sequential(dataset)

        t_par = res_par.total_time_ms
        speedup = t_seq / t_par if t_par > 0 else 1.0
        res_par.speedup_vs_sequential = speedup

        results["sequential_ms"].append(round(t_seq, 2))
        results["parallel_ms"].append(round(t_par, 2))
        results["speedup"].append(round(speedup, 2))
        results["qca_ms"].append(round(res_par.qca_time_ms, 2))
        results["parallel_worker_ms"].append(round(res_par.parallel_time_ms, 2))
        results["synthesis_ms"].append(round(res_par.synthesis_time_ms, 2))
        results["valuations_par"].append(round(res_par.global_valuation, 4))

        # Calculate percentage of sub-clusters successfully actualized
        tot_act = sum(c.actualized_count for c in res_par.cluster_results)
        act_rate = (tot_act / N) * 100.0 if N > 0 else 100.0
        results["actualized_rates"].append(round(act_rate, 2))

        print(
            f"  N={N:>4} | Seq: {t_seq:>7.2f} ms | Par: {t_par:>7.2f} ms | "
            f"Speedup: {speedup:>5.2f}× | QCA: {res_par.qca_time_ms:>5.2f} ms | "
            f"Val: {res_par.global_valuation:.4f}"
        )

    return results


if __name__ == "__main__":
    r = run()
    print("\n-- QCA Parallel Engine Benchmark Summary --")
    for i, N in enumerate(r["n_sizes"]):
        print(
            f"  N={N:>4}: Sequential={r['sequential_ms'][i]} ms, "
            f"Parallel={r['parallel_ms'][i]} ms, "
            f"Speedup={r['speedup'][i]}x"
        )
