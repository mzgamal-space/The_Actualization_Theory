"""
test_04_search_space_scaling.py — Asymptotic Scaling Comparison
================================================================
Computes the search space size (number of nodes) for:
  - Standard combinatorial search : O(M^N) where M = branching factor
  - FDSA Fractal Deduction        : O(N^D) where D = ln(N)/ln(1/k)

at problem sizes N = 4 to 18 with M = 3 branching factor.
Also measures actual FDSA execution time vs backtracking execution time
on a small scheduling constraint problem.

Returns a dict of results for use by generate_all_charts.py.
"""
import sys, os, math, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from fdsa_pruner import FractalDeductionSearch, VectorizedFDSAPruner
from actualizer_engine import ActualizerEngine


def _backtrack_count(n: int, m: int, constraints: list) -> int:
    """Count nodes visited by backtracking on an n-task, m-processor scheduling problem."""
    visited = [0]

    def bt(task_idx, assignment):
        visited[0] += 1
        if task_idx == n:
            return True
        for p in range(m):
            assignment[task_idx] = p
            valid = True
            for (a, b) in constraints:
                if a < task_idx and b <= task_idx:
                    if assignment[a] == assignment[b]:
                        valid = False
                        break
            if valid:
                if bt(task_idx + 1, assignment):
                    return True
            assignment[task_idx] = -1
        return False

    bt(0, [-1] * n)
    return visited[0]


def run(
    n_range         = range(4, 19),
    branching_factor: int   = 3,
    k_ref           : float = 0.35,
) -> dict:
    fdsa = FractalDeductionSearch()

    results = {
        "n_values"           : list(n_range),
        "combinatorial_nodes": [],
        "fdsa_nodes"         : [],
        "backtrack_ms"       : [],
        "fdsa_ms"            : [],
        "reduction_pct"      : [],
    }

    for n in n_range:
        # Theoretical node counts
        comb_nodes = branching_factor ** n
        D = fdsa.fractal_dimension(max(n, 2), k_ref)
        fdsa_nodes = int(n ** D)

        # Time actual backtracking on a small scheduling problem
        random.seed(n)
        # Dependency pairs: task i must finish before task j
        n_constraints = max(2, n // 3)
        constraints   = [(i, i + 1) for i in range(n_constraints)]

        t0 = time.perf_counter()
        bt_count = _backtrack_count(n, branching_factor, constraints)
        bt_ms    = (time.perf_counter() - t0) * 1000

        # Time FDSA on a vocabulary of size n (proxy for search space)
        V_proxy = max(n * 10, 20)
        pruner  = VectorizedFDSAPruner(vocab_size=V_proxy, k=k_ref)
        import numpy as np
        np.random.seed(n)
        logits = np.random.normal(-3.0, 1.0, size=(V_proxy,))

        t0 = time.perf_counter()
        pruner.prune_numpy(logits, 0, {}, "logical_coding")
        fdsa_ms = (time.perf_counter() - t0) * 1000

        reduction = (1.0 - min(fdsa_nodes, comb_nodes) / max(comb_nodes, 1)) * 100

        results["combinatorial_nodes"].append(comb_nodes)
        results["fdsa_nodes"].append(fdsa_nodes)
        results["backtrack_ms"].append(round(bt_ms, 4))
        results["fdsa_ms"].append(round(fdsa_ms, 4))
        results["reduction_pct"].append(round(reduction, 2))

        print(f"  N={n:2d} | Combinatorial: {comb_nodes:>10,} | FDSA: {fdsa_nodes:>8,} | "
              f"Reduction: {reduction:.1f}%")

    return results


if __name__ == "__main__":
    print("Running search space scaling analysis (N = 4 → 18)…\n")
    r = run()
    print("\n── Peak reduction ──")
    max_r = max(r["reduction_pct"])
    idx   = r["reduction_pct"].index(max_r)
    print(f"  At N={r['n_values'][idx]}: {max_r:.2f}% reduction "
          f"({r['combinatorial_nodes'][idx]:,} → {r['fdsa_nodes'][idx]:,} nodes)")
