"""
demo_qca_parallel.py — QCA Parallel Engine Interactive Demonstration
=====================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         Independent Researcher | ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com

Demonstrates the QCA Parallel Engine pipeline:
  Phase 1 — QCA Crystallization Front-End:
            Partitions a large problem dataset (N nodes) into K independent
            crystallization clusters using Quench Temperature T_q^RGG.
  Phase 2 — Parallel Cluster Execution:
            Dispatches clusters to K parallel worker processes. Each process
            independently executes FDSA vocabulary pruning and ActualizerEngine
            contractive steering.
  Phase 3 — Global Synthesis:
            Aggregates cluster actualized states and performs final FDSA + Actualizer
            pass to yield the globally unified solution S*.
  Phase 4 — Empirical Speedup Benchmark:
            Compares QCA Parallel execution against single-dataset sequential execution.

HOW TO RUN
----------
  cd Final_Output/05_Demo
  python demo_qca_parallel.py
"""

from __future__ import annotations

import sys
import os
import time
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent folders to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from qca import QCANode
from qca_parallel_engine import QCAParallelEngine


# ---------------------------------------------------------------------------
# Console Styling Helpers
# ---------------------------------------------------------------------------
class C:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    END    = '\033[0m'


def hr(char='─', n=74, color=C.DIM):
    print(f"{color}{char * n}{C.END}")


def tag(label, color, text):
    print(f"  {color}{C.BOLD}[{label}]{C.END} {text}")


def section(title):
    print()
    hr('=')
    print(f"{C.HEADER}{C.BOLD}  {title}{C.END}")
    hr('=')


# ---------------------------------------------------------------------------
# Main Demo Routine
# ---------------------------------------------------------------------------
def run_demo():
    print()
    print(f"{C.BOLD}{C.CYAN}")
    print("  ╔══════════════════════════════════════════════════════════════════════════╗")
    print("  ║    QCA PARALLEL ENGINE — HIGH-PERFORMANCE CLUSTERED INFERENCE DEMO      ║")
    print("  ║    Mohamed Gamal Eldin · ORCID 0009-0006-3991-1153                       ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════╝")
    print(C.END)
    time.sleep(0.2)

    # Config
    VOCAB_SIZE = 1000
    K = 5
    N = 200

    section("CONFIG & INPUT DATASET GENERATION")
    tag("SETUP", C.BLUE, f"Problem Dataset Size N = {N} nodes")
    tag("SETUP", C.BLUE, f"Target QCA Clusters K = {K}")
    tag("SETUP", C.BLUE, f"Substrate Vocabulary Size V = {VOCAB_SIZE:,}")
    tag("THEORY", C.YELLOW, f"Theoretical Complexity: Sequential O(N²) = {N**2:,} ops | QCA Parallel O(N²/K) = {int(N**2/K):,} ops ({K}.0× reduction)")

    # Generate synthetic nodes
    rng = random.Random(42)
    nodes = []
    for i in range(N):
        coords = [rng.uniform(0.0, 10.0) for _ in range(5)]
        prime_prof = [rng.uniform(0.1, 0.9) for _ in range(5)]
        s = sum(prime_prof) or 1.0
        prime_prof = [p / s for p in prime_prof]
        nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime_prof))

    tag("DATASET", C.GREEN, f"Created {len(nodes)} QCANodes with 5D spatial embeddings & 5-Prime profiles.")

    # Initialize Engine
    engine = QCAParallelEngine(
        K=K,
        vocab_size=VOCAB_SIZE,
        mercy_k=0.45,
        Q_c=1e-5,
        context_type="logical_coding",
        seed=42,
    )

    # -----------------------------------------------------------------------
    # Step 1: QCA Clustering
    # -----------------------------------------------------------------------
    section("PHASE 1 — QCA CRYSTALLIZATION (CLUSTERING)")
    t0 = time.perf_counter()
    qca_res = engine.qca.run(nodes)
    t1 = time.perf_counter()
    qca_time = (t1 - t0) * 1000.0

    tag("QCA", C.CYAN, f"Canonical Quench Temperature T_q^RGG = {qca_res.quench_temp:.6f}")
    tag("QCA", C.CYAN, f"Distance Matrix built & {len(qca_res.clusters)} crystallization clusters formed in {qca_time:.2f} ms:")
    for cluster in qca_res.clusters[:5]:
        c_p = ", ".join(f"{p:.2f}" for p in cluster.prime_profile)
        print(f"    • {C.BOLD}Cluster {cluster.cluster_id:^3}{C.END}: {len(cluster.nodes):>3} nodes | Prime Profile: [{c_p}]")
    if len(qca_res.clusters) > 5:
        print(f"    • ... {len(qca_res.clusters) - 5} additional crystallization clusters formed.")

    # -----------------------------------------------------------------------
    # Step 2: Parallel Cluster Execution
    # -----------------------------------------------------------------------
    section("PHASE 2 — PARALLEL CLUSTER EXECUTION (FDSA + ACTUALIZER WORKERS)")
    tag("WORKERS", C.BLUE, f"Dispatching {K} clusters across parallel worker processes...")

    t0_par = time.perf_counter()
    par_res = engine.process_parallel(nodes, verbose=False)
    t1_par = time.perf_counter()
    par_time = (t1_par - t0_par) * 1000.0

    for c_res in par_res.cluster_results[:5]:
        act_str = f"{C.GREEN}{c_res.actualized_count}/{len(c_res.node_ids)} Actualized{C.END}"
        tag("WORKER", C.GREEN, f"Cluster {c_res.cluster_id:^3} complete in {c_res.worker_time_ms:.2f} ms | {act_str} | Mean Valuation ν = {c_res.mean_valuation:.4f} | Mean Drift Tr(D) = {c_res.mean_drift:.4f}")
    if len(par_res.cluster_results) > 5:
        tag("WORKERS", C.GREEN, f"All {len(par_res.cluster_results)} parallel clusters successfully processed & actualized.")

    # -----------------------------------------------------------------------
    # Step 3: Global Synthesis
    # -----------------------------------------------------------------------
    section("PHASE 3 — GLOBAL SYNTHESIS (FINAL FDSA & ACTUALIZER PASS)")
    tag("SYNTHESIS", C.YELLOW, f"Collected {len(par_res.cluster_results)} cluster results into metacluster substrate.")
    tag("SYNTHESIS", C.GREEN, f"Final Actualized Global Token S* = {C.BOLD}{par_res.final_token}{C.END}")
    tag("SYNTHESIS", C.GREEN, f"Global Valuation ν_final = {C.BOLD}{par_res.global_valuation:.4f}{C.END}")
    tag("SYNTHESIS", C.GREEN, f"Global Trace Drift Tr(D_μν) = {par_res.global_drift:.4f} (Bifurcation Gated: {'True' if par_res.is_actualized else 'False'})")

    # -----------------------------------------------------------------------
    # Step 4: Speedup Comparison
    # -----------------------------------------------------------------------
    section("PHASE 4 — EMPIRICAL SPEEDUP BENCHMARK")
    tag("BENCHMARK", C.BLUE, "Running single-dataset sequential baseline for comparison...")

    t0_seq = time.perf_counter()
    t_seq = engine.process_sequential(nodes)
    t1_seq = time.perf_counter()

    speedup = t_seq / par_res.total_time_ms if par_res.total_time_ms > 0 else 1.0

    print()
    hr('─')
    print(f"  {C.BOLD}LATENCY & SPEEDUP SUMMARY (N={N}, K={K}, V={VOCAB_SIZE:,}){C.END}")
    hr('─')
    print(f"  Sequential Baseline Processing Time : {C.RED}{t_seq:>8.2f} ms{C.END}")
    print(f"  QCA Parallel Engine Total Time      : {C.GREEN}{par_res.total_time_ms:>8.2f} ms{C.END}")
    print(f"    ├─ QCA Clustering Front-End       : {par_res.qca_time_ms:>8.2f} ms")
    print(f"    ├─ Parallel Worker Execution      : {par_res.parallel_time_ms:>8.2f} ms")
    print(f"    └─ Global Synthesis Pass          : {par_res.synthesis_time_ms:>8.2f} ms")
    print(f"  {C.BOLD}NET PARALLEL SPEEDUP FACTOR         : {C.CYAN}{C.BOLD}{speedup:>8.2f}× faster{C.END}")
    hr('─')
    print()


if __name__ == "__main__":
    run_demo()
