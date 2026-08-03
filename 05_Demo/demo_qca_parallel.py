"""
demo_qca_parallel.py — QCA Parallel Engine & Pipeline Demonstration
=====================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         Independent Researcher | ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com
Pipeline: Actualizer_Engine_FDSA_QCA v1.0.0 (CKT V3_U1)

Demonstrates the QCA Parallel Engine and Unified Parallel Pipeline:
  Phase 1 — QCA Crystallization Front-End:
            Partitions a large problem dataset (N nodes) into K independent
            crystallization clusters using Quench Temperature T_q^RGG.
  Phase 2 — Parallel Cluster Execution:
            Dispatches clusters to K parallel workers (ThreadPool or Processes).
            Each worker executes FDSA vocabulary pruning and ActualizerEngine
            contractive steering.
  Phase 3 — Global Synthesis:
            Aggregates cluster actualized states and performs final FDSA + Actualizer
            pass to yield the globally unified solution S*.
  Phase 4 — Unified Parallel Pipeline Execution:
            Runs ActualizerFDSAQCAPipeline in parallel mode across autoregressive steps.

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
import numpy as np

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent folders to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_DIR = os.path.dirname(BASE_DIR)
PIPELINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "Actualizer_Engine_FDSA_QCA")
CORE_ENGINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "02_Core_Engine")

sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, CORE_ENGINE_DIR)

from qca import QCANode
from qca_parallel_engine import QCAParallelEngine
from pipeline import create_parallel_pipeline


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
    time.sleep(0.1)

    # Config
    VOCAB_SIZE = 1000
    K = 5
    N = 120

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

    # Initialize Engine (processes / auto backend)
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
    tag("WORKERS", C.BLUE, f"Dispatching {K} clusters across parallel worker threads/processes...")

    t0_par = time.perf_counter()
    par_res = engine.process_parallel(nodes, verbose=False)
    t1_par = time.perf_counter()

    for c_res in par_res.cluster_results[:5]:
        act_str = f"{C.GREEN}{c_res.actualized_count}/{len(c_res.node_ids)} Actualized{C.END}"
        tag("WORKER", C.GREEN, f"Cluster {c_res.cluster_id:^3} complete in {c_res.worker_time_ms:.2f} ms | {act_str} | Mean Valuation ν = {c_res.mean_valuation:.4f} | Mean Drift Tr(D) = {c_res.mean_drift:.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Global Synthesis & Pipeline Integration
    # -----------------------------------------------------------------------
    section("PHASE 3 — GLOBAL SYNTHESIS & UNIFIED PIPELINE INTEGRATION")
    tag("SYNTHESIS", C.YELLOW, f"Collected {len(par_res.cluster_results)} cluster results into metacluster substrate.")
    tag("SYNTHESIS", C.GREEN, f"Final Actualized Global Token S* = {C.BOLD}{par_res.final_token}{C.END}")
    tag("SYNTHESIS", C.GREEN, f"Global Valuation ν_final = {C.BOLD}{par_res.global_valuation:.4f}{C.END}")
    tag("SYNTHESIS", C.GREEN, f"Global Trace Drift Tr(D_μν) = {par_res.global_drift:.4f} (Actualized: {'True' if par_res.is_actualized else 'False'})")

    print()
    tag("UNIFIED PIPELINE", C.CYAN, "Executing unified ActualizerFDSAQCAPipeline (ThreadPool Backend)...")
    pipe = create_parallel_pipeline(vocab_size=VOCAB_SIZE, K=K, seed=42, verbose=False)
    raw_logits = np.random.normal(-4.0, 1.5, size=(VOCAB_SIZE,))
    p_res = pipe.run(context_ids=[10, 20, 30], logits=raw_logits, step=0)

    tag("PIPELINE", C.GREEN,
        f"Unified Parallel Pipeline: Selected Token = {p_res.final_token} | "
        f"Stage 2 Parallel Latency = {p_res.parallel_result.parallel_time_ms:.2f} ms | "
        f"Total End-to-End Latency = {p_res.total_time_ms:.2f} ms")

    pipe.shutdown()

    # -----------------------------------------------------------------------
    # Step 4: Summary
    # -----------------------------------------------------------------------
    section("PHASE 4 — LATENCY & EXECUTION SUMMARY")
    print(f"""
  +--------------------------------------------------------------------+
  |  Metric                             QCA Parallel Engine            |
  |  ------------------------------------------------------------------|
  |  Problem Size N                     {N:<30} |
  |  Clusters K                         {K:<30} |
  |  Vocabulary Size V                  {VOCAB_SIZE:<30,} |
  |  Quench Temperature T_q             {qca_res.quench_temp:<30.6f} |
  |  QCA Clustering Time                {par_res.qca_time_ms:<30.2f} ms |
  |  Parallel Worker Processing Time    {par_res.parallel_time_ms:<30.2f} ms |
  |  Global Synthesis Pass              {par_res.synthesis_time_ms:<30.2f} ms |
  |  Total Engine Latency               {par_res.total_time_ms:<30.2f} ms |
  +--------------------------------------------------------------------+
""")


if __name__ == "__main__":
    run_demo()
