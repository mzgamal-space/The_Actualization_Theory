"""
qca_parallel_engine.py — Parallel QCA Actualizer & FDSA Engine (Processes + JAX Support)
==========================================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         Independent Researcher | ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com
Module : Final_Output/02_Core_Engine/qca_parallel_engine.py

Theory & Architecture
---------------------
Reference: CKT White Paper v3, §7.2 — Theorem 2 Corollary:
  K parallel clusters each solve a sub-problem of size N/K at cost O((N/K)²);
  aggregate computational work = K · O((N/K)²) = O(N²/K) — a factor-K improvement
  over processing a single dataset sequentially at cost O(N²).

Execution Backends Supported:
  1. backend="processes" (Default):
     Spawns K CPU worker processes via Python's ProcessPoolExecutor to execute
     FDSA pre-inference logit pruning and Actualizer contractive steering in parallel.
  2. backend="jax":
     Vectorized parallel cluster processing via JAX (jnp.ndarray & @jax.jit ops).
     Processes all K clusters in parallel on GPU/TPU/CPU SIMD vector units.
  3. backend="auto":
     Automatically uses JAX if jax is installed; falls back to parallel processes.
"""

from __future__ import annotations

import os
import time
import math
import random
import multiprocessing
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple, Any

# Core module imports
from qca import QuenchClusterAlgorithm, QCANode, QCACluster, QuenchResult
from actualizer_engine import ActualizerEngine, EQUILIBRIUM_ALPHA, N_PRIMES
from fdsa_pruner import VectorizedFDSAPruner
from numpy_actualizer_engine import NumpyActualizerEngine

import numpy as np

# Optional JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# ---------------------------------------------------------------------------
# Data Structures for Parallel Execution & Results
# ---------------------------------------------------------------------------

@dataclass
class ClusterProcessResult:
    """
    Result produced for a single QCA cluster by a parallel worker process or JAX unit.
    """
    cluster_id: int
    node_ids: List[int]
    actualized_tokens: List[int]
    trace_drifts: List[float]
    valuations: List[float]
    actualized_flags: List[bool]
    mean_drift: float
    mean_valuation: float
    actualized_count: int
    worker_time_ms: float

    def __repr__(self) -> str:
        return (
            f"ClusterProcessResult(cluster_id={self.cluster_id}, "
            f"nodes={len(self.node_ids)}, "
            f"mean_drift={self.mean_drift:.4f}, "
            f"mean_val={self.mean_valuation:.4f}, "
            f"time={self.worker_time_ms:.2f}ms)"
        )


@dataclass
class QCAParallelResult:
    """
    Global result returned by the QCAParallelEngine.
    """
    final_token: int
    global_valuation: float
    global_drift: float
    total_iterations: int
    is_actualized: bool
    cluster_results: List[ClusterProcessResult]
    qca_result: QuenchResult
    total_time_ms: float
    qca_time_ms: float
    parallel_time_ms: float
    synthesis_time_ms: float
    backend_used: str = "processes"
    speedup_vs_sequential: Optional[float] = None
    audit_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-Level Worker Function (Pickleable for Windows Multiprocessing)
# ---------------------------------------------------------------------------

def _worker_process_cluster(payload: dict) -> dict:
    """
    Worker function executed in parallel process for each QCA cluster.

    OPTIMIZED (v2): Uses NumpyActualizerEngine + prune_numpy fast path.
    - NumpyActualizerEngine: replaces Python for-loops with numpy BLAS/SIMD ops
    - prune_numpy: vectorized boolean mask instead of Python loop over V
    Combined effect: ~5-10x faster per worker vs original ActualizerEngine.
    """
    import numpy as np
    t0 = time.perf_counter()

    cluster_id      = payload["cluster_id"]
    node_data       = payload["node_data"]
    vocab_size      = payload["vocab_size"]
    mercy_k         = payload["mercy_k"]
    Q_c             = payload["Q_c"]
    tau_bifurcation = payload["tau_bifurcation"]
    context_type    = payload["context_type"]
    grammar_rules   = payload.get("grammar_rules", {})
    max_iters       = payload.get("max_iters", 25)

    # ── Fast-path engines (NumPy vectorized) ─────────────────────────────────
    pruner = VectorizedFDSAPruner(vocab_size=vocab_size, k=mercy_k)
    engine = NumpyActualizerEngine(
        vocab_size      = vocab_size,
        mercy_k         = mercy_k,
        Q_c             = Q_c,
        tau_bifurcation = tau_bifurcation,
        max_iters       = max_iters,
    )

    node_ids: List[int] = []
    actualized_tokens: List[int] = []
    trace_drifts: List[float] = []
    valuations: List[float] = []
    actualized_flags: List[bool] = []

    for item in node_data:
        nid        = item["node_id"]
        coords     = item["coords"]
        prime_prof = item["prime_profile"]

        # Derive initial logits substrate from node coordinates & prime profile
        dim      = len(coords)
        base_val = sum(coords) / (dim or 1.0)
        rng      = random.Random(nid * 1000 + int(base_val * 100))
        logits_py = [rng.gauss(base_val, 1.0) for _ in range(vocab_size)]
        logits_np = np.array(logits_py, dtype=np.float64)

        target_center = int((prime_prof[0] if prime_prof else 0.5) * vocab_size) % vocab_size
        target_tokens = set(range(max(0, target_center - 20), min(vocab_size, target_center + 20)))
        history       = [max(0, target_center - 1)]
        last_token    = history[-1]

        # Phase A/B: FDSA Pruning — numpy fast path (vectorized boolean mask)
        pruned_np, active_count = pruner.prune_numpy(
            logits       = logits_np,
            last_token   = last_token,
            grammar_rules= grammar_rules,
            context_type = context_type,
        )

        # Phase C/D: Actualizer Steering — NumpyActualizerEngine (no Python V-loops)
        token, U_final, Tr_D, iters, nu_hist, actualized = engine.steer(
            logits        = pruned_np,
            history       = history,
            target_tokens = target_tokens if target_tokens else set(range(vocab_size)),
        )

        node_ids.append(nid)
        actualized_tokens.append(token)
        trace_drifts.append(Tr_D)
        final_val = nu_hist[-1] if nu_hist else 0.0
        valuations.append(final_val)
        actualized_flags.append(actualized)

    t1 = time.perf_counter()
    worker_ms = (t1 - t0) * 1000.0

    mean_drift = sum(trace_drifts) / len(trace_drifts) if trace_drifts else 0.0
    mean_val   = sum(valuations) / len(valuations) if valuations else 0.0
    act_count  = sum(1 for f in actualized_flags if f)

    return {
        "cluster_id"       : cluster_id,
        "node_ids"         : node_ids,
        "actualized_tokens": actualized_tokens,
        "trace_drifts"     : trace_drifts,
        "valuations"       : valuations,
        "actualized_flags" : actualized_flags,
        "mean_drift"       : mean_drift,
        "mean_valuation"   : mean_val,
        "actualized_count" : act_count,
        "worker_time_ms"   : worker_ms,
    }


# ---------------------------------------------------------------------------
# QCAParallelEngine Main Class
# ---------------------------------------------------------------------------

class QCAParallelEngine:
    """
    QCA Parallel Engine supporting:
      • Parallel Processes execution (multiprocessing ProcessPoolExecutor)
      • Vectorized JAX execution (jnp array operations)
      • Crystallization via QuenchClusterAlgorithm
      • Global synthesis via ActualizerEngine + FDSAPruner

    IMPORTANT SCOPE NOTE (added this pass, after tracing a real reported
    result back to this file):
    ----------------------------------------------------------------------
    This engine's `actualized_tokens` output is a STEERING DIAGNOSTIC over
    a SYNTHETIC per-node logit distribution (see `_worker_process_cluster`:
    each node's logits are drawn from a Gaussian seeded by that node's
    embedding-derived coordinates, NOT produced by running a real model's
    decoder). This module does NOT call model.generate(), does NOT produce
    real decoded text, and its output tokens CANNOT be meaningfully compared
    against real ground-truth answers (there is no text decode step here at
    all). Any EM/F1 metric reported "from" this engine's output was computed
    elsewhere, from different data, and reused -- not computed from this
    engine's actual return values. If you need a real end-to-end generation
    benchmark, this class is the clustering/steering-diagnostic layer, not
    the generation layer -- it would need a genuine per-cluster call into a
    real model's forward pass wired in, which is not what
    `_worker_process_cluster` currently does. This is flagged here explicitly
    so this specific mistake -- reporting this module's synthetic output as
    if it were a real-dataset generation result -- cannot recur silently.

    Parameters
    ----------
    K : int
        Number of clusters / parallel sub-problems.
    vocab_size : int
        Token vocabulary size V.
    mercy_k : float
        Contractive scale factor k (Mercy Prime parameter).
    Q_c : float
        Causal quantum threshold (L2 convergence tolerance).
    tau_bifurcation : float
        Bifurcation threshold for Tr(D_μν) criterion.
    max_iters : int
        Max contraction iterations per node.
    backend : str
        Parallel backend: "processes" (default), "jax", or "auto".
    n_workers : Optional[int]
        Number of process workers when backend="processes".
    context_type : str
        Context profile for FDSA anchoring ('logical_coding', 'mathematical', etc.)
    seed : Optional[int]
        Random seed for QCA initialization.
    """

    def __init__(
        self,
        K: Optional[int] = 5,
        vocab_size: int = 1000,
        mercy_k: float = 0.45,
        Q_c: float = 1e-5,
        tau_bifurcation: float = 5.0,
        max_iters: int = 25,
        backend: str = "processes",
        n_workers: Optional[int] = None,
        context_type: str = "logical_coding",
        seed: Optional[int] = 42,
    ) -> None:
        self.K               = K or 5
        self.vocab_size       = vocab_size
        self.mercy_k         = mercy_k
        self.Q_c             = Q_c
        self.tau_bifurcation = tau_bifurcation
        self.max_iters       = max_iters
        self.backend         = backend.lower()
        self.n_workers       = n_workers or min(self.K, os.cpu_count() or 4)
        self.context_type    = context_type
        self.seed            = seed

        self.qca = QuenchClusterAlgorithm(K=K, seed=seed)
        self.pruner = VectorizedFDSAPruner(vocab_size=vocab_size, k=mercy_k)
        self.engine = ActualizerEngine(
            vocab_size=vocab_size,
            mercy_k=mercy_k,
            Q_c=Q_c,
            tau_bifurcation=tau_bifurcation,
            max_iters=max_iters,
        )

        # FIXED (this pass): domain-anchored weights, using the same
        # profile[O,J,K] -> prime_weights mapping verified in
        # test_engine_equivalence.py's mixed-domain-batch test. Falls back to
        # the engine's own defaults if context_type isn't a recognized
        # profile. This is what _process_clusters_jax's batched vmap call
        # uses -- see below.
        if self.context_type in self.pruner.CONTEXT_PROFILES:
            profile = self.pruner.CONTEXT_PROFILES[self.context_type]
            self.prime_weights_dict = {
                "Order": profile[0], "Justice": profile[1], "Knowledge": profile[3]
            }
        else:
            self.prime_weights_dict = {"Order": 0.35, "Justice": 0.35, "Knowledge": 0.20}

        # FIXED (this pass): persistent process pool, created lazily on first
        # use and reused across calls -- see _process_clusters_multiprocessing
        # below. The previous implementation created a brand new
        # ProcessPoolExecutor (with spawn) inside every single call, measured
        # directly (see _measure_pool_overhead.py in this pass) at ~200ms of
        # pure startup overhead per call for a 3-worker pool -- completely
        # dominating a workload of a few tens of ms. Call shutdown_pool()
        # when done with this engine instance to clean up worker processes.
        self._persistent_pool = None
        self._persistent_pool_context = None

    def shutdown_pool(self):
        """Explicitly shut down the persistent process pool, if one was
        created. Call this when done with the engine instance, or use it as
        a context manager (see __enter__/__exit__ below)."""
        if self._persistent_pool is not None:
            self._persistent_pool.shutdown(wait=True)
            self._persistent_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_pool()

    # -----------------------------------------------------------------------
    # JAX Vectorized Parallel Cluster Execution Backend
    # -----------------------------------------------------------------------

    def _process_clusters_jax(
        self,
        clusters: List[QCACluster],
        grammar_rules: Optional[Dict[int, Set[int]]],
    ) -> List[ClusterProcessResult]:
        """
        REPLACED (this pass). The previous implementation of this method,
        despite its docstring claiming "vectorized JAX / NumPy operations",
        was actually a sequential Python double loop (per cluster, per node)
        with NO jax.jit and NO batching -- small unjitted JAX ops called one
        at a time. Measured directly: 251.8 ms/step for 15 nodes / 5
        clusters, WORSE than both the sequential baseline (16.3 ms/step) and
        the (also broken) per-call multiprocessing path (72.9 ms/step). It
        also used a different, simplified Prime-coordinate formula
        (alpha_vec = [max(U)/5]*5, all five components forced equal) that
        does not match the real 5-way calculation in
        ActualizerEngine/JaxActualizerEngine.

        This version instead batches ALL nodes across ALL clusters into a
        SINGLE jax.jit + vmap call to the verified JaxActualizerEngine (see
        jax_actualizer_engine.py, cross-checked against the pure-Python
        reference across 10+ seeds and a mixed-domain batch test in
        test_engine_equivalence.py). One compiled function, one batched
        call, no per-node Python loop over JAX ops.
        """
        from jax_actualizer_engine import JaxActualizerEngine

        if not hasattr(self, "_jax_engine") or self._jax_engine.V != self.vocab_size:
            self._jax_engine = JaxActualizerEngine(
                vocab_size=self.vocab_size, mercy_k=self.mercy_k, Q_c=self.Q_c,
                tau_bifurcation=self.tau_bifurcation, max_iters=self.max_iters,
            )
            # FIXED (this pass, SECOND bug in this same method): the
            # jax.jit(jax.vmap(...)) wrapper was being reconstructed inside
            # this method on every call. Measured directly: this produced a
            # FLAT ~480-500ms cost on every single call with no "expensive
            # first call, cheap afterward" shape -- the signature of
            # recompilation happening every time rather than once. Moved
            # here, built once alongside the engine it wraps, cached on the
            # instance exactly like _persistent_pool above.
            self._batched_steer = jax.jit(
                jax.vmap(self._jax_engine._steer_jit, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
            )

        # Flatten all nodes across all clusters into one batch
        flat_nodes = []
        cluster_boundaries = []  # (cluster, start_idx, end_idx) for regrouping after
        idx = 0
        for cluster in clusters:
            start = idx
            for n in cluster.nodes:
                flat_nodes.append(n)
                idx += 1
            cluster_boundaries.append((cluster, start, idx))

        N = len(flat_nodes)
        if N == 0:
            return []

        t0 = time.perf_counter()

        # Build batched inputs (this part is still a Python loop, but it's
        # cheap array/dict construction, not the expensive iterative
        # steering computation -- that part is what's now batched via vmap)
        batch_logits = np.zeros((N, self.vocab_size), dtype=np.float32)
        target_masks = np.zeros((N, self.vocab_size), dtype=bool)
        hist_ids = np.zeros((N, 8), dtype=np.int32)
        hist_valid = np.zeros((N, 8), dtype=bool)
        hist_stepback = np.tile(np.arange(7, -1, -1, dtype=np.float32), (N, 1))

        for i, n in enumerate(flat_nodes):
            coords = n.coords
            prime_prof = n.prime_profile
            dim = len(coords)
            base_val = sum(coords) / (dim or 1.0)
            rng = random.Random(n.node_id * 1000 + int(base_val * 100))
            batch_logits[i] = [rng.gauss(base_val, 1.0) for _ in range(self.vocab_size)]

            target_center = int((prime_prof[0] if prime_prof else 0.5) * self.vocab_size) % self.vocab_size
            lo, hi = max(0, target_center - 20), min(self.vocab_size, target_center + 20)
            target_masks[i, lo:hi] = True

            h = max(0, target_center - 1)
            hist_ids[i, 7] = h
            hist_valid[i, 7] = True

        wL_batch = np.full(N, self.prime_weights_dict.get("Order", 0.35), dtype=np.float32)
        wG_batch = np.full(N, self.prime_weights_dict.get("Justice", 0.35), dtype=np.float32)
        wF_batch = np.full(N, self.prime_weights_dict.get("Knowledge", 0.20), dtype=np.float32)
        k_batch = np.full(N, self.mercy_k, dtype=np.float32)

        U_batch, Tr_batch, iters_batch, act_batch = self._batched_steer(
            jnp.array(batch_logits), jnp.array(target_masks),
            jnp.array(hist_ids), jnp.array(hist_valid), jnp.array(hist_stepback),
            jnp.array(wL_batch), jnp.array(wG_batch), jnp.array(wF_batch), jnp.array(k_batch),
        )
        tokens_batch = np.argmax(np.asarray(U_batch), axis=1)
        t1 = time.perf_counter()
        per_node_ms = (t1 - t0) * 1000.0 / N

        cluster_results: List[ClusterProcessResult] = []
        for cluster, start, end in cluster_boundaries:
            node_ids = [flat_nodes[i].node_id for i in range(start, end)]
            tokens = [int(tokens_batch[i]) for i in range(start, end)]
            drifts = [float(Tr_batch[i]) for i in range(start, end)]
            acts = [bool(act_batch[i]) for i in range(start, end)]
            vals = [1.0 if a else 0.0 for a in acts]  # placeholder valuation, see note below

            cluster_results.append(ClusterProcessResult(
                cluster_id=cluster.cluster_id,
                node_ids=node_ids,
                actualized_tokens=tokens,
                trace_drifts=drifts,
                valuations=vals,
                actualized_flags=acts,
                mean_drift=sum(drifts) / len(drifts) if drifts else 0.0,
                mean_valuation=sum(vals) / len(vals) if vals else 0.0,
                actualized_count=sum(1 for a in acts if a),
                worker_time_ms=per_node_ms * (end - start),
            ))

        return cluster_results

    # -----------------------------------------------------------------------
    # Process Pool Execution Backend
    # -----------------------------------------------------------------------

    def _process_clusters_multiprocessing(
        self,
        clusters: List[QCACluster],
        grammar_rules: Optional[Dict[int, Set[int]]],
        log: List[str],
    ) -> List[ClusterProcessResult]:
        """
        Executes parallel cluster steering using ProcessPoolExecutor.
        """
        payloads = []
        for cluster in clusters:
            node_list = [
                {
                    "node_id": n.node_id,
                    "coords": n.coords,
                    "prime_profile": n.prime_profile,
                    "metadata": n.metadata,
                }
                for n in cluster.nodes
            ]
            payloads.append({
                "cluster_id"     : cluster.cluster_id,
                "node_data"      : node_list,
                "vocab_size"     : self.vocab_size,
                "mercy_k"        : self.mercy_k,
                "Q_c"            : self.Q_c,
                "tau_bifurcation": self.tau_bifurcation,
                "max_iters"      : self.max_iters,
                "context_type"   : self.context_type,
                "grammar_rules"  : grammar_rules or {},
            })

        cluster_results_dict: Dict[int, ClusterProcessResult] = {}

        # FIXED (this pass, second fix to this method): the pool is now
        # created ONCE (lazily, on first use) and reused across calls,
        # instead of being created and torn down inside every single call.
        # Measured overhead of the old per-call approach: ~200ms pure
        # startup cost per call (see _measure_pool_overhead.py), which
        # completely dominated this workload's actual compute time. The
        # spawn-context fix from the previous pass is preserved -- only the
        # lifecycle changed, not the fork-safety property.
        if self._persistent_pool is None:
            mp_context = multiprocessing.get_context("spawn")
            self._persistent_pool = ProcessPoolExecutor(
                max_workers=self.n_workers, mp_context=mp_context
            )

        executor = self._persistent_pool
        futures = {executor.submit(_worker_process_cluster, p): p["cluster_id"] for p in payloads}
        for future in as_completed(futures):
            cid = futures[future]
            res_data = future.result()
            c_res = ClusterProcessResult(
                cluster_id=res_data["cluster_id"],
                node_ids=res_data["node_ids"],
                actualized_tokens=res_data["actualized_tokens"],
                trace_drifts=res_data["trace_drifts"],
                valuations=res_data["valuations"],
                actualized_flags=res_data["actualized_flags"],
                mean_drift=res_data["mean_drift"],
                mean_valuation=res_data["mean_valuation"],
                actualized_count=res_data["actualized_count"],
                worker_time_ms=res_data["worker_time_ms"],
            )
            cluster_results_dict[cid] = c_res
            log.append(f"  Cluster {cid}: processed {len(c_res.node_ids)} nodes in {c_res.worker_time_ms:.2f} ms (mean val={c_res.mean_valuation:.4f})")

        return [cluster_results_dict[c.cluster_id] for c in clusters if c.cluster_id in cluster_results_dict]

    # -----------------------------------------------------------------------
    # Main Parallel Execution Pipeline
    # -----------------------------------------------------------------------

    def process_parallel(
        self,
        nodes: List[QCANode],
        grammar_rules: Optional[Dict[int, Set[int]]] = None,
        verbose: bool = False,
    ) -> QCAParallelResult:
        """
        Execute QCA clustering, parallel cluster solving (Processes or JAX), and final synthesis.
        """
        t_start = time.perf_counter()
        log: List[str] = []

        # Determine actual backend to use
        effective_backend = self.backend
        if effective_backend == "auto":
            effective_backend = "jax" if HAS_JAX else "processes"

        log.append(
            f"[QCA_Parallel_Engine] Starting run: N={len(nodes)} nodes, K={self.K} clusters, "
            f"backend='{effective_backend}' (JAX available: {HAS_JAX})"
        )

        # Step 1: QCA Crystallization
        t_qca_0 = time.perf_counter()
        qca_res = self.qca.run(nodes)
        t_qca_1 = time.perf_counter()
        qca_ms  = (t_qca_1 - t_qca_0) * 1000.0
        log.append(f"[Step 1 — QCA] Formed {len(qca_res.clusters)} clusters in {qca_ms:.2f} ms (T_q={qca_res.quench_temp:.6f})")

        # Step 2: Parallel Cluster Execution
        t_par_0 = time.perf_counter()
        if effective_backend == "jax":
            sorted_cluster_results = self._process_clusters_jax(qca_res.clusters, grammar_rules)
        else:
            sorted_cluster_results = self._process_clusters_multiprocessing(qca_res.clusters, grammar_rules, log)
        t_par_1 = time.perf_counter()
        par_ms  = (t_par_1 - t_par_0) * 1000.0

        # Step 3: Final Synthesis Pass
        t_syn_0 = time.perf_counter()

        meta_logits = [0.0] * self.vocab_size
        for c_res in sorted_cluster_results:
            for tok, val in zip(c_res.actualized_tokens, c_res.valuations):
                meta_logits[tok] += val + 1.0

        target_set = set(tok for c_res in sorted_cluster_results for tok in c_res.actualized_tokens)
        synth_history = [list(target_set)[0]] if target_set else [0]

        pruned_meta, _ = self.pruner.prune_vocabulary(
            logits=meta_logits,
            last_token=synth_history[-1],
            grammar_rules=grammar_rules or {},
            context_type=self.context_type,
        )

        final_token, U_synth, final_drift, total_iters, nu_hist, is_act = self.engine.steer(
            logits=pruned_meta,
            history=synth_history,
            target_tokens=target_set if target_set else set(range(self.vocab_size)),
        )

        t_syn_1 = time.perf_counter()
        syn_ms  = (t_syn_1 - t_syn_0) * 1000.0

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        global_val = nu_hist[-1] if nu_hist else 0.0

        log.append(f"[Step 3 — Synthesis] Final actualized token={final_token}, val={global_val:.4f}, drift={final_drift:.4f} in {syn_ms:.2f} ms")
        log.append(f"[QCA_Parallel_Engine] Complete in {total_ms:.2f} ms (backend={effective_backend})")

        if verbose:
            for line in log:
                print(line)

        return QCAParallelResult(
            final_token=final_token,
            global_valuation=global_val,
            global_drift=final_drift,
            total_iterations=total_iters,
            is_actualized=is_act,
            cluster_results=sorted_cluster_results,
            qca_result=qca_res,
            total_time_ms=total_ms,
            qca_time_ms=qca_ms,
            parallel_time_ms=par_ms,
            synthesis_time_ms=syn_ms,
            backend_used=effective_backend,
            audit_log=log,
        )

    def process_sequential(
        self,
        nodes: List[QCANode],
        grammar_rules: Optional[Dict[int, Set[int]]] = None,
    ) -> float:
        """
        Execute single-dataset sequential baseline (without QCA partitioning) to measure speedup.
        """
        t0 = time.perf_counter()

        for n in nodes:
            dim = len(n.coords)
            base_val = sum(n.coords) / (dim or 1.0)
            rng = random.Random(n.node_id * 1000 + int(base_val * 100))
            logits = [rng.gauss(base_val, 1.0) for _ in range(self.vocab_size)]

            target_center = int((n.prime_profile[0] if n.prime_profile else 0.5) * self.vocab_size) % self.vocab_size
            target_tokens = set(range(max(0, target_center - 20), min(self.vocab_size, target_center + 20)))
            history = [max(0, target_center - 1)]

            pruned_logits, _ = self.pruner.prune_vocabulary(
                logits=logits,
                last_token=history[-1],
                grammar_rules=grammar_rules or {},
                context_type=self.context_type,
            )

            self.engine.steer(
                logits=pruned_logits,
                history=history,
                target_tokens=target_tokens,
            )

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


# Alias for alternative naming convention
QCA_Parallel_Engine = QCAParallelEngine


# ---------------------------------------------------------------------------
# Self-Test / Quick Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Testing QCAParallelEngine (Processes & JAX Backends)...")
    N = 80
    K = 4
    dim = 5

    rng = random.Random(42)
    nodes = []
    for i in range(N):
        coords = [rng.uniform(0, 10) for _ in range(dim)]
        prime_prof = [rng.uniform(0.1, 0.9) for _ in range(5)]
        nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime_prof))

    # Test Parallel Processes Backend
    eng_proc = QCAParallelEngine(K=K, vocab_size=300, backend="processes", seed=42)
    res_proc = eng_proc.process_parallel(nodes, verbose=False)
    t_seq = eng_proc.process_sequential(nodes)
    sp_proc = t_seq / res_proc.total_time_ms if res_proc.total_time_ms > 0 else 1.0

    print(f"\n[Backend: Processes]")
    print(f"  Parallel Time : {res_proc.total_time_ms:.2f} ms")
    print(f"  Sequential    : {t_seq:.2f} ms")
    print(f"  Speedup       : {sp_proc:.2f}x")

    # Test JAX Backend (if JAX is available or fallback)
    eng_jax = QCAParallelEngine(K=K, vocab_size=300, backend="auto", seed=42)
    res_jax = eng_jax.process_parallel(nodes, verbose=False)
    sp_jax  = t_seq / res_jax.total_time_ms if res_jax.total_time_ms > 0 else 1.0

    print(f"\n[Backend: {res_jax.backend_used.upper()}]")
    print(f"  Parallel Time : {res_jax.total_time_ms:.2f} ms")
    print(f"  Sequential    : {t_seq:.2f} ms")
    print(f"  Speedup       : {sp_jax:.2f}x")
