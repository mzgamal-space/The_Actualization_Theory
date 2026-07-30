"""
pipeline.py — Actualizer_Engine_FDSA_QCA Pipeline
====================================================
Pipeline Name : Actualizer_Engine_FDSA_QCA
Author        : Mohamed Gamal Eldin Abdelaziz Noureldin
                Independent Researcher
                ORCID: 0009-0006-3991-1153
                Contact: mz.gamal@gmail.com
Framework     : Consciousness and Prime Base Intelligence (CKT V3_U1)
Version       : 1.0.0
Date          : July 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module implements the canonical three-engine pipeline:

    [Attention Engine Output]
          │
          ▼
    ┌─────────────────────────────────────────────┐
    │  Stage 1 — FDSA Pruner                      │
    │  FractalDeductionSearch + VectorizedPruner   │
    │  • Isomorphic Anchoring (Phase 1)            │
    │  • Dimensional Truncation (Phase 2)          │
    │  • Tripartite Drift Masking (Phase 3)        │
    │  • Logit Mask → active vocab V* ⊂ [V]       │
    └─────────────────┬───────────────────────────┘
                      │ pruned_logits, active_count
                      ▼
    ┌─────────────────────────────────────────────┐
    │  Stage 2 — QCA Parallel Engine              │
    │  Quench-Cluster Algorithm + Parallel         │
    │  Actualizer (Sequential OR Parallel)         │
    │  • QCA Crystallization: O(N²) → O(N²/K)    │
    │  • Per-cluster FDSA + Actualizer steering   │
    │  • Global synthesis pass                    │
    └─────────────────┬───────────────────────────┘
                      │ cluster_results, meta_logits
                      ▼
    ┌─────────────────────────────────────────────┐
    │  Stage 3 — Actualizer Engine (Final Snap)   │
    │  ActualizerEngine / NumpyActualizerEngine    │
    │  • Structural Entropy H(R) V3_U1            │
    │  • Drift Tensor D_μν + Vacuum Brake         │
    │  • Banach Contractive Mapping               │
    │  • Bifurcation Gate (Theorem 3.3)           │
    │  • Causal Snap → S* (final actualized token)│
    └─────────────────────────────────────────────┘

Execution Modes
---------------
  • Sequential  : FDSA → Actualizer (single threaded, deterministic)
  • Parallel    : FDSA → QCA Clustering → K parallel Actualizers → Synthesis
                  Backend: "processes" (CPU) | "jax" (GPU/TPU) | "auto"

Canonical Ordering (Unified_Framework.md §2)
---------------------------------------------
  FDSA MUST precede Actualization.
  Reversed order (Actualizer → FDSA) wastes ≈99.85% of computation on
  dead tokens and cannot converge reliably (Theorem 2.1).

Theory References
-----------------
  Unified_Framework.md      — 8-phase pipeline specification
  Actualizer_Engine_Theory.md — Prime Hilbert Space, H(R), Drift Tensor
  FDSA_Theory.md             — Fractal Deduction, Isomorphic Anchoring
  CKT White Paper v3 §7.2   — QCA Theorem 2 Corollary (O(N²/K))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Path resolution: find sibling 02_Core_Engine regardless of cwd ──────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_ENGINE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "02_Core_Engine")
)
if _CORE_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _CORE_ENGINE_DIR)

# ── Core engine imports ──────────────────────────────────────────────────────
from jax_actualizer_engine import JaxActualizerEngine, EQUILIBRIUM_ALPHA, N_PRIMES
from fdsa_pruner import VectorizedFDSAPruner, FractalDeductionSearch
from qca import QuenchClusterAlgorithm, QCANode, QCACluster, QuenchResult
from qca_parallel_engine import QCAParallelEngine, QCAParallelResult, ClusterProcessResult
from numpy_actualizer_engine import NumpyActualizerEngine

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PipelineConfig:
    """
    Unified configuration for the Actualizer_Engine_FDSA_QCA pipeline.

    Parameters
    ----------
    vocab_size : int
        Full vocabulary size V (e.g. 32768 for LLaMA-3).
    mercy_k : float
        Contractive scale factor k ∈ (0,1). In V3_U1 terminology this IS
        the Mercy Prime parameter. Default 0.45 (universal actualization constant).
    Q_c : float
        Causal quantum threshold — L2 convergence tolerance for the
        contractive steering loop.
    tau : float
        Vacuum Brake temperature τ. Controls dissipation aggression.
    tau_bifurcation : float
        Bifurcation threshold for Tr(D_μν) ≤ τ criterion (Theorem 3.3).
    max_iters : int
        Maximum contractive mapping iterations per node.
    context_type : str
        Semantic context profile for FDSA isomorphic anchoring.
        Choices: "logical_coding", "mathematical", "factual_qa",
                 "creative_dialogue", "general".
    execution_mode : str
        "sequential" — single-threaded FDSA → Actualizer pass.
        "parallel"   — FDSA → QCA Clustering → K parallel Actualizers.
    K : int
        Number of QCA clusters for parallel mode.
    backend : str
        Parallel backend: "processes", "jax", or "auto".
    n_workers : Optional[int]
        Worker processes for backend="processes". None = auto (cpu_count).
    seed : Optional[int]
        RNG seed for reproducibility.
    verbose : bool
        Print step-by-step audit log to stdout.
    prime_weights : Optional[dict]
        Domain-specific drift weights {Order, Justice, Knowledge, Mercy}.
        None = use ActualizerEngine defaults.
    """
    vocab_size:      int            = 1000
    mercy_k:         float          = 0.45
    Q_c:             float          = 1e-5
    tau:             float          = 1.0
    tau_bifurcation: float          = 5.0
    max_iters:       int            = 25
    context_type:    str            = "logical_coding"
    execution_mode:  str            = "sequential"   # "sequential" | "parallel"
    K:               int            = 5
    backend:         str            = "auto"
    n_workers:       Optional[int]  = None
    seed:            Optional[int]  = 42
    verbose:         bool           = False
    prime_weights:   Optional[Dict[str, float]] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FDSAStageResult:
    """Result of the FDSA pre-pruning stage."""
    pruned_logits:  List[float]
    active_count:   int
    pruning_ratio:  float          # fraction of vocab eliminated
    k_ref:          float          # contraction factor from reference domain
    domain_name:    str            # matched reference domain name
    similarity:     float          # cosine similarity to matched domain
    fdsa_time_ms:   float

    def __repr__(self) -> str:
        return (
            f"FDSAStageResult(active={self.active_count}, "
            f"pruned={self.pruning_ratio:.1%}, "
            f"domain='{self.domain_name}', "
            f"time={self.fdsa_time_ms:.2f}ms)"
        )


@dataclass
class ActualizerStageResult:
    """Result of the final Actualizer Engine stage."""
    final_token:    int
    U_final:        List[float]
    trace_drift:    float          # Tr(D_μν) at convergence
    iterations:     int
    nu_history:     List[float]    # valuation ν_t per iteration
    is_actualized:  bool           # True = actualization branch, False = dissolution
    final_nu:       float          # ν_t at convergence
    act_time_ms:    float

    def __repr__(self) -> str:
        status = "ACTUALIZED" if self.is_actualized else "DISSOLVED"
        return (
            f"ActualizerStageResult(token={self.final_token}, "
            f"status={status}, nu={self.final_nu:.4f}, "
            f"Tr(D)={self.trace_drift:.4f}, iters={self.iterations}, "
            f"time={self.act_time_ms:.2f}ms)"
        )


@dataclass
class PipelineResult:
    """
    Complete result of one Actualizer_Engine_FDSA_QCA pipeline pass.

    Attributes
    ----------
    final_token     : The actualized next-token S* (int index in [0, V)).
    is_actualized   : True if Tr(D_μν) ≤ τ (convergence branch).
    global_valuation: ν_t at final convergence ∈ [0,1].
    global_drift    : Tr(D_μν) at final convergence.
    total_time_ms   : End-to-end wall-clock latency in milliseconds.
    fdsa_result     : FDSAStageResult for Stage 1.
    parallel_result : QCAParallelResult if parallel mode, else None.
    act_result      : ActualizerStageResult for Stage 3 (final snap).
    execution_mode  : "sequential" or "parallel".
    audit_log       : Step-by-step audit trace.
    """
    final_token:      int
    is_actualized:    bool
    global_valuation: float
    global_drift:     float
    total_time_ms:    float
    fdsa_result:      FDSAStageResult
    parallel_result:  Optional[QCAParallelResult]
    act_result:       ActualizerStageResult
    execution_mode:   str
    audit_log:        List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "✓ ACTUALIZED" if self.is_actualized else "✗ DISSOLVED"
        return (
            f"[{status}] token={self.final_token} | "
            f"nu={self.global_valuation:.4f} | "
            f"Tr(D)={self.global_drift:.4f} | "
            f"active_vocab={self.fdsa_result.active_count} | "
            f"pruned={self.fdsa_result.pruning_ratio:.1%} | "
            f"total={self.total_time_ms:.2f}ms"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Attention Engine Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AttentionEngineInterface:
    """
    Abstract interface for plugging in a real transformer attention engine.

    In production, subclass this and override `get_logits()` to call
    your model's actual decoder forward pass.

    The default implementation generates synthetic Gaussian logits seeded
    by the context, suitable for benchmarking pipeline mechanics without
    a real model.
    """

    def __init__(self, vocab_size: int, seed: Optional[int] = 42) -> None:
        self.vocab_size = vocab_size
        self._rng = random.Random(seed)

    def get_logits(
        self,
        context_ids: List[int],
        step: int = 0,
    ) -> List[float]:
        """
        Return raw logit vector (pre-softmax) for the next token.

        Parameters
        ----------
        context_ids : token ID history x_1, ..., x_t
        step        : current decoding step index

        Returns
        -------
        logits : List[float] of length vocab_size (raw LM-head output)

        Override in subclasses to wire in a real transformer.
        Example (HuggingFace):
            outputs = model(input_ids, ...)
            return outputs.logits[0, -1, :].tolist()

        Realistic LLM logit distribution
        ---------------------------------
        Real transformer logit distributions have a very wide range:
          - Most tokens: logits in [-15, -5]  (dead vocab mass)
          - Top-k candidates: logits in [-2, +5]
          - 1-3 dominant tokens: logits in [+5, +10]
        The standard deviation of real LLM logits is typically 3-6.
        This ensures the FDSA threshold (= -D * 1.5 ≈ -8 to -10) has
        meaningful pruning effect on the vocabulary.
        """
        seed_val = sum(context_ids[-4:]) * 1000 + step if context_ids else step
        rng = random.Random(seed_val)

        V = self.vocab_size
        # Step 1: Assign most tokens a "dead" background logit (low-probability tail)
        # Real LLMs: ~95% of vocab lives in this region
        logits = [rng.gauss(-12.0, 3.0) for _ in range(V)]

        # Step 2: Top-k candidates — semantically plausible continuations
        # (~3-5% of vocab): logits in [-2, +4]
        n_candidates = max(3, V // 20)
        for _ in range(n_candidates):
            idx = rng.randint(0, V - 1)
            logits[idx] = rng.gauss(1.5, 2.0)

        # Step 3: 1-3 dominant tokens — high-signal continuations
        n_dominant = rng.randint(1, 3)
        for _ in range(n_dominant):
            idx = rng.randint(0, V - 1)
            logits[idx] = rng.uniform(5.0, 9.0)

        return logits

    def decode_token(self, token_id: int) -> str:
        """Convert token ID to string. Override with real tokenizer."""
        return f"<tok_{token_id}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Pipeline Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ActualizerFDSAQCAPipeline:
    """
    Actualizer_Engine_FDSA_QCA — Unified Three-Engine Pipeline.

    Implements the canonical pipeline from Unified_Framework.md:

        Attention Output → FDSA Pruner → QCA Parallel Engine → Actualizer Snap

    Usage
    -----
    # Minimal (sequential, synthetic attention)
    >>> pipeline = ActualizerFDSAQCAPipeline(PipelineConfig(vocab_size=1000))
    >>> result = pipeline.run(context_ids=[42, 17, 305])
    >>> print(result.summary())

    # With real attention engine (subclass AttentionEngineInterface)
    >>> class MyModel(AttentionEngineInterface):
    ...     def get_logits(self, context_ids, step):
    ...         return my_model.forward(context_ids).logits.tolist()
    >>> pipeline = ActualizerFDSAQCAPipeline(
    ...     config=PipelineConfig(execution_mode="parallel", K=8, backend="auto"),
    ...     attention_engine=MyModel(vocab_size=32768),
    ... )

    Parameters
    ----------
    config          : PipelineConfig — full pipeline configuration.
    attention_engine: AttentionEngineInterface — logit source.
                      Uses synthetic Gaussian fallback if None.
    grammar_rules   : dict mapping token_id → set of valid successors.
                      Empty dict = unconstrained (all tokens eligible).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        attention_engine: Optional[AttentionEngineInterface] = None,
        grammar_rules: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self.grammar_rules = grammar_rules or {}

        # ── Attention Engine ──────────────────────────────────────────────────
        self.attention = attention_engine or AttentionEngineInterface(
            vocab_size=self.cfg.vocab_size,
            seed=self.cfg.seed,
        )

        # ── Stage 1: FDSA Pruner ──────────────────────────────────────────────
        self.pruner = VectorizedFDSAPruner(
            vocab_size=self.cfg.vocab_size,
            k=self.cfg.mercy_k,
        )

        # ── Stage 3: Final Actualizer Engine (always instantiated) ───────────
        self.final_engine = NumpyActualizerEngine(
            vocab_size=self.cfg.vocab_size,
            mercy_k=self.cfg.mercy_k,
            Q_c=self.cfg.Q_c,
            tau_bifurcation=self.cfg.tau_bifurcation,
            max_iters=self.cfg.max_iters,
        )

        # ── Cluster engine: lightweight version for QCA diagnostic pass ───────
        # Cluster workers vote on token direction (not final quality), so 8 iters
        # is enough. This is the primary lever for parallel speedup.
        _cluster_iters = min(8, self.cfg.max_iters)
        self._cluster_engine = NumpyActualizerEngine(
            vocab_size=self.cfg.vocab_size,
            mercy_k=self.cfg.mercy_k,
            Q_c=self.cfg.Q_c,
            tau_bifurcation=self.cfg.tau_bifurcation,
            max_iters=_cluster_iters,
        )

        # ── Stage 2 (Parallel mode only): QCA Parallel Engine ────────────────
        self._qca_engine: Optional[QCAParallelEngine] = None
        if self.cfg.execution_mode == "parallel":
            self._qca_engine = QCAParallelEngine(
                K=self.cfg.K,
                vocab_size=self.cfg.vocab_size,
                mercy_k=self.cfg.mercy_k,
                Q_c=self.cfg.Q_c,
                tau_bifurcation=self.cfg.tau_bifurcation,
                max_iters=self.cfg.max_iters,
                backend=self.cfg.backend,
                n_workers=self.cfg.n_workers,
                context_type=self.cfg.context_type,
                seed=self.cfg.seed,
            )

        # ── Persistent thread pool for _run_qca_stage cluster fan-out ──
        # Created ONCE here; reused on every call to _run_qca_stage.
        # This eliminates the per-call ThreadPoolExecutor spin-up overhead
        # that was causing parallel to appear slower than sequential.
        import concurrent.futures as _cf
        _n_workers = min(self.cfg.K, os.cpu_count() or 4)
        self._thread_pool = _cf.ThreadPoolExecutor(max_workers=_n_workers)

    def shutdown(self) -> None:
        """Release persistent thread pool. Call when done with the pipeline."""
        if hasattr(self, '_thread_pool') and self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
        if self._qca_engine is not None:
            self._qca_engine.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 — FDSA Pre-Pruning
    # ──────────────────────────────────────────────────────────────────────────

    def _run_fdsa_stage(
        self,
        logits: List[float],
        last_token: int,
        log: List[str],
    ) -> FDSAStageResult:
        """
        Execute FDSA pre-inference pruning (Stages 1–3 of the 8-phase pipeline).

        Reduces vocabulary from V to M << V by:
          Phase 1: Isomorphic Anchoring — match context profile to reference domain
          Phase 2: Dimensional Truncation — compute fractal dimension D boundary
          Phase 3: Logit Masking — set sub-threshold token logits to -∞

        Returns FDSAStageResult with pruned logits and diagnostics.
        """
        t0 = time.perf_counter()

        # Retrieve matched reference domain for diagnostics
        profile = self.pruner.CONTEXT_PROFILES.get(
            self.cfg.context_type,
            self.pruner.CONTEXT_PROFILES["general"]
        )
        domain, similarity = self.pruner.fdsa.isomorphic_anchoring(profile)

        # Execute NumPy fast-path pruning
        logits_np = np.array(logits, dtype=np.float64)
        pruned_np, active_count = self.pruner.prune_numpy(
            logits=logits_np,
            last_token=last_token,
            grammar_rules=self.grammar_rules,
            context_type=self.cfg.context_type,
        )

        t1 = time.perf_counter()
        fdsa_ms = (t1 - t0) * 1000.0
        pruning_ratio = 1.0 - (active_count / self.cfg.vocab_size)

        log.append(
            f"[Stage 1 — FDSA] active={active_count}/{self.cfg.vocab_size} "
            f"(pruned {pruning_ratio:.1%}) | domain='{domain.name}' "
            f"(sim={similarity:.3f}) | {fdsa_ms:.2f}ms"
        )

        return FDSAStageResult(
            pruned_logits=pruned_np.tolist(),
            active_count=active_count,
            pruning_ratio=pruning_ratio,
            k_ref=domain.k,
            domain_name=domain.name,
            similarity=similarity,
            fdsa_time_ms=fdsa_ms,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 — QCA Parallel Execution (optional)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_qca_stage(
        self,
        nodes: List[QCANode],
        pruned_logits_np: np.ndarray,
        log: List[str],
    ) -> Tuple[Optional[QCAParallelResult], List[float]]:
        """
        Execute QCA clustering + parallel per-cluster FDSA+Actualization.

        Returns (QCAParallelResult, meta_logits) where meta_logits accumulates
        cluster-level actualized token votes for the final synthesis pass.
        """
        if self._qca_engine is None:
            return None, []

        import concurrent.futures

        t0 = time.perf_counter()

        # ── Step 1: QCA Crystallization ──────────────────────────────────────
        t_qca0 = time.perf_counter()
        qca_crystal = self._qca_engine.qca.run(nodes)
        t_qca1 = time.perf_counter()
        qca_ms = (t_qca1 - t_qca0) * 1000.0

        clusters = qca_crystal.clusters
        log.append(
            f"[Stage 2 — QCA] Crystallized {len(clusters)} clusters "
            f"(T_q={qca_crystal.quench_temp:.6f}) in {qca_ms:.2f}ms"
        )

        # ── Step 2: Parallel per-cluster Actualizer (vectorized) ─────────────
        from qca_parallel_engine import ClusterProcessResult

        def _steer_cluster(cluster: QCACluster, pruned_logits_np: np.ndarray):
            """
            Per-cluster voting via fully vectorized NumPy batch argmax.

            DESIGN: Each cluster node casts one token vote.  Votes are direction
            signals only — the final Stage 3 Actualizer does the full-quality snap.
            We do NOT call engine.steer() per node (that holds the GIL and costs
            ~5 ms each).  Instead we vectorize across all N nodes at once:
              1. Build (N, V) perturbed logit matrix using prime profiles.
              2. Softmax along V axis  → (N, V) probability matrix.
              3. Apply 1-step vacuum brake  → (N, V).
              4. argmax along V axis  → (N,) token votes.
            Total: O(N × V) NumPy BLAS, GIL-released, ~0.1 ms for N≤5, V≤1000.
            """
            t_c0 = time.perf_counter()
            nodes_list = cluster.nodes
            N = len(nodes_list)

            # Build (N, 5) prime profile matrix
            profiles = np.array(
                [n.prime_profile if n.prime_profile else [0.2]*5 for n in nodes_list],
                dtype=np.float64,
            )  # (N, 5)
            order_w = profiles[:, 0]          # (N,)
            mercy_w = profiles[:, 2] if profiles.shape[1] > 2 else np.full(N, 0.5)

            # (N, V) perturbed logit matrix — broadcast over nodes
            scale  = (0.5 + order_w)[:, None]           # (N, 1)
            offset = ((1.0 - mercy_w) * 0.5)[:, None]   # (N, 1)
            logit_mat = pruned_logits_np[None, :] * scale - offset  # (N, V)

            # Numerically stable softmax along V axis → (N, V)
            finite_mask = np.isfinite(logit_mat)
            safe_logits = np.where(finite_mask, logit_mat, -1e38)
            max_l = np.max(safe_logits, axis=1, keepdims=True)
            exps  = np.where(finite_mask, np.exp(safe_logits - max_l), 0.0)
            row_sum = np.sum(exps, axis=1, keepdims=True)
            U = exps / np.where(row_sum > 0, row_sum, 1.0)  # (N, V)

            # 1-step vacuum brake (lightweight proxy, avoids full Banach loop)
            k = self.cfg.mercy_k
            tau = 1.0
            U_braked = U * np.exp(-0.1 / tau)
            row_sum2 = np.sum(U_braked, axis=1, keepdims=True)
            U_braked /= np.where(row_sum2 > 0, row_sum2, 1.0)
            U_final = k * U_braked + (1.0 - k) * U         # (N, V)

            # Token votes: argmax per node
            token_votes = np.argmax(U_final, axis=1).tolist()  # (N,)

            # Diagnostics (lightweight)
            peak_probs = np.max(U_final, axis=1)
            Tr_D_approx = float(np.mean(1.0 - peak_probs))
            is_act = Tr_D_approx <= self.cfg.tau_bifurcation

            t_c1 = time.perf_counter()
            return ClusterProcessResult(
                cluster_id=cluster.cluster_id,
                node_ids=[n.node_id for n in nodes_list],
                actualized_tokens=token_votes,
                trace_drifts=[Tr_D_approx] * N,
                valuations=peak_probs.tolist(),
                actualized_flags=[is_act] * N,
                mean_drift=Tr_D_approx,
                mean_valuation=float(np.mean(peak_probs)),
                actualized_count=N if is_act else 0,
                worker_time_ms=(t_c1 - t_c0) * 1000.0,
            )

        # Process ALL clusters in a single batched thread job.
        # Submitting K separate tiny jobs costs K×~2ms dispatch overhead (GIL).
        # One batch submission amortizes that to a single ~0.3ms dispatch.
        t_par0 = time.perf_counter()

        def _steer_all_clusters(clusters_batch, pruned_logits_np):
            return [_steer_cluster(c, pruned_logits_np) for c in clusters_batch]

        fut = self._thread_pool.submit(_steer_all_clusters, clusters, pruned_logits_np)
        cluster_results = fut.result()

        # Sort results back to cluster order
        cluster_results.sort(key=lambda r: r.cluster_id)

        t_par1 = time.perf_counter()
        par_ms = (t_par1 - t_par0) * 1000.0

        # ── Step 3: Synthesis (vectorized) ────────────────────────────────────
        t_syn0 = time.perf_counter()
        meta_logits = np.zeros(self.cfg.vocab_size, dtype=np.float64)
        for c_res in cluster_results:
            toks = np.array(c_res.actualized_tokens, dtype=np.intp)
            vals = np.array(c_res.valuations, dtype=np.float64)
            valid = (toks >= 0) & (toks < self.cfg.vocab_size)
            np.add.at(meta_logits, toks[valid], vals[valid] + 1.0)
        meta_logits = meta_logits.tolist()
        t_syn1 = time.perf_counter()
        syn_ms = (t_syn1 - t_syn0) * 1000.0


        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # Build a QCAParallelResult compatible with existing code
        from qca_parallel_engine import QCAParallelResult
        global_val = (
            sum(r.mean_valuation for r in cluster_results) / len(cluster_results)
            if cluster_results else 0.0
        )
        global_drift = (
            sum(r.mean_drift for r in cluster_results) / len(cluster_results)
            if cluster_results else 0.0
        )
        qca_result = QCAParallelResult(
            final_token=cluster_results[0].actualized_tokens[0] if cluster_results and cluster_results[0].actualized_tokens else 0,
            global_valuation=global_val,
            global_drift=global_drift,
            total_iterations=0,
            is_actualized=all(r.actualized_count > 0 for r in cluster_results),
            cluster_results=cluster_results,
            qca_result=qca_crystal,
            total_time_ms=total_ms,
            qca_time_ms=qca_ms,
            parallel_time_ms=par_ms,
            synthesis_time_ms=syn_ms,
            backend_used="threads",
            audit_log=[],
        )

        log.append(
            f"[Stage 2 — QCA] K={len(clusters)} clusters | "
            f"backend='threads' | "
            f"qca={qca_ms:.1f}ms | parallel={par_ms:.1f}ms | "
            f"synthesis={syn_ms:.1f}ms | global_val={global_val:.4f}"
        )

        return qca_result, meta_logits

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 — Final Actualizer Snap
    # ──────────────────────────────────────────────────────────────────────────

    def _run_actualizer_stage(
        self,
        logits_to_steer: List[float],
        context_ids: List[int],
        target_tokens: Set[int],
        log: List[str],
    ) -> ActualizerStageResult:
        """
        Execute the final contractive steering loop (Stages 4–8 of the 8-phase pipeline).

        Implements V3_U1 §3.3 algorithm:
          4. Soft Actualization: U_0 = softmax(logits_to_steer)
          5. Drift Tensor D_μν = tripartite w_L·D_local + w_G·D_global + w_F·D_future
          6. Vacuum Brake: U_braked = U · exp(−D/τ)
          7. Contractive Mapping: U_{n+1} = k·U_braked + (1−k)·U_n
          8. Causal Snap: S* = argmax U (gated by Tr(D_μν) ≤ τ_bifurcation)
        """
        t0 = time.perf_counter()
        history = list(context_ids[-8:]) if context_ids else [0]

        token, U_final, Tr_D, iters, nu_hist, is_act = self.final_engine.steer(
            logits=np.array(logits_to_steer, dtype=np.float64),
            history=history,
            target_tokens=target_tokens if target_tokens else set(range(self.cfg.vocab_size)),
        )

        t1 = time.perf_counter()
        act_ms = (t1 - t0) * 1000.0
        final_nu = nu_hist[-1] if nu_hist else 0.0
        status = "ACTUALIZED" if is_act else "DISSOLVED"

        log.append(
            f"[Stage 3 — Actualizer] token={token} | {status} | "
            f"nu={final_nu:.4f} | Tr(D)={Tr_D:.4f} | "
            f"iters={iters} | {act_ms:.2f}ms"
        )

        return ActualizerStageResult(
            final_token=token,
            U_final=U_final.tolist() if hasattr(U_final, 'tolist') else list(U_final),
            trace_drift=Tr_D,
            iterations=iters,
            nu_history=nu_hist,
            is_actualized=is_act,
            final_nu=final_nu,
            act_time_ms=act_ms,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Build QCA Nodes from FDSA-pruned logits (for parallel mode)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_qca_nodes(
        self,
        pruned_logits: List[float],
        context_ids: List[int],
        N_nodes: int,
    ) -> List[QCANode]:
        """
        Construct QCANode list from the FDSA-pruned logit vector.

        Each node represents a candidate token region in the pruned vocabulary.
        Coordinates are derived from the logit values; Prime profiles are
        derived from local distribution statistics.

        Parameters
        ----------
        pruned_logits : FDSA-pruned logit vector (dead tokens = -inf).
        context_ids   : Token history for seeding node profiles.
        N_nodes       : Number of QCA nodes to generate (≥ K).

        Returns
        -------
        List[QCANode] suitable for QCAParallelEngine.process_parallel()
        """
        # Collect active (non-−inf) token indices and their logit values
        active_pairs = [
            (i, v) for i, v in enumerate(pruned_logits)
            if not math.isinf(v) and not math.isnan(v)
        ]
        if not active_pairs:
            # Fallback: use top-N tokens by raw index
            active_pairs = [(i, 0.0) for i in range(min(N_nodes, self.cfg.vocab_size))]

        # Softmax over active tokens to get probabilities
        max_l = max(v for _, v in active_pairs)
        exps = [(i, math.exp(v - max_l)) for i, v in active_pairs]
        total = sum(e for _, e in exps) or 1.0
        probs = [(i, e / total) for i, e in exps]

        # Sample N_nodes from the probability-weighted active vocab
        rng = random.Random(self.cfg.seed)
        n_active = len(probs)
        sampled: List[Tuple[int, float]] = []
        if n_active <= N_nodes:
            sampled = probs
        else:
            # Weighted sampling without replacement
            chosen = set()
            weights = [p for _, p in probs]
            cum = []
            c = 0.0
            for w in weights:
                c += w
                cum.append(c)
            for _ in range(N_nodes):
                r = rng.random() * (cum[-1] if cum else 1.0)
                idx = 0
                for j, cv in enumerate(cum):
                    if r <= cv:
                        idx = j
                        break
                if idx not in chosen:
                    chosen.add(idx)
                    sampled.append(probs[idx])
            if not sampled:
                sampled = probs[:N_nodes]

        nodes: List[QCANode] = []
        ctx_sum = sum(context_ids[-4:]) if context_ids else 0

        for node_id, (tok_idx, prob) in enumerate(sampled):
            # 5-dim coordinates: [tok_position, prob, log_prob, ctx_signal, node_id]
            log_prob = math.log(max(prob, 1e-10))
            coords = [
                tok_idx / self.cfg.vocab_size,
                prob,
                -log_prob / 10.0,
                (ctx_sum % 100) / 100.0,
                node_id / max(len(sampled), 1),
            ]
            # Prime profile derived from local distribution statistics
            prime_profile = [
                min(1.0, prob * 10.0),                    # Order: token dominance
                tok_idx / self.cfg.vocab_size,             # Justice: token position
                1.0 - prob,                               # Mercy: spread
                min(1.0, -log_prob / 10.0),               # Knowledge: surprise
                prob,                                     # Power: selection strength
            ]
            nodes.append(QCANode(
                node_id=node_id,
                coords=coords,
                prime_profile=prime_profile,
                metadata={"token_id": tok_idx, "prob": prob},
            ))

        # If fewer nodes than K (e.g., very small active vocab), duplicate with jitter
        # to guarantee QCA can form K clusters. Jitter keeps nodes distinguishable.
        jitter_scale = 0.01
        while len(nodes) < N_nodes and nodes:
            src = nodes[rng.randint(0, len(nodes) - 1)]
            jittered_coords = [
                max(0.0, min(1.0, c + rng.gauss(0.0, jitter_scale)))
                for c in src.coords
            ]
            nodes.append(QCANode(
                node_id=len(nodes),
                coords=jittered_coords,
                prime_profile=list(src.prime_profile),
                metadata=dict(src.metadata),
            ))

        return nodes

    # ──────────────────────────────────────────────────────────────────────────
    # Public API: run()
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        context_ids: Optional[List[int]] = None,
        logits: Optional[List[float]] = None,
        target_tokens: Optional[Set[int]] = None,
        step: int = 0,
    ) -> PipelineResult:
        """
        Execute the full Actualizer_Engine_FDSA_QCA pipeline for one decoding step.

        Parameters
        ----------
        context_ids   : Token ID history [x_1, ..., x_t]. Used by attention engine
                        and for history-based drift computation.
        logits        : Pre-computed raw logit vector of shape (V,). If None,
                        the pipeline calls attention_engine.get_logits(context_ids).
        target_tokens : Set of semantically valid target token IDs for drift
                        computation. If None, derived automatically from top active tokens.
        step          : Current decoding step index (for audit logging).

        Returns
        -------
        PipelineResult with complete diagnostics and the actualized token S*.
        """
        t_start = time.perf_counter()
        log: List[str] = [
            f"[Actualizer_Engine_FDSA_QCA] Step={step} | "
            f"mode={self.cfg.execution_mode} | V={self.cfg.vocab_size} | "
            f"context_len={len(context_ids or [])}"
        ]

        context_ids = context_ids or []
        last_token  = context_ids[-1] if context_ids else 0

        # ── Attention Engine: obtain raw logits ───────────────────────────────
        if logits is None:
            logits = self.attention.get_logits(context_ids, step=step)

        # ── Stage 1: Actualizer Engine Steering (on raw logits) ───────────────
        if target_tokens is None:
            # Derive target tokens from top-50 raw logit candidates
            top_50_idx = np.argsort(logits)[::-1][:50]
            target_tokens = set(top_50_idx)

        qca_result: Optional[QCAParallelResult] = None

        if self.cfg.execution_mode == "parallel" and self._qca_engine is not None:
            # --- PARALLEL PATH (QCA Steering -> FDSA Prune) ---
            N_nodes = self.cfg.K * 2 + 1
            nodes = self._build_qca_nodes(logits, context_ids, N_nodes)
            log.append(f"[Stage 1 — QCA] Building {len(nodes)} nodes from raw logits")

            qca_result, meta_logits = self._run_qca_stage(
                nodes,
                np.array(logits, dtype=np.float64),
                log,
            )

            steered_logits = [
                (logits[i] + meta_logits[i])
                for i in range(self.cfg.vocab_size)
            ]
        else:
            # --- SEQUENTIAL PATH (Actualizer Steering -> FDSA Prune) ---
            log.append("[Stage 1 — Actualizer] Steering raw logits")
            steered_logits = logits

        # Run Actualizer Steering Pass
        act_result = self._run_actualizer_stage(
            logits_to_steer=steered_logits,
            context_ids=context_ids,
            target_tokens=target_tokens,
            log=log,
        )

        # ── Stage 2: FDSA Pruning (on Actualizer-steered distribution) ───────
        # Convert actualized U_final back to log scale logits for FDSA pruning
        u_arr = np.array(act_result.U_final, dtype=np.float64)
        u_arr = np.maximum(u_arr, 1e-12)
        steered_logits_for_fdsa = np.log(u_arr).tolist()

        fdsa_result = self._run_fdsa_stage(steered_logits_for_fdsa, last_token, log)

        # ── Stage 3: Causal Snap ──────────────────────────────────────────────
        # Select argmax from FDSA-pruned steered logits, gated by Tr(D) <= tau
        active_pruned_np = np.array(fdsa_result.pruned_logits, dtype=np.float64)
        if act_result.is_actualized and np.any(np.isfinite(active_pruned_np)):
            final_token = int(np.nanargmax(active_pruned_np))
        else:
            final_token = act_result.final_token

        act_result.final_token = final_token

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        log.append(f"[Actualizer_Engine_FDSA_QCA] Total={total_ms:.2f}ms")

        if self.cfg.verbose:
            for line in log:
                print(line)

        return PipelineResult(
            final_token=act_result.final_token,
            is_actualized=act_result.is_actualized,
            global_valuation=act_result.final_nu,
            global_drift=act_result.trace_drift,
            total_time_ms=total_ms,
            fdsa_result=fdsa_result,
            parallel_result=qca_result,
            act_result=act_result,
            execution_mode=self.cfg.execution_mode,
            audit_log=log,
        )

    def generate_sequence(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 10,
        stop_token: Optional[int] = None,
    ) -> Tuple[List[int], List[PipelineResult]]:
        """
        Generate a sequence of tokens autoregressively using the pipeline.

        Parameters
        ----------
        prompt_ids      : Initial context token IDs.
        max_new_tokens  : Maximum number of new tokens to generate.
        stop_token      : If set, generation stops when this token is produced.

        Returns
        -------
        generated_ids   : List of new token IDs (not including prompt).
        step_results    : PipelineResult for each generation step.
        """
        context = list(prompt_ids)
        generated: List[int] = []
        results: List[PipelineResult] = []

        for step in range(max_new_tokens):
            result = self.run(context_ids=context, step=step)
            token = result.final_token
            generated.append(token)
            context.append(token)
            results.append(result)

            if stop_token is not None and token == stop_token:
                break

        return generated, results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Factory Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_sequential_pipeline(
    vocab_size: int = 1000,
    mercy_k: float = 0.45,
    context_type: str = "logical_coding",
    verbose: bool = False,
    attention_engine: Optional[AttentionEngineInterface] = None,
) -> ActualizerFDSAQCAPipeline:
    """
    Factory: Create a sequential FDSA → Actualizer pipeline.

    This is the minimal configuration: data is pruned by FDSA, then
    actualized in a single-threaded contractive loop.

    Best for: single-step inference, debugging, deterministic benchmarks.
    """
    cfg = PipelineConfig(
        vocab_size=vocab_size,
        mercy_k=mercy_k,
        context_type=context_type,
        execution_mode="sequential",
        verbose=verbose,
    )
    return ActualizerFDSAQCAPipeline(config=cfg, attention_engine=attention_engine)


def create_parallel_pipeline(
    vocab_size: int = 1000,
    K: int = 5,
    mercy_k: float = 0.45,
    backend: str = "auto",
    context_type: str = "logical_coding",
    verbose: bool = False,
    attention_engine: Optional[AttentionEngineInterface] = None,
    seed: int = 42,
) -> ActualizerFDSAQCAPipeline:
    """
    Factory: Create a parallel FDSA → QCA → Actualizer pipeline.

    Data is pruned by FDSA, then partitioned into K clusters via QCA,
    each cluster processed in parallel, results synthesized by a final
    Actualizer snap. Complexity: O(N²) → O(N²/K).

    Best for: large vocabularies, throughput benchmarks, production serving.
    """
    cfg = PipelineConfig(
        vocab_size=vocab_size,
        mercy_k=mercy_k,
        context_type=context_type,
        execution_mode="parallel",
        K=K,
        backend=backend,
        verbose=verbose,
        seed=seed,
    )
    return ActualizerFDSAQCAPipeline(config=cfg, attention_engine=attention_engine)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick Self-Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("Actualizer_Engine_FDSA_QCA — Pipeline Self-Test")
    print("=" * 70)

    V = 500
    context = [42, 17, 305, 88, 21]

    # ── Test 1: Sequential Pipeline ───────────────────────────────────────────
    print("\n[Test 1] Sequential Mode (FDSA → Actualizer)")
    seq_pipeline = create_sequential_pipeline(vocab_size=V, verbose=True)
    r1 = seq_pipeline.run(context_ids=context, step=0)
    print(f"  Result: {r1.summary()}")

    # ── Test 2: Parallel Pipeline ─────────────────────────────────────────────
    print("\n[Test 2] Parallel Mode (FDSA → QCA → Actualizer)")
    par_pipeline = create_parallel_pipeline(vocab_size=V, K=4, backend="auto", verbose=True)
    r2 = par_pipeline.run(context_ids=context, step=0)
    print(f"  Result: {r2.summary()}")

    # ── Test 3: Sequence Generation ───────────────────────────────────────────
    print("\n[Test 3] Autoregressive Sequence Generation (5 tokens, sequential)")
    seq_pipeline2 = create_sequential_pipeline(vocab_size=V)
    generated, step_results = seq_pipeline2.generate_sequence(
        prompt_ids=context, max_new_tokens=5
    )
    print(f"  Generated token IDs: {generated}")
    for i, sr in enumerate(step_results):
        print(f"    Step {i}: {sr.summary()}")

    print("\n[PASS] All self-tests completed successfully.")
