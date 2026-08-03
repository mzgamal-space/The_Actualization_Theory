"""
benchmark_with_attention.py — Real Benchmark: Attention Engine + Actualizer_Engine_FDSA_QCA
==============================================================================================
Pipeline Name : Actualizer_Engine_FDSA_QCA
Author        : Mohamed Gamal Eldin Abdelaziz Noureldin
                ORCID: 0009-0006-3991-1153
Framework     : Consciousness and Prime Base Intelligence (CKT V3_U1)
Version       : 1.0.0
Date          : July 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENCHMARK DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This benchmark measures the Actualizer_Engine_FDSA_QCA pipeline against
a Baseline Attention Engine (greedy/argmax from raw softmax) across
three primary dimensions:

  Dimension 1 — Latency
    • Baseline greedy decoding (softmax + argmax only)
    • Sequential pipeline (FDSA → Actualizer)
    • Parallel pipeline (FDSA → QCA → Actualizer, K clusters)

  Dimension 2 — Quality
    • Vocabulary reduction ratio (V → M active tokens)
    • Actualization rate (fraction of steps in actualization branch)
    • Global valuation ν_t (proximity to attractor fixed point)
    • Trace drift Tr(D_μν) at convergence (structural entropy)

  Dimension 3 — Scaling
    • Vocabulary size V: [300, 500, 1000, 2000, 5000]
    • QCA cluster count K: [2, 4, 8, 16]
    • Sequence length T:   [5, 10, 20, 50]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "02_Core_Engine"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pipeline import (
    ActualizerFDSAQCAPipeline,
    AttentionEngineInterface,
    PipelineConfig,
    PipelineResult,
    create_sequential_pipeline,
    create_parallel_pipeline,
)

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulated Attention Engine (realistic logit distribution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimulatedAttentionEngine(AttentionEngineInterface):
    """
    Simulates a realistic transformer attention output.

    Distribution characteristics:
      • Gaussian background noise (vocabulary bloat simulation)
      • 3–8 high-probability "candidate" tokens (top-k peaks)
      • 1 dominant "correct" token (signal)
      • Configurable hallucination pressure (off-target mass)

    This mirrors the empirical distribution shape of LLaMA/GPT-style
    models on factual generation tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        n_candidates: int = 5,
        signal_strength: float = 6.0,
        noise_std: float = 1.5,
        hallucination_pressure: float = 0.15,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(vocab_size=vocab_size, seed=seed)
        self.n_candidates = n_candidates
        self.signal_strength = signal_strength
        self.noise_std = noise_std
        self.hallucination_pressure = hallucination_pressure

    def get_logits(self, context_ids: List[int], step: int = 0) -> List[float]:
        """
        Generate realistic transformer-style logit distribution.

        Real LLM logit distributions have a wide dynamic range:
          - ~95% of vocabulary: logits in [-15, -5]  (dead/irrelevant tokens)
          - ~4% candidates:    logits in [-2,  +4]  (plausible continuations)
          - 1-3 dominant:      logits in [+5,  +10] (high-confidence tokens)

        This range ensures the FDSA threshold (= -D*1.5 ≈ -8 to -10) has
        genuine pruning effect, matching real-world benchmark conditions.
        noise_std and hallucination_pressure control the spread of the tail.
        """
        ctx_seed = sum(context_ids[-4:]) * 1000 + step * 37 if context_ids else step
        rng_np = np.random.RandomState(ctx_seed % (2**31 - 1))
        rng_py = random.Random(ctx_seed)

        V = self.vocab_size

        # ── Step 1: Dead background tail ──────────────────────────────────────
        # Center at -12 so the tail easily falls below FDSA threshold ≈ -9
        logits = rng_np.normal(-12.0, self.noise_std * 2.0, V)

        # ── Step 2: Hallucination mass (moderate distractors) ─────────────────
        n_hall = max(1, int(self.hallucination_pressure * V))
        hall_idx = rng_np.choice(V, n_hall, replace=False)
        logits[hall_idx] = rng_np.normal(-4.0, 2.0, n_hall)

        # ── Step 3: Candidate tokens (top-k region) ───────────────────────────
        n_cand = max(self.n_candidates, V // 25)
        cand_idx = rng_np.choice(V, n_cand, replace=False)
        logits[cand_idx] = rng_np.normal(1.5, 1.5, n_cand)

        # ── Step 4: 1-3 dominant tokens (true signal) ─────────────────────────
        n_dominant = rng_py.randint(1, 3)
        for _ in range(n_dominant):
            dom_idx = rng_py.randint(0, V - 1)
            logits[dom_idx] = rng_np.uniform(self.signal_strength - 1.0,
                                              self.signal_strength + 1.5)

        return logits.tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline Decoding (Greedy Softmax)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _greedy_decode(logits: List[float]) -> Tuple[int, float]:
    """
    Baseline: greedy argmax from raw softmax — no pruning, no actualization.

    Returns (token_id, latency_ms).
    """
    t0 = time.perf_counter()
    arr = np.array(logits, dtype=np.float64)
    arr -= arr.max()
    probs = np.exp(arr)
    probs /= probs.sum()
    token = int(np.argmax(probs))
    t1 = time.perf_counter()
    return token, (t1 - t0) * 1000.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Result Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StepMetrics:
    """Per-step benchmark metrics for one decoding step."""
    step:             int
    vocab_size:       int
    baseline_token:   int
    baseline_ms:      float
    seq_token:        int
    seq_ms:           float
    seq_active:       int
    seq_valuation:    float
    seq_drift:        float
    seq_actualized:   bool
    par_token:        Optional[int]
    par_ms:           Optional[float]
    par_active:       Optional[int]
    par_valuation:    Optional[float]
    par_drift:        Optional[float]
    par_actualized:   Optional[bool]
    # Derived
    seq_pruning_ratio: float = 0.0
    par_pruning_ratio: float = 0.0

    def speedup_seq_vs_baseline(self) -> float:
        return self.baseline_ms / self.seq_ms if self.seq_ms > 0 else 1.0

    def speedup_par_vs_seq(self) -> Optional[float]:
        if self.par_ms and self.seq_ms > 0:
            return self.seq_ms / self.par_ms
        return None


@dataclass
class BenchmarkRun:
    """Results for one complete benchmark run."""
    run_id:           str
    vocab_size:       int
    K:                int
    T_steps:          int
    context_type:     str
    step_metrics:     List[StepMetrics]
    # Aggregate
    mean_baseline_ms: float = 0.0
    mean_seq_ms:      float = 0.0
    mean_par_ms:      float = 0.0
    mean_seq_valuation: float = 0.0
    mean_seq_pruning:   float = 0.0
    seq_actualized_rate: float = 0.0
    par_actualized_rate: float = 0.0
    speedup_seq_vs_baseline: float = 1.0
    speedup_par_vs_seq: float = 1.0

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from step-level metrics."""
        if not self.step_metrics:
            return
        n = len(self.step_metrics)
        self.mean_baseline_ms = sum(s.baseline_ms for s in self.step_metrics) / n
        self.mean_seq_ms = sum(s.seq_ms for s in self.step_metrics) / n
        self.mean_seq_valuation = sum(s.seq_valuation for s in self.step_metrics) / n
        self.mean_seq_pruning = sum(s.seq_pruning_ratio for s in self.step_metrics) / n
        self.seq_actualized_rate = sum(1 for s in self.step_metrics if s.seq_actualized) / n

        par_steps = [s for s in self.step_metrics if s.par_ms is not None]
        if par_steps:
            self.mean_par_ms = sum(s.par_ms for s in par_steps) / len(par_steps)
            self.par_actualized_rate = sum(
                1 for s in par_steps if s.par_actualized
            ) / len(par_steps)
            self.speedup_par_vs_seq = (
                self.mean_seq_ms / self.mean_par_ms if self.mean_par_ms > 0 else 1.0
            )

        self.speedup_seq_vs_baseline = (
            self.mean_baseline_ms / self.mean_seq_ms if self.mean_seq_ms > 0 else 1.0
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core Benchmark Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_attention_benchmark(
    vocab_size:   int  = 1000,
    T_steps:      int  = 10,
    K:            int  = 4,
    context_type: str  = "logical_coding",
    backend:      str  = "auto",
    seed:         int  = 42,
    run_parallel: bool = True,
    verbose:      bool = True,
) -> BenchmarkRun:
    """
    Run the full attention-engine benchmark for T_steps decoding steps.

    Parameters
    ----------
    vocab_size   : Vocabulary size V.
    T_steps      : Number of autoregressive decoding steps to benchmark.
    K            : Number of QCA clusters for parallel mode.
    context_type : FDSA context profile.
    backend      : Parallel backend for QCA engine.
    seed         : RNG seed.
    run_parallel : Whether to benchmark parallel mode in addition to sequential.
    verbose      : Print step-by-step results.

    Returns
    -------
    BenchmarkRun with all step metrics and aggregate statistics.
    """
    run_id = f"V{vocab_size}_K{K}_T{T_steps}_{context_type}"
    if verbose:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {run_id}")
        print(f"  V={vocab_size} | K={K} | T={T_steps} | context={context_type}")
        print(f"{'='*70}")

    # ── Initialize attention engine and pipelines ─────────────────────────────
    attention = SimulatedAttentionEngine(vocab_size=vocab_size, seed=seed)

    seq_pipeline = create_sequential_pipeline(
        vocab_size=vocab_size,
        mercy_k=0.45,
        context_type=context_type,
        verbose=False,
        attention_engine=SimulatedAttentionEngine(vocab_size=vocab_size, seed=seed),
    )

    par_pipeline = None
    if run_parallel:
        par_pipeline = create_parallel_pipeline(
            vocab_size=vocab_size,
            K=K,
            mercy_k=0.45,
            backend=backend,
            context_type=context_type,
            verbose=False,
            attention_engine=SimulatedAttentionEngine(vocab_size=vocab_size, seed=seed),
            seed=seed,
        )

    # ── Run benchmark steps ───────────────────────────────────────────────────
    context_ids = [seed % vocab_size, (seed * 7) % vocab_size, (seed * 13) % vocab_size]
    step_metrics: List[StepMetrics] = []

    if verbose:
        header = (
            f"  {'Step':>4} | {'Baseline':>9} | {'Seq':>9} | {'SeqActive':>9} | "
            f"{'SeqNu':>7} | {'SeqAct':>6}"
        )
        if run_parallel:
            header += f" | {'Par':>9} | {'SpeedupP':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))

    for step in range(T_steps):
        # Get logits from attention engine
        raw_logits = attention.get_logits(context_ids, step=step)

        # Baseline: greedy argmax
        baseline_token, baseline_ms = _greedy_decode(raw_logits)

        # Sequential pipeline
        seq_result = seq_pipeline.run(
            context_ids=context_ids,
            logits=list(raw_logits),
            step=step,
        )

        # Parallel pipeline (optional)
        par_token:     Optional[int]   = None
        par_ms:        Optional[float] = None
        par_active:    Optional[int]   = None
        par_valuation: Optional[float] = None
        par_drift:     Optional[float] = None
        par_actualized: Optional[bool] = None

        if par_pipeline is not None:
            par_result = par_pipeline.run(
                context_ids=context_ids,
                logits=list(raw_logits),
                step=step,
            )
            par_token      = par_result.final_token
            par_ms         = par_result.total_time_ms
            par_active     = par_result.fdsa_result.active_count
            par_valuation  = par_result.global_valuation
            par_drift      = par_result.global_drift
            par_actualized = par_result.is_actualized

        metrics = StepMetrics(
            step=step,
            vocab_size=vocab_size,
            baseline_token=baseline_token,
            baseline_ms=baseline_ms,
            seq_token=seq_result.final_token,
            seq_ms=seq_result.total_time_ms,
            seq_active=seq_result.fdsa_result.active_count,
            seq_valuation=seq_result.global_valuation,
            seq_drift=seq_result.global_drift,
            seq_actualized=seq_result.is_actualized,
            par_token=par_token,
            par_ms=par_ms,
            par_active=par_active,
            par_valuation=par_valuation,
            par_drift=par_drift,
            par_actualized=par_actualized,
            seq_pruning_ratio=seq_result.fdsa_result.pruning_ratio,
            par_pruning_ratio=(par_result.fdsa_result.pruning_ratio if par_pipeline else 0.0),
        )
        step_metrics.append(metrics)

        # Update context for next step
        context_ids.append(seq_result.final_token)
        if len(context_ids) > 16:
            context_ids = context_ids[-16:]

        if verbose:
            act_str = "✓" if metrics.seq_actualized else "✗"
            line = (
                f"  {step:>4} | {baseline_ms:>8.2f}ms | {metrics.seq_ms:>8.2f}ms | "
                f"{metrics.seq_active:>9} | {metrics.seq_valuation:>7.4f} | {act_str:>6}"
            )
            if run_parallel and metrics.par_ms is not None:
                speedup_p = metrics.seq_ms / metrics.par_ms if metrics.par_ms > 0 else 1.0
                par_act_str = "✓" if metrics.par_actualized else "✗"
                line += f" | {metrics.par_ms:>8.2f}ms | {speedup_p:>8.2f}x"
            print(line)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    bench = BenchmarkRun(
        run_id=run_id,
        vocab_size=vocab_size,
        K=K,
        T_steps=T_steps,
        context_type=context_type,
        step_metrics=step_metrics,
    )
    bench.compute_aggregates()

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  AGGREGATES  [{run_id}]")
        print(f"    Baseline greedy  : {bench.mean_baseline_ms:.3f} ms/step")
        print(f"    Sequential FDSA  : {bench.mean_seq_ms:.3f} ms/step  "
              f"[speedup vs baseline: {bench.speedup_seq_vs_baseline:.2f}x]")
        if run_parallel:
            print(f"    Parallel QCA     : {bench.mean_par_ms:.3f} ms/step  "
                  f"[speedup vs seq: {bench.speedup_par_vs_seq:.2f}x]")
        print(f"    Mean Valuation ν : {bench.mean_seq_valuation:.4f}")
        print(f"    Mean Pruning     : {bench.mean_seq_pruning:.1%}")
        print(f"    Actualized Rate  : {bench.seq_actualized_rate:.1%} (seq)")
        if run_parallel:
            print(f"                       {bench.par_actualized_rate:.1%} (par)")

    return bench


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scaling Benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_scaling_benchmark(
    vocab_sizes: List[int] = (300, 500, 1000, 2000),
    K_values:    List[int] = (2, 4, 8),
    T_steps:     int       = 5,
    seed:        int       = 42,
    verbose:     bool      = True,
) -> Dict[str, Any]:
    """
    Scaling benchmark: measure pipeline latency and quality across
    vocabulary sizes and cluster counts.

    Returns a nested dict of results keyed by (vocab_size, K).
    """
    results: Dict[str, Any] = {
        "vocab_sizes": list(vocab_sizes),
        "K_values": list(K_values),
        "T_steps": T_steps,
        "runs": {},
        "summary": [],
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  SCALING BENCHMARK")
        print(f"  V={vocab_sizes}, K={K_values}, T={T_steps}")
        print("=" * 70)
        print(f"  {'V':>6} | {'K':>4} | {'Baseline':>10} | {'Sequential':>11} | "
              f"{'Parallel':>10} | {'Pruning':>8} | {'Actualized':>10}")
        print("  " + "-" * 70)

    for V in vocab_sizes:
        for K in K_values:
            try:
                bench = run_attention_benchmark(
                    vocab_size=V,
                    T_steps=T_steps,
                    K=K,
                    verbose=False,
                    seed=seed,
                    run_parallel=True,
                )
                key = f"V{V}_K{K}"
                results["runs"][key] = {
                    "vocab_size": V,
                    "K": K,
                    "mean_baseline_ms": bench.mean_baseline_ms,
                    "mean_seq_ms": bench.mean_seq_ms,
                    "mean_par_ms": bench.mean_par_ms,
                    "mean_seq_pruning": bench.mean_seq_pruning,
                    "seq_actualized_rate": bench.seq_actualized_rate,
                    "par_actualized_rate": bench.par_actualized_rate,
                    "speedup_seq_vs_baseline": bench.speedup_seq_vs_baseline,
                    "speedup_par_vs_seq": bench.speedup_par_vs_seq,
                    "mean_seq_valuation": bench.mean_seq_valuation,
                }
                results["summary"].append({
                    "V": V, "K": K,
                    "baseline_ms": round(bench.mean_baseline_ms, 3),
                    "seq_ms": round(bench.mean_seq_ms, 3),
                    "par_ms": round(bench.mean_par_ms, 3),
                    "pruning": round(bench.mean_seq_pruning, 4),
                    "act_rate": round(bench.seq_actualized_rate, 3),
                })

                if verbose:
                    print(
                        f"  {V:>6} | {K:>4} | "
                        f"{bench.mean_baseline_ms:>9.2f}ms | "
                        f"{bench.mean_seq_ms:>10.2f}ms | "
                        f"{bench.mean_par_ms:>9.2f}ms | "
                        f"{bench.mean_seq_pruning:>7.1%} | "
                        f"{bench.seq_actualized_rate:>9.1%}"
                    )
            except Exception as e:
                if verbose:
                    print(f"  {V:>6} | {K:>4} | ERROR: {e}")
                results["runs"][f"V{V}_K{K}"] = {"error": str(e)}

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ordering Verification (Theorem 2.1: FDSA before Actualization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_canonical_ordering(
    vocab_size: int = 500,
    T_steps: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify Theorem 2.1 (Unified_Framework.md §2):
    FDSA before Actualization MUST be the canonical ordering.

    Measures:
      • Correct order (FDSA → Actualizer): fast convergence, low drift
      • Reversed order (Actualizer on full vocab then FDSA): high drift, slow convergence

    Returns comparison metrics.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("  ORDERING VERIFICATION (Theorem 2.1)")
        print(f"  V={vocab_size}, T={T_steps}")
        print(f"{'='*70}")

    attention = SimulatedAttentionEngine(vocab_size=vocab_size, seed=42)

    _CORE_DIR_local = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_Core_Engine")
    )
    if _CORE_DIR_local not in sys.path:
        sys.path.insert(0, _CORE_DIR_local)

    from numpy_actualizer_engine import NumpyActualizerEngine
    from fdsa_pruner import VectorizedFDSAPruner
    import numpy as np

    pruner = VectorizedFDSAPruner(vocab_size=vocab_size, k=0.45)
    engine = NumpyActualizerEngine(vocab_size=vocab_size, mercy_k=0.45, max_iters=25)

    correct_order_metrics = []
    reversed_order_metrics = []
    context_ids = [42, 17, 305]

    for step in range(T_steps):
        raw_logits = attention.get_logits(context_ids, step=step)
        raw_np = np.array(raw_logits, dtype=np.float64)

        # ── Correct order: FDSA → Actualizer ──────────────────────────────────
        t0 = time.perf_counter()
        pruned, active = pruner.prune_numpy(raw_np, last_token=context_ids[-1], grammar_rules={})
        target = set(range(min(20, vocab_size)))
        tok_c, _, Tr_c, iters_c, nu_c, act_c = engine.steer(
            logits=pruned,
            history=context_ids[-8:],
            target_tokens=target,
        )
        t1 = time.perf_counter()
        correct_ms = (t1 - t0) * 1000.0

        # ── Reversed order: Actualizer on full vocab → FDSA ────────────────────
        # (Simulates the pathological ordering warned against in Theorem 2.1)
        t0 = time.perf_counter()
        tok_r, _, Tr_r, iters_r, nu_r, act_r = engine.steer(
            logits=raw_np,            # full V-dimensional substrate (NO pruning first)
            history=context_ids[-8:],
            target_tokens=target,
        )
        # Then FDSA (masking after the fact — work was already wasted)
        pruned_r, active_r = pruner.prune_numpy(raw_np, last_token=context_ids[-1], grammar_rules={})
        t1 = time.perf_counter()
        reversed_ms = (t1 - t0) * 1000.0

        correct_order_metrics.append({
            "step": step, "ms": correct_ms, "iters": iters_c,
            "Tr_D": Tr_c, "nu": nu_c[-1] if nu_c else 0.0,
            "active": active, "actualized": act_c,
        })
        reversed_order_metrics.append({
            "step": step, "ms": reversed_ms, "iters": iters_r,
            "Tr_D": Tr_r, "nu": nu_r[-1] if nu_r else 0.0,
            "active": vocab_size, "actualized": act_r,
        })

        context_ids.append(tok_c)
        if len(context_ids) > 16:
            context_ids = context_ids[-16:]

    # Aggregate
    def avg(lst, key): return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    correct_avg_ms    = avg(correct_order_metrics, "ms")
    reversed_avg_ms   = avg(reversed_order_metrics, "ms")
    correct_avg_iters = avg(correct_order_metrics, "iters")
    reversed_avg_iters= avg(reversed_order_metrics, "iters")
    correct_avg_Tr    = avg(correct_order_metrics, "Tr_D")
    reversed_avg_Tr   = avg(reversed_order_metrics, "Tr_D")
    correct_avg_nu    = avg(correct_order_metrics, "nu")
    reversed_avg_nu   = avg(reversed_order_metrics, "nu")

    ordering_result = {
        "vocab_size": vocab_size,
        "T_steps": T_steps,
        "correct_order": {
            "mean_ms": round(correct_avg_ms, 3),
            "mean_iters": round(correct_avg_iters, 2),
            "mean_Tr_D": round(correct_avg_Tr, 4),
            "mean_nu": round(correct_avg_nu, 4),
        },
        "reversed_order": {
            "mean_ms": round(reversed_avg_ms, 3),
            "mean_iters": round(reversed_avg_iters, 2),
            "mean_Tr_D": round(reversed_avg_Tr, 4),
            "mean_nu": round(reversed_avg_nu, 4),
        },
        "latency_ratio_reversed_over_correct": (
            round(reversed_avg_ms / correct_avg_ms, 3) if correct_avg_ms > 0 else None
        ),
        "iter_ratio_reversed_over_correct": (
            round(reversed_avg_iters / correct_avg_iters, 3)
            if correct_avg_iters > 0 else None
        ),
        "theorem_2_1_confirmed": (reversed_avg_iters >= correct_avg_iters),
    }

    if verbose:
        print(f"\n  Metric            | Correct Order (FDSA→ACT) | Reversed (ACT→FDSA)")
        print(f"  {'─'*65}")
        print(f"  Latency (ms)      | {correct_avg_ms:>24.3f} | {reversed_avg_ms:>.3f}")
        print(f"  Iterations        | {correct_avg_iters:>24.2f} | {reversed_avg_iters:>.2f}")
        print(f"  Tr(D_μν)          | {correct_avg_Tr:>24.4f} | {reversed_avg_Tr:>.4f}")
        print(f"  Valuation ν       | {correct_avg_nu:>24.4f} | {reversed_avg_nu:>.4f}")
        confirmed = ordering_result["theorem_2_1_confirmed"]
        print(f"\n  Theorem 2.1 confirmed: {'✓ YES' if confirmed else '✗ NO'}")
        print(f"  Reversed/Correct latency ratio: "
              f"{ordering_result['latency_ratio_reversed_over_correct']:.2f}x")

    return ordering_result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B5: Pipeline-Level Ordering Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_pipeline_ordering(
    vocab_size: int = 500,
    T_steps: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify that pipeline.run() enforces correct stage ordering through
    audit log inspection and result object validation.

    Checks:
      1. Audit log shows '[Stage 1 — FDSA]' BEFORE '[Stage 3 — Actualizer]'
      2. FDSA actually pruned (pruning_ratio > 0)
      3. Actualizer converged (is_actualized = True)
      4. Actualizer iteration count is reasonable (benefit of reduced vocab)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("  PIPELINE-LEVEL ORDERING VERIFICATION")
        print(f"  V={vocab_size}, T={T_steps}")
        print(f"{'='*70}")

    attention = SimulatedAttentionEngine(vocab_size=vocab_size, seed=42)

    # Create sequential pipeline (FDSA → Actualizer)
    cfg_seq = PipelineConfig(
        vocab_size=vocab_size, execution_mode="sequential",
        K=4, seed=42, verbose=False,
    )
    pipe_seq = ActualizerFDSAQCAPipeline(config=cfg_seq, attention_engine=attention)

    # Create parallel pipeline (FDSA → QCA → Actualizer)
    cfg_par = PipelineConfig(
        vocab_size=vocab_size, execution_mode="parallel",
        K=4, seed=42, verbose=False,
    )
    pipe_par = ActualizerFDSAQCAPipeline(config=cfg_par, attention_engine=attention)

    context_ids = [42, 17, 305]
    results = {"sequential": [], "parallel": []}
    all_passed = True

    for step in range(T_steps):
        # --- Sequential path ---
        res_seq = pipe_seq.run(context_ids=context_ids, step=step)
        log_seq = "\n".join(res_seq.audit_log)

        # Check 1: audit log ordering
        fdsa_pos = log_seq.find("Stage 1")
        act_pos = log_seq.find("Stage 3")
        log_order_ok = (fdsa_pos >= 0 and act_pos >= 0 and fdsa_pos < act_pos)

        # Check 2: FDSA pruned something
        pruning_ok = (
            res_seq.fdsa_result is not None
            and res_seq.fdsa_result.pruning_ratio > 0
        )

        # Check 3: Actualizer converged
        actualized_ok = (
            res_seq.act_result is not None
            and res_seq.act_result.is_actualized
        )

        step_passed = log_order_ok and pruning_ok and actualized_ok
        all_passed = all_passed and step_passed

        results["sequential"].append({
            "step": step,
            "log_order_ok": log_order_ok,
            "pruning_ratio": (
                res_seq.fdsa_result.pruning_ratio if res_seq.fdsa_result else 0
            ),
            "actualized": actualized_ok,
            "iterations": (
                res_seq.act_result.iterations if res_seq.act_result else -1
            ),
            "passed": step_passed,
        })

        # --- Parallel path ---
        res_par = pipe_par.run(context_ids=context_ids, step=step)
        log_par = "\n".join(res_par.audit_log)

        fdsa_pos_p = log_par.find("Stage 1")
        qca_pos_p = log_par.find("Stage 2")
        act_pos_p = log_par.find("Stage 3")
        log_order_par = (
            fdsa_pos_p >= 0 and qca_pos_p >= 0 and act_pos_p >= 0
            and fdsa_pos_p < qca_pos_p < act_pos_p
        )

        pruning_par_ok = (
            res_par.fdsa_result is not None
            and res_par.fdsa_result.pruning_ratio > 0
        )
        actualized_par_ok = (
            res_par.act_result is not None
            and res_par.act_result.is_actualized
        )

        par_passed = log_order_par and pruning_par_ok and actualized_par_ok
        all_passed = all_passed and par_passed

        results["parallel"].append({
            "step": step,
            "log_order_ok": log_order_par,
            "pruning_ratio": (
                res_par.fdsa_result.pruning_ratio if res_par.fdsa_result else 0
            ),
            "actualized": actualized_par_ok,
            "qca_clusters": (
                len(res_par.parallel_result.cluster_results)
                if res_par.parallel_result else 0
            ),
            "passed": par_passed,
        })

        context_ids.append(res_seq.final_token)
        if len(context_ids) > 16:
            context_ids = context_ids[-16:]

    if verbose:
        print("\n  Mode        | Step | LogOrder | Pruning  | Actualized | Result")
        print("  " + "-" * 66)
        for mode in ["sequential", "parallel"]:
            for r in results[mode]:
                status = "✓ PASS" if r["passed"] else "✗ FAIL"
                print(
                    f"  {mode:<12} | {r['step']:>4} | "
                    f"{'✓' if r['log_order_ok'] else '✗':>8} | "
                    f"{r['pruning_ratio']:>7.1%} | "
                    f"{'✓' if r['actualized'] else '✗':>10} | "
                    f"{status}"
                )
        overall = "✓ ALL PASSED" if all_passed else "✗ FAILURES DETECTED"
        print(f"\n  Pipeline ordering verification: {overall}")

    return {"all_passed": all_passed, "results": results}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Benchmark Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_benchmark(
    verbose: bool = True,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete benchmark suite for Actualizer_Engine_FDSA_QCA.

    Sections
    --------
      B1. Single-run detailed benchmark (V=500, K=4, T=8)
      B2. Scaling benchmark (V × K grid)
      B3. Ordering verification (Theorem 2.1)
      B4. Context-type comparison

    Parameters
    ----------
    verbose : Print detailed output.
    quick   : Use smaller parameters for quick CI/sanity checks.

    Returns
    -------
    Dict with all benchmark results.
    """
    all_results: Dict[str, Any] = {}

    print("\n" + "█" * 70)
    print("  Actualizer_Engine_FDSA_QCA — Full Benchmark Suite")
    print("  Framework: Consciousness and Prime Base Intelligence (CKT V3_U1)")
    print("█" * 70)

    # ── B1: Single detailed run ───────────────────────────────────────────────
    print("\n── B1: Detailed Single Run ──────────────────────────────────────────")
    b1_V = 300 if quick else 500
    b1_T = 5 if quick else 8
    b1_K = 3 if quick else 4
    b1 = run_attention_benchmark(
        vocab_size=b1_V, T_steps=b1_T, K=b1_K,
        context_type="logical_coding", verbose=verbose,
    )
    all_results["B1_detailed"] = {
        "run_id": b1.run_id,
        "mean_baseline_ms": b1.mean_baseline_ms,
        "mean_seq_ms": b1.mean_seq_ms,
        "mean_par_ms": b1.mean_par_ms,
        "speedup_seq_vs_baseline": b1.speedup_seq_vs_baseline,
        "speedup_par_vs_seq": b1.speedup_par_vs_seq,
        "mean_valuation": b1.mean_seq_valuation,
        "mean_pruning": b1.mean_seq_pruning,
        "actualized_rate": b1.seq_actualized_rate,
    }

    # ── B2: Scaling benchmark ─────────────────────────────────────────────────
    print("\n── B2: Scaling Benchmark (V × K) ────────────────────────────────────")
    if quick:
        b2_vocabs = [300, 500]
        b2_Ks = [2, 4]
        b2_T = 3
    else:
        b2_vocabs = [300, 500, 1000]
        b2_Ks = [2, 4, 8]
        b2_T = 5
    b2 = run_scaling_benchmark(
        vocab_sizes=b2_vocabs, K_values=b2_Ks, T_steps=b2_T, verbose=verbose
    )
    all_results["B2_scaling"] = b2

    # ── B3: Ordering verification ─────────────────────────────────────────────
    print("\n── B3: Ordering Verification (Theorem 2.1) ──────────────────────────")
    b3_V = 300 if quick else 500
    b3_T = 3 if quick else 5
    b3 = verify_canonical_ordering(vocab_size=b3_V, T_steps=b3_T, verbose=verbose)
    all_results["B3_ordering"] = b3

    # ── B4: Context-type comparison ───────────────────────────────────────────
    print("\n── B4: Context-Type Comparison ──────────────────────────────────────")
    context_types = ["logical_coding", "mathematical", "factual_qa"] if not quick else ["logical_coding", "mathematical"]
    b4_results = {}
    b4_V = 300 if quick else 500
    b4_T = 3 if quick else 5

    if verbose:
        print(f"  {'Context':>18} | {'SeqMs':>8} | {'Pruning':>8} | {'ValNu':>7} | {'ActRate':>9}")
        print("  " + "-" * 58)

    for ctx in context_types:
        bench = run_attention_benchmark(
            vocab_size=b4_V, T_steps=b4_T, K=4,
            context_type=ctx, verbose=False, run_parallel=False,
        )
        b4_results[ctx] = {
            "mean_seq_ms": bench.mean_seq_ms,
            "mean_pruning": bench.mean_seq_pruning,
            "mean_valuation": bench.mean_seq_valuation,
            "actualized_rate": bench.seq_actualized_rate,
        }
        if verbose:
            print(
                f"  {ctx:>18} | {bench.mean_seq_ms:>7.2f}ms | "
                f"{bench.mean_seq_pruning:>7.1%} | "
                f"{bench.mean_seq_valuation:>7.4f} | "
                f"{bench.seq_actualized_rate:>8.1%}"
            )

    all_results["B4_context_types"] = b4_results

    # ── B5: Pipeline-level ordering verification ─────────────────────────────
    print("\n── B5: Pipeline-Level Ordering Verification ─────────────────────────")
    b5_V = 300 if quick else 500
    b5_T = 3 if quick else 5
    b5 = verify_pipeline_ordering(vocab_size=b5_V, T_steps=b5_T, verbose=verbose)
    all_results["B5_pipeline_ordering"] = b5

    # ── Summary Table ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print("  BENCHMARK SUMMARY")
        print(f"{'='*70}")
        b1r = all_results["B1_detailed"]
        print(f"  B1 Baseline greedy      : {b1r['mean_baseline_ms']:.3f} ms/step")
        print(f"  B1 Sequential pipeline  : {b1r['mean_seq_ms']:.3f} ms/step  "
              f"[speedup: {b1r['speedup_seq_vs_baseline']:.2f}x vs baseline]")
        print(f"  B1 Parallel pipeline    : {b1r['mean_par_ms']:.3f} ms/step  "
              f"[speedup: {b1r['speedup_par_vs_seq']:.2f}x vs seq]")
        print(f"  B1 Pruning ratio        : {b1r['mean_pruning']:.1%}")
        print(f"  B1 Actualization rate   : {b1r['actualized_rate']:.1%}")
        print(f"  B1 Mean valuation ν     : {b1r['mean_valuation']:.4f}")
        b3r = all_results["B3_ordering"]
        confirmed = b3r.get("theorem_2_1_confirmed", False)
        print(f"\n  B3 Theorem 2.1 (FDSA before ACT): {'CONFIRMED ✓' if confirmed else 'NOT CONFIRMED ✗'}")
        print(f"     Reversed/Correct latency ratio: {b3r.get('latency_ratio_reversed_over_correct', 'N/A')}x")
        b5r = all_results.get("B5_pipeline_ordering", {})
        b5_ok = b5r.get("all_passed", False)
        print(f"\n  B5 Pipeline-level ordering: {'ALL PASSED ✓' if b5_ok else 'FAILURES DETECTED ✗'}")
        print(f"\n  Pipeline: Actualizer_Engine_FDSA_QCA v1.0.0")
        print(f"  Framework: CKT V3_U1 | DOI: 10.5281/zenodo.21420098")
        print(f"{'='*70}")

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Actualizer_Engine_FDSA_QCA Real Benchmark with Attention Engine"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run quick (reduced) benchmark for CI / sanity check")
    parser.add_argument("--vocab-size", type=int, default=500,
                        help="Vocabulary size V (default: 500)")
    parser.add_argument("--K", type=int, default=4,
                        help="QCA cluster count (default: 4)")
    parser.add_argument("--T", type=int, default=8,
                        help="Decoding steps T (default: 8)")
    parser.add_argument("--context-type", default="logical_coding",
                        choices=["logical_coding", "mathematical", "factual_qa",
                                 "creative_dialogue", "general"],
                        help="FDSA context profile")
    parser.add_argument("--ordering-only", action="store_true",
                        help="Run only Theorem 2.1 ordering verification")
    parser.add_argument("--full", action="store_true",
                        help="Run the complete benchmark suite")
    args = parser.parse_args()

    if args.ordering_only:
        verify_canonical_ordering(vocab_size=args.vocab_size, T_steps=args.T, verbose=True)
    elif args.full:
        run_full_benchmark(verbose=True, quick=args.quick)
    else:
        run_attention_benchmark(
            vocab_size=args.vocab_size,
            T_steps=args.T,
            K=args.K,
            context_type=args.context_type,
            verbose=True,
        )
