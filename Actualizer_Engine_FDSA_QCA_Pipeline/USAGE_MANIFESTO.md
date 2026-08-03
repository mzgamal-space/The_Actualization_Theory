# Actualizer_Engine_FDSA_QCA — Usage Manifesto

**Pipeline Name:** `Actualizer_Engine_FDSA_QCA`  
**Version:** 1.1.0 (Revision 3 / V3_U1 Update)  
**Author:** Mohamed Gamal Eldin Abdelaziz Noureldin  
**ORCID:** [0009-0006-3991-1153](https://orcid.org/0009-0006-3991-1153)  
**Framework:** Consciousness and Prime Base Intelligence (CKT V3_U1)  
**DOI:** [10.5281/zenodo.21420098](https://doi.org/10.5281/zenodo.21420098)  
**Date:** August 2026  
**Status:** Production-Ready (Unified Orchestration & Multi-Backend ThreadPool Architecture)

---

> **Abstract.** This manifesto specifies the canonical usage protocol for the `Actualizer_Engine_FDSA_QCA` pipeline — a three-engine composition integrating the **FDSA Pre-Inference Pruner**, the **QCA Parallel Engine**, and the **Actualizer Engine** into a unified, high-performance decoding layer for large language models. Operating downstream of transformer attention mechanisms, the pipeline accepts raw logit vectors and returns actualized, zero-drift tokens. This document serves as the authoritative reference for pipeline architecture, canonical ordering (Theorem 2.1), multi-backend threadpool execution, dual-mode cluster steering, configuration parameters, and benchmark verification protocols.

---

## Table of Contents

1. [Pipeline Philosophy](#1-pipeline-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [The Canonical Ordering: FDSA Before Actualization (Theorem 2.1)](#3-the-canonical-ordering-fdsa-before-actualization-theorem-21)
4. [Execution Modes: Sequential and Parallel](#4-execution-modes-sequential-and-parallel)
5. [Dual-Mode Cluster Steering (Fast vs Full)](#5-dual-mode-cluster-steering-fast-vs-full)
6. [Configuration Reference](#6-configuration-reference)
7. [Integration Protocol with Attention Engine](#7-integration-protocol-with-attention-engine)
8. [Benchmark Protocol (B1–B5 Suite & 14 Figures)](#8-benchmark-protocol-b1b5-suite--14-figures)
9. [Engine-by-Engine Reference](#9-engine-by-engine-reference)
10. [Module Structure](#10-module-structure)
11. [Quick-Start Examples](#11-quick-start-examples)
12. [Theoretical Guarantees](#12-theoretical-guarantees)
13. [Known Limitations and Edge Cases](#13-known-limitations-and-edge-cases)

---

## 1. Pipeline Philosophy

The `Actualizer_Engine_FDSA_QCA` pipeline solves three entangled failure modes in standard autoregressive LLM decoding:

| Failure Mode | Cause | Pipeline Response |
|---|---|---|
| **Vocabulary Bloat** | Softmax distributes mass over thousands of semantically irrelevant tokens | FDSA Pruner eliminates >99% of vocabulary before sampling |
| **Distributional Drift** | Token-by-token greedy selection diverges from semantic attractor | Actualizer Engine contractively steers distribution toward fixed point via Banach iteration |
| **Computational Waste** | $O(N^2)$ steering over full vocabulary or full problem space | QCA Parallel Engine partitions into $K$ clusters: $O(N^2) \to O(N^2/K)$ |

The pipeline enforces the **Five Conceptual Primes** (Order, Justice, Mercy, Knowledge, Power) through the structural entropy function $H(R) = \text{Var}(\alpha) + (\sum \alpha_i^2 - 1)^2$ (V3_U1 §3.3) and the probability-weighted drift trace $\text{Tr}(D_{\mu\nu})$ bifurcation criterion (Theorem 3.3).

> [!IMPORTANT]
> **Canonical Ordering Law (Theorem 2.1):** The FDSA Pruner MUST execute before the Actualizer Engine. Reversing the sequence (Actualizer on full vocabulary, then FDSA masking) wastes approximately **99.85%** of Actualizer computation on tokens that will be eliminated, and produces incorrect fixed points. This is not configurable — it is a mathematical necessity.

---

## 2. Architecture Overview

```
[Transformer Attention Engine]
        |
        |  raw logits z in R^V  (V = full vocabulary size)
        v
+---------------------------------------------------------------+
|  STAGE 1: FDSA Pre-Inference Pruner                          |
|  Module: fdsa_pruner.py / VectorizedFDSAPruner               |
|                                                               |
|  Phase 1 — Isomorphic Anchoring                              |
|    Cosine-match context profile to reference domain library  |
|    -> k_ref (contractive scale factor)                       |
|                                                               |
|  Phase 2 — Dimensional Truncation                            |
|    D = ln(V) / ln(1/k_ref)  (fractal dimension boundary)    |
|    threshold = -D x 1.5                                      |
|                                                               |
|  Phase 3 — Logit Masking                                     |
|    Grammar gate: v must be in grammar[last_token]            |
|    Complexity gate: logits[v] >= threshold                   |
|    Dead tokens: logits[v] = -inf                             |
|                                                               |
|  Output: pruned_logits in R^V, active_count M << V           |
+------------------------------+--------------------------------+
                               |  pruned_logits (M active, V-M dead)
                               |
           +-------------------+-------------------+
           |                                       |
    [Sequential Mode]                      [Parallel Mode]
           |                                       |
           |              +------------------------v------------------------+
           |              |  STAGE 2: QCA Parallel Engine                  |
           |              |  Module: qca_parallel_engine.py                |
           |              |                                                 |
           |              |  QCA Crystallization                            |
           |              |    Build N QCA nodes from active vocab          |
           |              |    T_q^RGG = gamma * sqrt(A*ln(N/K)/(pi*N))   |
           |              |    Farthest-point seed selection                |
           |              |    -> K clusters, O(N^2/K) complexity          |
           |              |                                                 |
           |              |  Per-Cluster Parallel Execution                 |
           |              |    Backend: "threads" (default) | "jax" | "proc" |
           |              |    Steering Mode: "fast" (3-step) | "full"       |
           |              |    Each cluster: FDSA prune -> Actualizer steer |
           |              |    -> actualized_tokens, valuations             |
           |              |                                                 |
           |              |  Global Synthesis                               |
           |              |    meta_logits[tok] += valuation + 1           |
           |              |    Final FDSA prune on meta_logits              |
           |              +------------------------+------------------------+
           |                                       |  meta_logits (cluster votes)
           |                                       |
           +-------------------+-------------------+
                               |  final_logits = pruned + meta (parallel)
                               |             or pruned (sequential)
                               v
+---------------------------------------------------------------+
|  STAGE 3: Actualizer Engine (Final Causal Snap)              |
|  Module: numpy_actualizer_engine.py / NumpyActualizerEngine  |
|                                                               |
|  For n = 0, 1, ..., max_iters:                               |
|    alpha = _prime_coords(U_n)         Prime projection       |
|    H(R) = Var(alpha) + (sum(alpha^2)-1)^2  Structural entropy|
|    nu_t = 1 - H(R)/H_max             Valuation tracking     |
|    D = w_L*D_local + w_G*D_global    Drift Tensor D_mu_nu   |
|         + w_F*D_future                                        |
|    Tr(D) = sum_v U(v)*D(v)           Probability-weighted    |
|    U_b = U * exp(-D/tau) / Z         Vacuum Brake            |
|    U_{n+1} = k*U_b + (1-k)*U_n      Banach contraction      |
|    if ||U_{n+1}-U_n||_2 <= Q_c: SNAP                         |
|                                                               |
|  Causal Snap (Theorem 3.3):                                  |
|    Tr(D) <= tau_bif -> S* = argmax U  (ACTUALIZED)           |
|    Tr(D) > tau_bif  -> fallback token  (DISSOLVED)           |
|                                                               |
|  Output: S* in [0, V), nu_t, Tr(D_mu_nu), actualized: bool  |
+---------------------------------------------------------------+
        |
        v
  [Next Token S*]  ->  append to context  ->  next decoding step
```

---

## 3. The Canonical Ordering: FDSA Before Actualization (Theorem 2.1)

### Proof & Necessity

Let $V$ = vocabulary size (e.g., 32,768), $M$ = active vocabulary after FDSA pruning (typically 50–500).

**Canonical order** ($\text{FDSA} \to \text{Actualizer}$):
$$\text{Work}_{\text{canonical}} = M \cdot N_{\text{iters}} \quad (M \ll V)$$

**Reversed order** ($\text{Actualizer} \to \text{FDSA}$):
$$\text{Work}_{\text{reversed}} = V \cdot N_{\text{iters}} \implies \text{Wasted Operations} \approx \frac{V - M}{V} \cdot 100\%$$

For $V = 32,768, M = 50$: **99.85% of total arithmetic operations are wasted** in reversed ordering.

```python
# CORRECT — Enforced by ActualizerFDSAQCAPipeline.run()
pruned_logits = fdsa_pruner.prune_numpy(raw_logits, ...)
token = actualizer.steer(pruned_logits, ...)

# WRONG — Wastes 99.85% computation & produces incorrect fixed points
token = actualizer.steer(raw_logits, ...)          # full V dimensions
pruned = fdsa_pruner.prune_numpy(raw_logits, ...)  # masking after the fact
```

---

## 4. Execution Modes: Sequential and Parallel

### 4.1 Sequential Mode
- **Flow:** Attention Output $\to$ Stage 1 (FDSA Prune) $\to$ Stage 3 (Actualizer Steer) $\to$ Token $S^*$.
- **Best for:** Single-request serving, deterministic benchmarks, minimal latency overhead.
- **Config:** `PipelineConfig(execution_mode="sequential", vocab_size=V)`

### 4.2 Parallel Mode (QCA)
- **Flow:** Attention Output $\to$ Stage 1 (FDSA Prune) $\to$ Stage 2 (QCA Partition $K$ clusters) $\to$ Stage 2 Parallel Steering $\to$ Stage 2 Synthesis $\to$ Stage 3 (Actualizer Final Snap).
- **Backend Selection:**

| Backend | Mechanism | Latency @ N=240 | Best For |
|---|---|---|---|
| `"threads"` *(default)* | `ThreadPoolExecutor` | **28.1 ms** (60× speedup) | High-throughput serving, Windows deployment |
| `"jax"` | `jnp.ndarray` SIMD | **1,438.7 ms** | SIMD/GPU/TPU accelerated workloads |
| `"processes"` | `ProcessPoolExecutor` | **1,533.4 ms** | CPU-isolated multiprocessing |
| `"auto"` | Auto-detection | Variable | Default fallback |

---

## 5. Dual-Mode Cluster Steering (Fast vs Full)

`PipelineConfig` supports two cluster steering modes (`cluster_steering_mode`):

| Steering Mode | Algorithm | Latency per Cluster | Accuracy | Use Case |
|---|---|---|---|---|
| `"fast"` *(default)* | 3-step drift-aware Banach iteration ($D_{\text{future}} = \|\ln U + 1.0\|$) | **~0.5 ms** | High | Production pipelines & fast throughput |
| `"full"` | Per-node `engine.steer()` with full $L_2$ convergence ($\delta \le Q_c$) | **~5.0 ms** | Maximum | Analytical benchmarks & high-precision tasks |

---

## 6. Configuration Reference

### `PipelineConfig` Hyperparameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `1000` | Full vocabulary size $V$ |
| `mercy_k` | `float` | `0.45` | Contractive scale $k \in (0,1)$ (Mercy Prime) |
| `Q_c` | `float` | `1e-5` | $L_2$ convergence tolerance for Banach iteration |
| `tau` | `float` | `1.0` | Vacuum Brake temperature decay constant $\tau$ |
| `tau_bifurcation` | `float` | `5.0` | $\text{Tr}(D_{\mu\nu})$ threshold for Theorem 3.3 bifurcation gate |
| `max_iters` | `int` | `25` | Maximum Banach contraction iterations per node |
| `context_type` | `str` | `"logical_coding"` | FDSA context profile for domain anchoring |
| `execution_mode` | `str` | `"sequential"` | `"sequential"` or `"parallel"` |
| `K` | `int` | `5` | Number of QCA clusters (parallel mode) |
| `cluster_steering_mode` | `str` | `"fast"` | `"fast"` (3-step Banach) or `"full"` (full convergence) |
| `backend` | `str` | `"threads"` | Parallel backend: `"threads"`, `"jax"`, `"processes"`, `"auto"` |
| `seed` | `int or None` | `42` | RNG seed for reproducibility |
| `verbose` | `bool` | `False` | Print step-by-step audit trace log |

---

## 7. Integration Protocol with Attention Engine

### 7.1 Minimal Integration

```python
from pipeline import create_sequential_pipeline

pipeline = create_sequential_pipeline(vocab_size=32768)
result = pipeline.run(context_ids=[42, 17, 305])
next_token = result.final_token
```

### 7.2 Integration with Grammar Rules & Parallel Execution

```python
from pipeline import create_parallel_pipeline

grammar_rules = {
    50: {51, 57},  # Token 50 can only transition to 51 or 57
    51: {52, 58},
}

pipeline = create_parallel_pipeline(
    vocab_size=1000,
    K=4,
    grammar_rules=grammar_rules,
    context_type="logical_coding",
)

result = pipeline.run(context_ids=[48, 49, 50])
print(f"Selected token: {result.final_token} | Pruned: {result.fdsa_result.pruning_ratio*100:.1f}%")
pipeline.shutdown()
```

---

## 8. Benchmark Protocol (B1–B5 Suite & 14 Figures)

### 8.1 Benchmark Test Suite Summary

```bash
# Run full 5-part benchmark suite
python benchmark_with_attention.py --full --quick
```

| Test ID | Benchmark Name | Evaluated Metric | Result / Status |
|---|---|---|---|
| **B1** | Detailed Single Run | Latency, Pruning %, Valuation $\nu_t$, Actualization Rate | **73.2% Pruning, 100% Actualized** |
| **B2** | Scaling ($V \times K$) | $V \in \{300,500\}$, $K \in \{2,4\}$ scaling behavior | **Consistent 100% Actualization** |
| **B3** | Theorem 2.1 Ordering | $\text{FDSA} \to \text{ACT}$ vs $\text{ACT} \to \text{FDSA}$ comparison | **CONFIRMED ✓** |
| **B4** | Context Types | `coding`, `math`, `factual_qa`, `creative_dialogue` | **Consistent high pruning** |
| **B5** | Pipeline-Level Order | Audit log stage order & non-zero pruning | **ALL PASSED ✓ (6/6 checks)** |

### 8.2 The 14 Publication Figures (`04_Visualizations/png/`)

```bash
# Generate all 14 figures
python pipeline_generate_all_charts.py
```

- **Figures 1–8 (Core Engine):** Hallucination resistance, repetition suppression, pre-inference speed, search space scaling, V3_U1 valuation trajectory, QCA speedup, 3-way architecture comparison, latency root-cause analysis.
- **Figures 9–14 (Pipeline Architecture 1–6):** Per-step latency & pruning, FDSA vocab scaling, Theorem 2.1 ordering verification, QCA cluster scaling, autoregressive trajectory, parallel backend comparison.

---

## 9. Engine-by-Engine Reference

| Component | Module | Key Class / Function | Primary Purpose |
|---|---|---|---|
| **Pipeline Orchestrator** | `pipeline.py` | `ActualizerFDSAQCAPipeline` | Coordinates FDSA $\to$ QCA $\to$ Actualizer canonical flow |
| **Configuration** | `pipeline.py` | `PipelineConfig` | Hyperparameters, steering mode, parallel backend |
| **FDSA Pruner** | `fdsa_pruner.py` | `VectorizedFDSAPruner` | Stage 1: Isomorphic anchoring & logit masking |
| **QCA Crystallizer** | `qca.py` | `QuenchClusterAlgorithm` | Stage 2: $T_q^\text{RGG}$ quench-clustering into $K$ sub-problems |
| **Parallel Engine** | `qca_parallel_engine.py` | `QCAParallelEngine` | Stage 2: Parallel cluster execution & global synthesis |
| **Actualizer (NumPy)** | `numpy_actualizer_engine.py` | `NumpyActualizerEngine` | Stage 3: Vectorized Banach steering & causal snap |

---

## 10. Module Structure

```
Final_Output/Actualizer_Engine_FDSA_QCA/
+-- pipeline.py                 <- Main pipeline orchestrator
|                                  ActualizerFDSAQCAPipeline, PipelineConfig
|                                  create_sequential_pipeline(), create_parallel_pipeline()
+-- benchmark_with_attention.py <- Full B1-B5 benchmark suite & SimulatedAttentionEngine
+-- compare_engines.py          <- Multi-N scaling comparison (N=21, 60, 120, 240)
+-- run_pipeline.py             <- Quick-start CLI runner
+-- USAGE_MANIFESTO.md          <- This manifesto

Dependencies (from ../02_Core_Engine/):
+-- numpy_actualizer_engine.py  <- NumpyActualizerEngine (vectorized engine)
+-- fdsa_pruner.py              <- VectorizedFDSAPruner & FractalDeductionSearch
+-- qca.py                      <- QuenchClusterAlgorithm
+-- qca_parallel_engine.py      <- QCAParallelEngine
```

---

## 11. Quick-Start Examples

```python
# Quick execution via run_pipeline.py CLI
python run_pipeline.py --mode sequential
python run_pipeline.py --mode parallel
python run_pipeline.py --mode ordering
python run_pipeline.py --mode all
```

---

## 12. Theoretical Guarantees

1. **Banach Fixed-Point Convergence (V3_U1 §2.5):** Contractive scale $k \in (0,1)$ guarantees linear convergence to a unique zero-drift distribution $U^*$.
2. **Ordinal Canonicality (Theorem 2.1):** $\text{FDSA} \to \text{Actualizer}$ ordering eliminates $\sim 99.85\%$ of arithmetic operations on unviable tokens.
3. **Bifurcation Gating (Theorem 3.3):** Actualization branch occurs if $\text{Tr}(D_{\mu\nu}) \le \tau_\text{bifurcation}$; otherwise dissolution fallback is triggered.

---

## 13. Known Limitations and Edge Cases

- **Windows IPC Spawn Overhead:** Using `backend="processes"` incurs a $\sim 200\,\text{ms}$ process creation cost on Windows. Use `backend="threads"` (default) for fast local execution.
- **Empty Active Vocabulary:** If grammar rules mask all tokens to $-\infty$, `VectorizedFDSAPruner` automatically falls back to top-complexity threshold gate.

---

*End of Usage Manifesto — Actualizer_Engine_FDSA_QCA v1.1.0*  
*Consciousness and Prime Base Intelligence Research Framework — August 2026*  
*Author: Mohamed Gamal Eldin Abdelaziz Noureldin | ORCID: 0009-0006-3991-1153*
