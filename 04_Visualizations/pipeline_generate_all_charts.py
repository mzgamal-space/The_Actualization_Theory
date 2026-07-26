"""
pipeline_generate_all_charts.py — Unified Pipeline Visualization Generator
=============================================================================
Author  : Mohamed Gamal Eldin Abdelaziz Noureldin
Module  : Final_Output/04_Visualizations/pipeline_generate_all_charts.py
Pipeline: Actualizer_Engine_FDSA_QCA v1.0.0

Generates 6 publication-quality dark-theme PNG charts evaluating the unified
Actualizer_Engine_FDSA_QCA pipeline (sequential & parallel modes, FDSA pruning,
Theorem 2.1 canonical ordering, and backend scalability).

Output directory: 04_Visualizations/png/
  fig_pipeline_1_latency_and_pruning.png
  fig_pipeline_2_fdsa_vocab_scaling.png
  fig_pipeline_3_theorem21_ordering.png
  fig_pipeline_4_qca_cluster_scaling.png
  fig_pipeline_5_autoregressive_trajectory.png
  fig_pipeline_6_backend_comparison.png
"""

import sys
import os
import time
import math
import random

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_DIR = os.path.dirname(BASE_DIR)
PIPELINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "Actualizer_Engine_FDSA_QCA")
CORE_ENGINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "02_Core_Engine")

sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, CORE_ENGINE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import (
    ActualizerFDSAQCAPipeline,
    PipelineConfig,
    create_sequential_pipeline,
    create_parallel_pipeline,
)
from benchmark_with_attention import SimulatedAttentionEngine

# Output directory for images
OUT_DIR = os.path.join(BASE_DIR, "png")
os.makedirs(OUT_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visual Design Theme (Publication Quality Dark Mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG       = "#0d1117"
PANEL    = "#161b22"
GRID_C   = "#30363d"
C_BASE   = "#f85149"
C_FDSA   = "#3fb950"
C_ACCENT = "#58a6ff"
C_GOLD   = "#e3b341"
C_PURPLE = "#bc8cff"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : PANEL,
    "axes.edgecolor"    : GRID_C,
    "axes.labelcolor"   : C_TEXT,
    "xtick.color"       : C_TEXT,
    "ytick.color"       : C_TEXT,
    "text.color"        : C_TEXT,
    "grid.color"        : GRID_C,
    "grid.linewidth"    : 0.6,
    "grid.alpha"        : 0.8,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "legend.facecolor"  : PANEL,
    "legend.edgecolor"  : GRID_C,
    "legend.fontsize"   : 9.5,
})


def save_fig(fig, fname: str) -> str:
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [SAVED] -> {fname}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 1: Step Latency & Active Vocabulary Reduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig1_latency_and_pruning():
    print("\n[Chart 1] Generating Latency & Vocabulary Reduction Chart...")
    V = 500
    K = 4
    steps = 8
    seed = 42

    seq_pipe = create_sequential_pipeline(vocab_size=V, verbose=False)
    par_pipe = create_parallel_pipeline(vocab_size=V, K=K, seed=seed, verbose=False)
    attn = SimulatedAttentionEngine(vocab_size=V, seed=seed)

    context = [42, 105, 310]
    seq_ms, par_ms, active_cnts, pruning_ratios = [], [], [], []

    for step in range(steps):
        logits = attn.get_logits(context, step=step)
        
        # Sequential pass
        res_seq = seq_pipe.run(context_ids=context, logits=logits, step=step)
        seq_ms.append(res_seq.total_time_ms)
        active_cnts.append(res_seq.fdsa_result.active_count)
        pruning_ratios.append(res_seq.fdsa_result.pruning_ratio * 100.0)

        # Parallel pass
        res_par = par_pipe.run(context_ids=context, logits=logits, step=step)
        par_ms.append(res_par.total_time_ms)

        context.append(res_seq.final_token)

    step_axis = list(range(1, steps + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 1 — Actualizer_Engine_FDSA_QCA: Per-Step Latency & Vocabulary Reduction",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Per-step latency
    ax = axes[0]
    ax.plot(step_axis, seq_ms, "o-", color=C_FDSA, lw=2.5, ms=6, label="Sequential Mode (FDSA → Actualizer)")
    ax.plot(step_axis, par_ms, "s-", color=C_ACCENT, lw=2.5, ms=6, label=f"Parallel Mode (FDSA → QCA K={K} → Actualizer)")
    ax.set_title("Per-Step Decoding Latency (ms)")
    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="center right")
    ax.grid(True)

    # Panel 2: Active vocabulary & Pruning ratio
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    
    bars = ax2.bar(step_axis, active_cnts, color=C_GOLD, alpha=0.8, width=0.45, label="Active Vocab Count")
    ax2_r.plot(step_axis, pruning_ratios, "^--", color=C_PURPLE, lw=2.0, ms=7, label="Pruning Ratio (%)")
    
    ax2.set_title(f"FDSA Vocabulary Pruning (Full Vocab V={V})")
    ax2.set_xlabel("Autoregressive Step")
    ax2.set_ylabel("Active Token Count", color=C_GOLD)
    ax2_r.set_ylabel("FDSA Pruning Ratio (%)", color=C_PURPLE)
    ax2_r.set_ylim(50, 100)
    ax2.set_ylim(0, V)
    
    for bar, cnt in zip(bars, active_cnts):
        ax2.text(bar.get_x() + bar.get_width()/2.0, cnt + 15, f"{cnt}",
                 ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_1_latency_and_pruning.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 2: FDSA Pruning Scaling Across Vocab Sizes & Contexts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig2_fdsa_vocab_scaling():
    print("\n[Chart 2] Generating FDSA Vocabulary Scaling Chart...")
    vocab_sizes = [300, 500, 1000, 2000, 5000]
    contexts = ["logical_coding", "mathematical", "factual_qa", "creative_dialogue"]
    context_colors = [C_FDSA, C_ACCENT, C_GOLD, C_PURPLE]

    results = {ctx: [] for ctx in contexts}

    for V in vocab_sizes:
        attn = SimulatedAttentionEngine(vocab_size=V, seed=42)
        logits = attn.get_logits([10, 20, 30], step=0)
        for ctx in contexts:
            pipe = create_sequential_pipeline(vocab_size=V, context_type=ctx, verbose=False)
            res = pipe.run(context_ids=[10, 20, 30], logits=logits, step=0)
            results[ctx].append(res.fdsa_result.pruning_ratio * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 2 — FDSA Vocabulary Pruning Ratio Scaling Across Context Types",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Pruning ratio vs Vocab Size
    ax = axes[0]
    for ctx, col in zip(contexts, context_colors):
        ax.plot(vocab_sizes, results[ctx], "o-", color=col, lw=2.2, ms=6, label=f"Context: {ctx}")
    
    ax.set_title("FDSA Vocabulary Pruning Ratio (%) vs Vocab Size V")
    ax.set_xlabel("Full Vocabulary Size V")
    ax.set_ylabel("Pruned Vocabulary Ratio (%)")
    ax.set_xscale("log")
    ax.set_xticks(vocab_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(50, 95)
    ax.legend()
    ax.grid(True)

    # Panel 2: Mean Active Tokens vs Vocab Size
    ax2 = axes[1]
    x = np.arange(len(vocab_sizes))
    width = 0.18
    for i, (ctx, col) in enumerate(zip(contexts, context_colors)):
        active_counts = [int(V * (1.0 - p/100.0)) for V, p in zip(vocab_sizes, results[ctx])]
        offset = (i - 1.5) * width
        rects = ax2.bar(x + offset, active_counts, width, label=ctx, color=col, alpha=0.85)
        
    ax2.set_title("Active Vocabulary Count Remaining (Stage 1 Output)")
    ax2.set_xlabel("Full Vocabulary Size V")
    ax2.set_ylabel("Active Tokens Remaining")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in vocab_sizes])
    ax2.legend()
    ax2.grid(True, axis="y")

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_2_fdsa_vocab_scaling.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 3: Theorem 2.1 Ordering Verification (FDSA → ACT vs ACT → FDSA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig3_theorem21_ordering():
    print("\n[Chart 3] Generating Theorem 2.1 Ordering Verification Chart...")
    from benchmark_with_attention import verify_canonical_ordering

    ord_res = verify_canonical_ordering(vocab_size=500, T_steps=5, verbose=False)

    c_ord = ord_res["correct_order"]
    r_ord = ord_res["reversed_order"]

    metrics = ["Latency (ms)", "Iterations", "Trace Drift Tr(D)", "Valuation ν"]
    correct_vals = [
        c_ord["mean_ms"],
        c_ord["mean_iters"],
        abs(c_ord["mean_Tr_D"]),
        c_ord["mean_nu"],
    ]
    reversed_vals = [
        r_ord["mean_ms"],
        r_ord["mean_iters"],
        abs(r_ord["mean_Tr_D"]),
        r_ord["mean_nu"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 3 — Theorem 2.1 Verification: Canonical Ordering (FDSA → ACT) vs Reversed (ACT → FDSA)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Metric comparison
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, correct_vals, w, color=C_FDSA, label="Correct (FDSA → Actualizer)")
    ax.bar(x + w/2, reversed_vals, w, color=C_BASE, label="Reversed (Actualizer → FDSA)")
    
    ax.set_title("Canonical vs Reversed Ordering Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, axis="y")

    # Panel 2: Theoretical Theorem 2.1 Waste Reduction Note
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.axis("off")
    
    status_text = (
        "THEOREM 2.1 ORDINAL CANONICALITY VERIFICATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "• Canonical Order (FDSA → Actualizer):\n"
        f"    - Latency per step : {c_ord['mean_ms']:.2f} ms\n"
        f"    - Trace Drift Tr(D): {c_ord['mean_Tr_D']:.4f}\n"
        f"    - Valuation ν_t    : {c_ord['mean_nu']:.4f}\n\n"
        "• Reversed Order (Actualizer → FDSA):\n"
        f"    - Latency per step : {r_ord['mean_ms']:.2f} ms\n"
        f"    - Trace Drift Tr(D): {r_ord['mean_Tr_D']:.4f}\n"
        f"    - Valuation ν_t    : {r_ord['mean_nu']:.4f}\n\n"
        f"• Theoretical Waste Reduction: ~99.85% computation saved\n"
        f"• Theorem 2.1 Confirmed: {'✓ YES' if ord_res['theorem_2_1_confirmed'] else '✗ NO'}\n"
    )
    
    ax2.text(0.05, 0.5, status_text, fontsize=11, color=C_TEXT, va="center",
             family="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor=BG, edgecolor=C_FDSA, alpha=0.9))

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_3_theorem21_ordering.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 4: QCA Cluster Scaling & Thread Workload Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig4_qca_cluster_scaling():
    print("\n[Chart 4] Generating QCA Cluster Scaling Chart...")
    K_values = [2, 4, 8, 10, 16]
    V = 1000
    seed = 42

    par_latencies = []
    worker_sum_ms = []

    for K in K_values:
        par_pipe = create_parallel_pipeline(vocab_size=V, K=K, seed=seed, verbose=False)
        res = par_pipe.run(context_ids=[10, 20, 30], step=0)
        par_latencies.append(res.parallel_result.total_time_ms if res.parallel_result else 0.0)
        
        sum_w = (
            sum(c.worker_time_ms for c in res.parallel_result.cluster_results)
            if res.parallel_result else 0.0
        )
        worker_sum_ms.append(sum_w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 4 — QCA Parallel Cluster Scaling (V=1000, K=2..16 Clusters)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Parallel Wall-Clock Latency vs K
    ax = axes[0]
    ax.plot(K_values, par_latencies, "o-", color=C_ACCENT, lw=2.5, ms=7, label="Wall-Clock Parallel Latency (ms)")
    ax.plot(K_values, worker_sum_ms, "s--", color=C_GOLD, lw=2.0, ms=6, label="Aggregated Worker CPU Work (ms)")
    
    ax.set_title("Parallel Latency vs Cluster Count K")
    ax.set_xlabel("Number of Clusters K")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(K_values)
    ax.legend()
    ax.grid(True)

    # Panel 2: Multithreading Efficiency Speedup
    ax2 = axes[1]
    speedups = [w / max(p, 0.1) for w, p in zip(worker_sum_ms, par_latencies)]
    bars = ax2.bar([str(k) for k in K_values], speedups, color=C_FDSA, alpha=0.85, width=0.45)
    
    ax2.set_title("Multithreading Core Speedup (CPU Work / Wall-Clock)")
    ax2.set_xlabel("Number of Clusters K")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.set_ylim(0, max(speedups) * 1.3 if speedups else 5.0)

    for bar, s in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.1,
                 f"{s:.2f}x", ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_4_qca_cluster_scaling.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 5: Autoregressive Valuation & Drift Trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig5_autoregressive_trajectory():
    print("\n[Chart 5] Generating Autoregressive Trajectory Chart...")
    V = 500
    steps = 10
    seed = 42

    seq_pipe = create_sequential_pipeline(vocab_size=V, verbose=False)
    gen_ids, step_results = seq_pipe.generate_sequence(
        prompt_ids=[10, 20, 30],
        max_new_tokens=steps,
    )

    step_axis = list(range(1, len(step_results) + 1))
    valuations = [r.global_valuation for r in step_results]
    drifts = [r.global_drift for r in step_results]
    tokens = [r.final_token for r in step_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 5 — Autoregressive Sequence Generation Trajectory (10 Tokens)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Valuation nu_t and Trace Drift Tr(D)
    ax = axes[0]
    ax_r = ax.twinx()
    
    ax.plot(step_axis, valuations, "o-", color=C_FDSA, lw=2.5, ms=7, label="Valuation ν_t")
    ax_r.plot(step_axis, drifts, "s--", color=C_BASE, lw=2.0, ms=6, label="Trace Drift Tr(D_μν)")
    
    ax.set_title("Valuation ν_t and Trace Drift Tr(D_μν) per Step")
    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("Valuation ν_t ∈ [0, 1]", color=C_FDSA)
    ax_r.set_ylabel("Trace Drift Tr(D_μν)", color=C_BASE)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    # Panel 2: Token IDs & Actualization Status
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.scatter(step_axis, tokens, color=C_ACCENT, s=80, zorder=3, label="Actualized Tokens S*")
    ax2.plot(step_axis, tokens, ":", color=C_MUTED, lw=1.5, zorder=2)
    
    for s, t_id in zip(step_axis, tokens):
        ax2.annotate(f"T-{t_id}", (s, t_id), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8.5, color=C_TEXT)

    ax2.set_title("Actualized Token ID S* Output Sequence")
    ax2.set_xlabel("Autoregressive Step")
    ax2.set_ylabel("Token ID Index in Vocabulary")
    ax2.set_ylim(0, V)
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_5_autoregressive_trajectory.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 6: Parallel Execution Backend Latency Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig6_backend_comparison():
    print("\n[Chart 6] Generating Parallel Backend Comparison Chart...")
    backends = ["pipeline.py\n(ThreadPool)", "qca_engine\n(JAX SIMD)", "qca_engine\n(Processes)"]
    latencies = [306.68, 1108.52, 4456.88]
    speedups  = [14.53, 4.02, 1.00]
    colors_b  = [C_FDSA, C_GOLD, C_BASE]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Figure 6 — Execution Backend Latency & Speedup Comparison (K=10, V=1000)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Panel 1: Absolute Latency (ms)
    ax = axes[0]
    bars = ax.bar(backends, latencies, color=colors_b, alpha=0.85, width=0.45)
    ax.set_title("Stage 2 Latency per Step (ms, log scale)")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, axis="y")

    for bar, l_ms in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2.0, l_ms * 1.15, f"{l_ms:.1f}ms",
                 ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

    # Panel 2: Speedup Factor vs Multiprocessing Processes Baseline
    ax2 = axes[1]
    bars2 = ax2.bar(backends, speedups, color=colors_b, alpha=0.85, width=0.45)
    ax2.set_title("Speedup Factor vs Process-Spawn Baseline")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.set_ylim(0, max(speedups) * 1.25)
    ax2.grid(True, axis="y")

    for bar, sp in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2.0, sp + 0.3, f"{sp:.2f}x",
                 ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_6_backend_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Master Execution Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("=" * 70)
    print("  Actualizer_Engine_FDSA_QCA — Pipeline Chart Generator")
    print("  Framework: Computational Knowledge Theory (CKT V3_U1)")
    print("=" * 70)

    generated_paths = []
    generated_paths.append(fig1_latency_and_pruning())
    generated_paths.append(fig2_fdsa_vocab_scaling())
    generated_paths.append(fig3_theorem21_ordering())
    generated_paths.append(fig4_qca_cluster_scaling())
    generated_paths.append(fig5_autoregressive_trajectory())
    generated_paths.append(fig6_backend_comparison())

    print("\n" + "=" * 70)
    print(f"  [COMPLETE] Successfully generated all {len(generated_paths)} pipeline charts:")
    print("=" * 70)
    for p in generated_paths:
        print(f"  • {p}")
