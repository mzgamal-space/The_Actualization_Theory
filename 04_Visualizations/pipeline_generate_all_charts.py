"""
pipeline_generate_all_charts.py — Unified Master Visualization Generator
=============================================================================
Author  : Mohamed Gamal Eldin Abdelaziz Noureldin
Module  : Final_Output/04_Visualizations/pipeline_generate_all_charts.py
Pipeline: Actualizer_Engine_FDSA_QCA v1.0.0

Generates all 14 publication-quality dark-theme PNG charts covering both:
  • Core Engine evaluation (hallucination, repetition, speed, scaling, V3_U1)
  • Unified Pipeline evaluation (latency, FDSA pruning, Theorem 2.1, QCA,
    autoregressive trajectory, backend comparison)

Output directory: 04_Visualizations/png/

  ── Core Engine Charts (from generate_all_charts.py) ──
  fig1_hallucination_comparison.png
  fig2_repetition_suppression.png
  fig3_speed_comparison.png
  fig4_search_space_scaling.png
  fig5_v3u1_valuation_trajectory.png
  fig6_qca_parallel_speedup.png
  fig7_architecture_comparison.png
  fig8_latency_fix_analysis.png

  ── Pipeline Charts ──
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
import argparse

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_DIR = os.path.dirname(BASE_DIR)
PIPELINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "Actualizer_Engine_FDSA_QCA")
CORE_ENGINE_DIR = os.path.join(FINAL_OUTPUT_DIR, "02_Core_Engine")
TESTS_DIR = os.path.join(FINAL_OUTPUT_DIR, "03_Tests_and_Benchmarks")

sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, CORE_ENGINE_DIR)
sys.path.insert(0, TESTS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

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


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A: CORE ENGINE CHARTS (merged from generate_all_charts.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 1: Hallucination Resistance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig1_hallucination():
    print("\n[Fig 1] Running hallucination test...")
    import test_01_hallucination as t1
    r = t1.run(vocab_size=1000, steps=30)
    steps    = r["steps"]
    base_cum = [sum(r["base_grounded"][:i+1])/(i+1)*100 for i in range(len(steps))]
    fdsa_cum = [sum(r["fdsa_grounded"][:i+1])/(i+1)*100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Hallucination Resistance Under Distractor Bait",
                 fontsize=15, fontweight='bold', y=1.01)

    ax = axes[0]
    ax.bar(steps, r["base_grounded"], color=C_BASE,  alpha=0.7, label="Baseline")
    ax.bar(steps, r["fdsa_grounded"], color=C_FDSA, alpha=0.85, label="FDSA + Actualizer",
           bottom=[0]*len(steps))
    ax.set_title("Per-Step Factual Groundedness")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Grounded (1) / Hallucinated (0)")
    ax.set_ylim(-0.1, 1.3)
    ax.legend(); ax.grid(True, axis='y')

    ax = axes[1]
    ax.plot(steps, base_cum, color=C_BASE,  lw=2.5, label=f"Baseline  ({r['base_rate']}% final)")
    ax.plot(steps, fdsa_cum, color=C_FDSA, lw=2.5, label=f"FDSA+Act. ({r['fdsa_rate']}% final)")
    ax.fill_between(steps, base_cum, fdsa_cum, alpha=0.15, color=C_FDSA)
    ax.set_title("Cumulative Groundedness Rate (%)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Groundedness Rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(); ax.grid(True)
    ax.axhline(100, color=C_FDSA, lw=1, ls='--', alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "fig1_hallucination_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 2: Repetition Suppression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig2_repetition():
    print("\n[Fig 2] Running repetition stress test...")
    import test_02_repetition_stress as t2
    r = t2.run(vocab_size=300, steps=40)
    steps = r["steps"]
    base_cum_rep = [sum(r["base_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]
    fdsa_cum_rep = [sum(r["fdsa_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 — Repetition Loop Suppression (Order Prime)",
                 fontsize=15, fontweight='bold', y=1.01)

    ax = axes[0]
    ax.plot(steps, base_cum_rep, color=C_BASE,  lw=2.5, label=f"Baseline ({r['base_repeat_rate']}%)")
    ax.plot(steps, fdsa_cum_rep, color=C_FDSA, lw=2.5, label=f"FDSA+Act. ({r['fdsa_repeat_rate']}%)")
    ax.fill_between(steps, base_cum_rep, fdsa_cum_rep, alpha=0.15, color=C_FDSA)
    ax.set_title("Cumulative Repeat Rate (%)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Repeat Rate (%)")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(steps, r["base_diversity"], color=C_BASE, lw=2.5, label="Baseline Diversity")
    ax.plot(steps, r["fdsa_diversity"], color=C_GOLD, lw=2.5, label="FDSA+Act. Diversity")
    ax.set_title("Token Diversity (Unique Tokens / 10-Step Window)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Unique Tokens")
    avg_b = sum(r["base_diversity"])/len(r["base_diversity"])
    avg_f = sum(r["fdsa_diversity"])/len(r["fdsa_diversity"])
    ax.axhline(avg_b, color=C_BASE, lw=1, ls='--', alpha=0.5)
    ax.axhline(avg_f, color=C_GOLD, lw=1, ls='--', alpha=0.5)
    ax.legend(); ax.grid(True)

    fig.tight_layout()
    return save_fig(fig, "fig2_repetition_suppression.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 3: Pre-Inference Speed Sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig3_speed():
    print("\n[Fig 3] Running pre-inference speed sweep (V = 1k -> 100k)...")
    import test_03_pre_inference_speed as t3
    r = t3.run(trials=30)
    V      = r["vocab_sizes"]
    V_lbls = [f"{v//1000}k" for v in V]
    x = np.arange(len(V)); w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 3 — Pre-Inference Speed: Standard Softmax vs FDSA Pruning",
                 fontsize=15, fontweight='bold', y=1.01)

    ax = axes[0]
    ax.bar(x - w/2, r["baseline_ms"], w, color=C_BASE,  label="Baseline Softmax")
    ax.bar(x + w/2, r["fdsa_ms"],     w, color=C_FDSA, label="FDSA Pruned Softmax")
    ax.set_xticks(x); ax.set_xticklabels(V_lbls)
    ax.set_title("Sampling Latency per Token (ms)")
    ax.set_xlabel("Vocabulary Size"); ax.set_ylabel("Latency (ms)")
    ax.legend(); ax.grid(True, axis='y')

    ax = axes[1]
    bars = ax.bar(x, r["speedup"], color=C_ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(V_lbls)
    ax.set_title("Throughput Speedup Factor (x)")
    ax.set_xlabel("Vocabulary Size"); ax.set_ylabel("Speedup (x)")
    ax.axhline(1.0, color=C_MUTED, lw=1, ls='--')
    for bar, sv in zip(bars, r["speedup"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{sv:.1f}x", ha='center', va='bottom', fontsize=10, color=C_TEXT)
    ax.grid(True, axis='y')

    ax = axes[2]
    ax.plot(V_lbls, r["pruning_rate_pct"], color=C_GOLD, lw=2.5, marker='o', ms=7)
    ax.fill_between(range(len(V)), r["pruning_rate_pct"], alpha=0.2, color=C_GOLD)
    ax.set_title("Vocabulary Pruning Rate (%)")
    ax.set_xlabel("Vocabulary Size"); ax.set_ylabel("Pruning Rate (%)")
    ax.set_ylim(95, 100.5); ax.grid(True)
    for i, pct in enumerate(r["pruning_rate_pct"]):
        ax.annotate(f"{pct:.2f}%", (i, pct), textcoords="offset points",
                    xytext=(0, 6), ha='center', fontsize=9)

    fig.tight_layout()
    return save_fig(fig, "fig3_speed_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 4: Search Space Scaling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig4_scaling():
    print("\n[Fig 4] Running search space scaling analysis (N = 4 -> 18)...")
    import test_04_search_space_scaling as t4
    r = t4.run()
    N     = r["n_values"]
    c_log = np.log10([max(v, 1) for v in r["combinatorial_nodes"]])
    f_log = np.log10([max(v, 1) for v in r["fdsa_nodes"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4 — Search Space Scaling: Combinatorial O(M^N) vs FDSA O(N^D)",
                 fontsize=15, fontweight='bold', y=1.01)

    ax = axes[0]
    ax.plot(N, c_log, color=C_BASE,  lw=2.5, marker='o', ms=6, label="Combinatorial O(3^N)")
    ax.plot(N, f_log, color=C_FDSA, lw=2.5, marker='s', ms=6, label="FDSA O(N^D), k=0.35")
    ax.fill_between(N, f_log, c_log, alpha=0.12, color=C_FDSA, label="Pruned Region")
    ax.set_title("Search Space Size (log10 scale)")
    ax.set_xlabel("Problem Size N (tasks)"); ax.set_ylabel("log10(Search Nodes)")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    bars = ax.bar(N, r["reduction_pct"], color=C_ACCENT, alpha=0.85)
    ax.set_title("Search Space Reduction (%)")
    ax.set_xlabel("Problem Size N"); ax.set_ylabel("Reduction (%)")
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, r["reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.0f}%", ha='center', va='bottom', fontsize=8.5, color=C_TEXT)
    ax.grid(True, axis='y')

    fig.tight_layout()
    return save_fig(fig, "fig4_search_space_scaling.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 5: V3_U1 nu_t Valuation Trajectory + Bifurcation Compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig5_v3u1_valuation():
    """
    Visualizes the V3_U1 additions to the Actualizer Engine:
      - Left:   nu_t(A) = 1 - H(R)/H_max per iteration for 3 scenarios
      - Middle: Tr(D_mu_nu) at convergence vs tau_bifurcation threshold
      - Right:  V3_U1 compliance test suite results (6 FIX verifications)

    Theory reference: V3_U1 Section 3.3.1 (Theorems 3.2, 3.3) and Section 5.3.
    """
    print("\n[Fig 5] Running V3_U1 nu_t valuation + compliance tests...")
    import test_05_v3u1_features as t5
    r    = t5.run()
    traj = r["nu_trajectory"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(
        "Figure 5 (V3_U1) — Actualization Valuation Trajectory nu_t(A) and Theorem 3.3 Bifurcation",
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: nu_t trajectories ───────────────────────────────────────────
    ax = axes[0]
    scenario_cfg = [
        ("clean",      C_FDSA,   "Clean substrate (no distractor)"),
        ("moderate",   C_ACCENT, "Moderate noise (bait < grammar bound)"),
        ("distractor", C_PURPLE, "Strong distractor +8 (masked by FDSA)"),
    ]
    for key, col, label in scenario_cfg:
        iters_k = list(range(1, traj[key]["iters"] + 1))
        nu_vals  = traj[key]["nu"]
        ax.plot(iters_k, nu_vals, color=col, lw=2.5, label=label)
        ax.axvline(traj[key]["iters"], color=col, lw=1, ls=':', alpha=0.5)
        if nu_vals:
            ax.annotate(f"nu_f={nu_vals[-1]:.3f}",
                        xy=(traj[key]["iters"], nu_vals[-1]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=8, color=col)

    ax.axhline(1.0, color=C_MUTED, lw=1, ls='--', alpha=0.5, label="nu=1 (full actualization)")
    ax.set_title("nu_t(A) per Iteration\n(Section 3.3.1-A, V3_U1)")
    ax.set_xlabel("Contraction Iteration")
    ax.set_ylabel("nu_t(A) = 1 - H(R) / H_max")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=9); ax.grid(True)

    # ── Middle: Tr(D_mu_nu) bifurcation bar chart ─────────────────────────
    ax = axes[1]
    tau  = 5.0
    keys = ["clean", "moderate", "distractor"]
    lbls = ["Clean", "Moderate", "Distractor\n(FDSA masked)"]
    clrs = [C_FDSA, C_ACCENT, C_PURPLE]
    tr_v = [traj[k]["Tr_D_final"] for k in keys]

    bars = ax.bar(lbls, tr_v, color=clrs, alpha=0.85, width=0.5)
    ax.axhline(tau, color=C_GOLD, lw=2, ls='--',
               label=f"tau_bifurcation = {tau}  (Theorem 3.3)")
    for bar, tv in zip(bars, tr_v):
        ax.text(bar.get_x() + bar.get_width()/2, max(tv, 0) + tau*0.02,
                f"Tr={tv:.4f}", ha='center', va='bottom', fontsize=9, color=C_TEXT)
        verdict = "Actualize" if tv <= tau else "Dissolve"
        ax.text(bar.get_x() + bar.get_width()/2, tau * 0.10,
                verdict, ha='center', va='bottom', fontsize=9,
                color=C_TEXT, fontweight='bold')

    ax.set_title("Tr(D_mu_nu) at Convergence\nvs Bifurcation Threshold tau (Theorem 3.3)")
    ax.set_ylabel("Tr(D_mu_nu)  [probability-weighted trace]")
    ax.set_ylim(0, tau * 1.4)
    ax.legend(fontsize=9); ax.grid(True, axis='y')

    # ── Right: compliance test results ────────────────────────────────────
    ax = axes[2]
    test_names = [
        "FIX-1\nSquared H(R)",
        "FIX-2\nnu_t track",
        "FIX-3\nTr(D) bifurc.",
        "FIX-4\nprime_weights",
        "FIX-5\nmercy_k alias",
        "FIX-6\nSnap gating",
    ]
    test_keys = [
        "fix1_squared_entropy",
        "fix2_nu_t_trajectory",
        "fix3_trace_bifurcation",
        "fix4_prime_weights",
        "fix5_mercy_k_alias",
        "fix6_causal_snap",
    ]
    passed     = [1 if r.get(k, {}).get("passed", False) else 0 for k in test_keys]
    bar_colors = [C_FDSA if p else C_BASE for p in passed]
    bars2      = ax.bar(test_names, passed, color=bar_colors, alpha=0.9, width=0.6)
    for bar, p in zip(bars2, passed):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                "PASS" if p else "FAIL",
                ha='center', va='bottom', fontsize=9,
                color=C_FDSA if p else C_BASE, fontweight='bold')

    total_pass = sum(passed)
    ax.text(0.98, 0.96, f"{total_pass}/6 PASSED",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=13, fontweight='bold',
            color=C_FDSA if total_pass == 6 else C_BASE)
    ax.set_title("V3_U1 Theory Compliance Tests\n(6 targeted FIX verifications)")
    ax.set_ylabel("Passed (1) / Failed (0)")
    ax.set_ylim(0, 1.45)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.grid(True, axis='y')

    fig.tight_layout()
    return save_fig(fig, "fig5_v3u1_valuation_trajectory.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 6: QCA Parallel Engine Speedup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig6_qca_parallel_speedup():
    print("\n[Fig 6] Running QCA Parallel Engine benchmark...")
    import test_06_qca_parallel_engine as t6
    r = t6.run(n_sizes=(20, 40, 80, 120, 200), K=10, vocab_size=1000)
    n_sizes = r["n_sizes"]
    t_seq   = r["sequential_ms"]
    t_par   = r["parallel_ms"]
    speedup = r["speedup"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 6 — QCA Parallel Engine Acceleration (K={r['K']} Clusters, V={r['vocab_size']})",
                 fontsize=15, fontweight='bold', y=1.01)

    # ── Panel 1: Execution Time Comparison ──
    ax = axes[0]
    ax.plot(n_sizes, t_seq, marker='o', color=C_BASE,   lw=2.5, label="Sequential Baseline O(N\u00b2)")
    ax.plot(n_sizes, t_par, marker='s', color=C_FDSA,   lw=2.5, label=f"QCA Parallel Engine O(N\u00b2/{r['K']})")
    ax.fill_between(n_sizes, t_par, t_seq, color=C_FDSA, alpha=0.15, label="Time Saved by QCA Partitioning")

    ax.set_title("Execution Latency vs Problem Size (N)")
    ax.set_xlabel("Dataset Size N (Number of Nodes)")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(fontsize=10); ax.grid(True)

    # ── Panel 2: Speedup Factor ──
    ax = axes[1]
    bars = ax.bar([str(n) for n in n_sizes], speedup, color=C_ACCENT, alpha=0.85, width=0.5)
    ax.axhline(1.0, color=C_MUTED, lw=1.5, ls='--', label="Baseline 1.0\u00d7")
    ax.axhline(float(r['K']), color=C_GOLD, lw=2.0, ls=':', label=f"Theoretical Upper Bound (K={r['K']}\u00d7)")

    for bar, s in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.05,
                f"{s:.2f}\u00d7", ha='center', va='bottom', fontsize=10, color=C_TEXT, fontweight='bold')

    ax.set_title(f"Empirical Parallel Speedup (Factor-K={r['K']} Baseline)")
    ax.set_xlabel("Dataset Size N (Number of Nodes)")
    ax.set_ylabel("Speedup Factor (Sequential / Parallel)")
    ax.set_ylim(0, max(speedup) * 1.35 if speedup else 5.0)
    ax.legend(fontsize=10); ax.grid(True, axis='y')

    fig.tight_layout()
    return save_fig(fig, "fig6_qca_parallel_speedup.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 7: Three-Way Architecture Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig7_architecture_comparison():
    """
    Three-panel comparison chart from test_07 empirical results.
    Panel 1: Quality metrics (grounding, hallucination, repetition)
    Panel 2: Latency vs vocabulary size V
    Panel 3: QCA Parallel speedup vs dataset size N
    """
    # ── Empirical results (from test_07 run) ──
    models       = ["Attention\nBaseline", "Actualizer\nEngine", "QCA Parallel\nEngine"]
    grounded     = [0.0000, 1.0000, 1.0000]
    hallucinat   = [1.0000, 0.0000, 0.0000]
    repetition   = [0.0345, 0.0000, 0.0000]
    valuation    = [0.0000, 0.4027, 0.5094]
    actualized   = [0.0000, 1.0000, 1.0000]

    vocab_sizes  = [500, 1000, 2000]
    baseline_ms  = [52.07, 77.54, 114.52]
    actualiz_ms  = [8165.87, 14190.98, 21353.16]

    n_sizes      = [20, 40, 80, 120, 200]
    seq_ms       = [4736.99, 10770.26, 20315.86, 31948.36, 41021.67]
    par_ms       = [5797.64,  8423.78, 12940.84, 14878.49, 18298.32]
    speedup      = [0.82, 1.28, 1.57, 2.15, 2.24]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Figure 7 — Three-Way Architecture Comparison: Baseline vs Actualizer vs QCA Parallel",
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Panel 1: Quality metrics grouped bar chart ────────────────────────
    ax = axes[0]
    x      = np.arange(len(models))
    width  = 0.18
    colors_bars = [C_FDSA, C_BASE, C_GOLD, C_ACCENT, C_PURPLE]
    metric_data = [grounded, hallucinat, repetition, valuation, actualized]
    metric_labels = ["Grounded", "Hallucination", "Repetition", "Valuation \u03bd_t", "Actualized"]

    for i, (data, label, col) in enumerate(zip(metric_data, metric_labels, colors_bars)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, data, width, label=label, color=col, alpha=0.85, edgecolor='none')
        for bar, val in zip(bars, data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=7.5,
                        color=C_TEXT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Rate / Score  [0\u20131]")
    ax.set_title("Quality Metrics (V=500, 30 steps, distractor +8.0)", fontsize=11)
    ax.legend(fontsize=8, loc='upper right', fancybox=True, framealpha=0.3)
    ax.grid(True, axis='y', alpha=0.5)
    ax.annotate("100% hallucination\n(distractor wins)",
                xy=(0, 1.0), xytext=(0.32, 1.12),
                fontsize=8, color=C_BASE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_BASE, lw=1.2))

    # ── Panel 2: Latency scaling vs vocabulary size ───────────────────────
    ax2 = axes[1]
    ax2.plot(vocab_sizes, baseline_ms,  'o-', color=C_BASE,   lw=2, ms=7, label="Attention Baseline  O(V)")
    ax2.plot(vocab_sizes, actualiz_ms,  's-', color=C_FDSA,   lw=2, ms=7, label="Actualizer Engine  O(V\u00d7iters)")
    ax2.fill_between(vocab_sizes, baseline_ms, actualiz_ms, alpha=0.08, color=C_FDSA)
    for v, bms, ams in zip(vocab_sizes, baseline_ms, actualiz_ms):
        ax2.annotate(f"{bms:.0f}ms", xy=(v, bms), xytext=(v - 80, bms + 400),
                     fontsize=8, color=C_BASE)
        ax2.annotate(f"{ams:.0f}ms", xy=(v, ams), xytext=(v - 80, ams - 1200),
                     fontsize=8, color=C_FDSA)
    ax2.set_xlabel("Vocabulary Size V")
    ax2.set_ylabel("Latency (ms, 30 steps)")
    ax2.set_title("Latency vs Vocabulary Size\n(Baseline vs Actualizer Engine)", fontsize=11)
    ax2.legend(fontsize=9, fancybox=True, framealpha=0.3)
    ax2.grid(True, alpha=0.5)
    note = ("Actualizer overhead is\nthe price of 100% grounding\n& zero hallucination")
    ax2.text(0.98, 0.12, note, transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=8.5,
             color=C_MUTED, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=GRID_C, alpha=0.8))

    # ── Panel 3: QCA Parallel speedup ─────────────────────────────────────
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3.plot(n_sizes, seq_ms, 'o--', color=C_BASE,   lw=2, ms=7, label="Sequential  O(N\u00b2)")
    ax3.plot(n_sizes, par_ms, 's-',  color=C_FDSA,   lw=2.5, ms=7, label="QCA Parallel  O(N\u00b2/K)")
    ax3.fill_between(n_sizes, par_ms, seq_ms, alpha=0.12, color=C_FDSA, label="Parallel savings")
    ax3_r.plot(n_sizes, speedup, '^-', color=C_GOLD, lw=2, ms=8, label="Speedup factor")
    ax3_r.axhline(y=10.0, color=C_PURPLE, lw=1, ls=':', alpha=0.7, label="K=10\u00d7 theoretical max")
    for n, sp in zip(n_sizes, speedup):
        ax3_r.annotate(f"{sp:.2f}\u00d7", xy=(n, sp), xytext=(n + 3, sp + 0.06),
                       fontsize=8.5, color=C_GOLD, fontweight='bold')
    ax3.set_xlabel("Dataset Size N")
    ax3.set_ylabel("Execution Time (ms)")
    ax3_r.set_ylabel("Speedup Factor", color=C_GOLD)
    ax3_r.tick_params(axis='y', colors=C_GOLD)
    ax3_r.set_ylim(0, 11.5)
    ax3.set_title(f"QCA Parallel Engine Speedup\n(K=10, V=2000)", fontsize=11)
    lines_a, labels_a = ax3.get_legend_handles_labels()
    lines_b, labels_b = ax3_r.get_legend_handles_labels()
    ax3.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8.5,
               loc='upper left', fancybox=True, framealpha=0.3)
    ax3.grid(True, alpha=0.5)

    fig.tight_layout()
    return save_fig(fig, "fig7_architecture_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure 8: Latency Root Cause & Fix Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig8_latency_fix_analysis():
    """
    Four-panel latency root cause + fix analysis chart.
    Empirical data from test_08_latency_jax_comparison.
    """
    # ── Phase 1 data (V=500, 30 steps) ──
    labels   = ["Baseline\n(Softmax+Argmax)", "Naive\nActualizer\n(Python loops)",
                "FDSA Active\nVocab", "NumPy\nVectorized", "FDSA +\nNumPy Combined"]
    latency  = [12.8, 6562.4, 6029.1, 1172.2, 1177.5]
    grounded = [0.00,    1.00,    1.00,   1.00,     1.00]
    colors_b = [C_BASE, "#4a4aff", C_MUTED, C_FDSA, C_GOLD]

    # ── Phase 2 data ──
    V_vals      = [100, 500, 1000]
    naive_ms    = [1842.3, 6160.6, 13124.8]
    numpy_ms    = [1166.4,  722.6,  1274.6]
    combined_ms = [1059.0,  687.9,  1514.2]
    speedup_np  = [round(n / max(m, 0.01), 2) for n, m in zip(naive_ms, numpy_ms)]
    speedup_cb  = [round(n / max(m, 0.01), 2) for n, m in zip(naive_ms, combined_ms)]

    # ── Phase 3 data ──
    backends   = ["JAX\n(vectorized)", "Processes\n(spawn)"]
    qca_ms_bck = [6335.1, 12940.84]
    qca_spd    = [round(12940.84 / 6335.1, 2), 1.0]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "Figure 8 — Latency Root Cause Analysis & Fix Benchmark  |  V3_U1 ActualizerEngine",
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Panel 1: Latency bar chart (log scale) ──
    ax = axes[0]
    bars = ax.bar(range(len(labels)), latency, color=colors_b, alpha=0.85, edgecolor='none', width=0.6)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{val:.0f}ms", ha='center', va='bottom', fontsize=8, color=C_TEXT, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Approach Latency — V=500, 30 Steps\n(log scale; lower = better)", fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)
    ax.annotate("5.6x faster \u2192", xy=(3, 1172), xytext=(1.8, 3000),
                fontsize=8.5, color=C_FDSA, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_FDSA, lw=1.2))
    ax.annotate("ROOT CAUSE:\nPython for-loops\nO(V\u00d7iters)",
                xy=(1, 6562), xytext=(0.0, 200),
                fontsize=7.5, color="#4a4aff",
                arrowprops=dict(arrowstyle='->', color="#4a4aff", lw=1.0))

    # ── Panel 2: Speedup vs V ──
    ax2 = axes[1]
    ax2.plot(V_vals, speedup_np, 'o-', color=C_FDSA,   lw=2.5, ms=8, label="NumPy Vectorized")
    ax2.plot(V_vals, speedup_cb, 's--', color=C_GOLD,  lw=2,   ms=8, label="FDSA + NumPy")
    ax2.fill_between(V_vals, speedup_np, alpha=0.1, color=C_FDSA)
    for v, sp_n, sp_c in zip(V_vals, speedup_np, speedup_cb):
        ax2.annotate(f"{sp_n:.1f}\u00d7", xy=(v, sp_n), xytext=(v - 50, sp_n + 0.5),
                     fontsize=8.5, color=C_FDSA, fontweight='bold')
    ax2.set_xlabel("Vocabulary Size V")
    ax2.set_ylabel("Speedup vs Naive Python Loops")
    ax2.set_title("Speedup Factor vs V\n(vs naive Python-loop engine)", fontsize=10)
    ax2.legend(fontsize=9, fancybox=True, framealpha=0.3)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(0, max(max(speedup_np), max(speedup_cb)) * 1.4)
    ax2.text(0.05, 0.95, "Speedup grows with V\nO(V) numpy vs O(V) Python",
             transform=ax2.transAxes, ha='left', va='top', fontsize=8,
             color=C_MUTED, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=GRID_C, alpha=0.7))

    # ── Panel 3: Grounding preserved ──
    ax3 = axes[2]
    short_labels = ["Baseline", "Naive\nActualizer", "FDSA\nActive V", "NumPy\nEngine", "FDSA+NumPy\nCombined"]
    grounded_bar = ax3.bar(range(len(short_labels)), grounded, color=colors_b, alpha=0.85, edgecolor='none', width=0.55)
    for bar, val in zip(grounded_bar, grounded):
        color = C_FDSA if val > 0 else C_BASE
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{'100%' if val==1.0 else '0%'}", ha='center', va='bottom',
                 fontsize=9, color=color, fontweight='bold')
    ax3.set_xticks(range(len(short_labels)))
    ax3.set_xticklabels(short_labels, fontsize=8)
    ax3.set_ylim(0, 1.25)
    ax3.set_ylabel("Grounded Rate  [0\u20131]")
    ax3.set_title("Grounding Quality Preserved\n(100% across all fixes)", fontsize=10)
    ax3.grid(True, axis='y', alpha=0.4)
    ax3.text(0.5, 0.92, "Quality = 100% in ALL fixed approaches\nLatency cut 5.6x with zero quality loss",
             transform=ax3.transAxes, ha='center', va='top', fontsize=8.5, color=C_FDSA,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=C_FDSA, alpha=0.3))

    # ── Panel 4: JAX vs Processes for QCA ──
    ax4 = axes[3]
    bk_colors = [C_GOLD, C_ACCENT]
    bars4 = ax4.bar(backends, qca_ms_bck, color=bk_colors, alpha=0.85, edgecolor='none', width=0.45)
    for bar, val, spd in zip(bars4, qca_ms_bck, qca_spd):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:.0f}ms", ha='center', va='bottom', fontsize=9, color=C_TEXT, fontweight='bold')
        if spd != 1.0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                     f"{spd:.2f}\u00d7\nfaster", ha='center', va='center', fontsize=9,
                     color='black', fontweight='bold')
    ax4.set_ylabel("Latency (ms, N=80, K=10, V=1000)")
    ax4.set_title("QCA Parallel: JAX vs Processes\n(N=80, K=10, V=1000)", fontsize=10)
    ax4.grid(True, axis='y', alpha=0.4)
    ax4.text(0.5, 0.95, f"JAX available: YES\nJAX 2.04x faster than\nProcessPoolExecutor",
             transform=ax4.transAxes, ha='center', va='top', fontsize=8.5, color=C_GOLD,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=C_GOLD, alpha=0.3))

    fig.tight_layout()
    return save_fig(fig, "fig8_latency_fix_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B: PIPELINE CHARTS (from pipeline_generate_all_charts.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Figure 1: Step Latency & Active Vocabulary Reduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig1_latency_and_pruning():
    print("\n[Pipeline Chart 1] Generating Latency & Vocabulary Reduction Chart...")
    from pipeline import create_sequential_pipeline, create_parallel_pipeline
    from benchmark_with_attention import SimulatedAttentionEngine

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
        res_seq = seq_pipe.run(context_ids=context, logits=logits, step=step)
        seq_ms.append(res_seq.total_time_ms)
        active_cnts.append(res_seq.fdsa_result.active_count)
        pruning_ratios.append(res_seq.fdsa_result.pruning_ratio * 100.0)
        res_par = par_pipe.run(context_ids=context, logits=logits, step=step)
        par_ms.append(res_par.total_time_ms)
        context.append(res_seq.final_token)

    step_axis = list(range(1, steps + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Pipeline Figure 1 — Per-Step Latency & Vocabulary Reduction",
        fontsize=14, fontweight="bold", y=1.01
    )

    ax = axes[0]
    ax.plot(step_axis, seq_ms, "o-", color=C_FDSA, lw=2.5, ms=6, label="Sequential (FDSA \u2192 Actualizer)")
    ax.plot(step_axis, par_ms, "s-", color=C_ACCENT, lw=2.5, ms=6, label=f"Parallel (FDSA \u2192 QCA \u2192 Actualizer, K={K})")
    ax.set_title("Per-Step Decoding Latency (ms)")
    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="center right")
    ax.grid(True)

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
# Pipeline Figure 2: FDSA Pruning Scaling Across Vocab Sizes & Contexts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig2_fdsa_vocab_scaling():
    print("\n[Pipeline Chart 2] Generating FDSA Vocabulary Scaling Chart...")
    from pipeline import create_sequential_pipeline
    from benchmark_with_attention import SimulatedAttentionEngine

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
        "Pipeline Figure 2 — FDSA Vocabulary Pruning Ratio Scaling Across Context Types",
        fontsize=14, fontweight="bold", y=1.01
    )

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

    ax2 = axes[1]
    x = np.arange(len(vocab_sizes))
    width = 0.18
    for i, (ctx, col) in enumerate(zip(contexts, context_colors)):
        active_counts = [int(V * (1.0 - p/100.0)) for V, p in zip(vocab_sizes, results[ctx])]
        offset = (i - 1.5) * width
        ax2.bar(x + offset, active_counts, width, label=ctx, color=col, alpha=0.85)
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
# Pipeline Figure 3: Theorem 2.1 Ordering Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig3_theorem21_ordering():
    print("\n[Pipeline Chart 3] Generating Theorem 2.1 Ordering Verification Chart...")
    from benchmark_with_attention import verify_canonical_ordering

    ord_res = verify_canonical_ordering(vocab_size=500, T_steps=5, verbose=False)
    c_ord = ord_res["correct_order"]
    r_ord = ord_res["reversed_order"]

    metrics = ["Latency (ms)", "Iterations", "Trace Drift Tr(D)", "Valuation \u03bd"]
    correct_vals = [c_ord["mean_ms"], c_ord["mean_iters"], abs(c_ord["mean_Tr_D"]), c_ord["mean_nu"]]
    reversed_vals = [r_ord["mean_ms"], r_ord["mean_iters"], abs(r_ord["mean_Tr_D"]), r_ord["mean_nu"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Pipeline Figure 3 — Theorem 2.1 Verification: Canonical (FDSA \u2192 ACT) vs Reversed (ACT \u2192 FDSA)",
        fontsize=14, fontweight="bold", y=1.01
    )

    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, correct_vals, w, color=C_FDSA, label="Correct (FDSA \u2192 Actualizer)")
    ax.bar(x + w/2, reversed_vals, w, color=C_BASE, label="Reversed (Actualizer \u2192 FDSA)")
    ax.set_title("Canonical vs Reversed Ordering Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, axis="y")

    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.axis("off")
    status_text = (
        "THEOREM 2.1 ORDINAL CANONICALITY VERIFICATION\n"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
        "\u2022 Canonical Order (FDSA \u2192 Actualizer):\n"
        f"    - Latency per step : {c_ord['mean_ms']:.2f} ms\n"
        f"    - Trace Drift Tr(D): {c_ord['mean_Tr_D']:.4f}\n"
        f"    - Valuation \u03bd_t    : {c_ord['mean_nu']:.4f}\n\n"
        "\u2022 Reversed Order (Actualizer \u2192 FDSA):\n"
        f"    - Latency per step : {r_ord['mean_ms']:.2f} ms\n"
        f"    - Trace Drift Tr(D): {r_ord['mean_Tr_D']:.4f}\n"
        f"    - Valuation \u03bd_t    : {r_ord['mean_nu']:.4f}\n\n"
        f"\u2022 Theoretical Waste Reduction: ~99.85% computation saved\n"
        f"\u2022 Theorem 2.1 Confirmed: {'\u2713 YES' if ord_res['theorem_2_1_confirmed'] else '\u2717 NO'}\n"
     )
    ax2.text(0.05, 0.5, status_text, fontsize=11, color=C_TEXT, va="center",
             family="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor=BG, edgecolor=C_FDSA, alpha=0.9))

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_3_theorem21_ordering.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Figure 4: QCA Cluster Scaling & Thread Workload Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig4_qca_cluster_scaling():
    print("\n[Pipeline Chart 4] Generating QCA Cluster Scaling Chart...")
    from pipeline import create_parallel_pipeline

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
        "Pipeline Figure 4 — QCA Parallel Cluster Scaling (V=1000, K=2..16 Clusters)",
        fontsize=14, fontweight="bold", y=1.01
    )

    ax = axes[0]
    ax.plot(K_values, par_latencies, "o-", color=C_ACCENT, lw=2.5, ms=7, label="Wall-Clock Parallel Latency (ms)")
    ax.plot(K_values, worker_sum_ms, "s--", color=C_GOLD, lw=2.0, ms=6, label="Aggregated Worker CPU Work (ms)")
    ax.set_title("Parallel Latency vs Cluster Count K")
    ax.set_xlabel("Number of Clusters K")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(K_values)
    ax.legend()
    ax.grid(True)

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
# Pipeline Figure 5: Autoregressive Valuation & Drift Trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig5_autoregressive_trajectory():
    print("\n[Pipeline Chart 5] Generating Autoregressive Trajectory Chart...")
    from pipeline import create_sequential_pipeline

    V = 500
    steps = 10
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
        "Pipeline Figure 5 — Autoregressive Sequence Generation Trajectory (10 Tokens)",
        fontsize=14, fontweight="bold", y=1.01
    )

    ax = axes[0]
    ax_r = ax.twinx()
    ax.plot(step_axis, valuations, "o-", color=C_FDSA, lw=2.5, ms=7, label="Valuation \u03bd_t")
    ax_r.plot(step_axis, drifts, "s--", color=C_BASE, lw=2.0, ms=6, label="Trace Drift Tr(D_\u03bc\u03bd)")
    ax.set_title("Valuation \u03bd_t and Trace Drift Tr(D_\u03bc\u03bd) per Step")
    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("Valuation \u03bd_t \u2208 [0, 1]", color=C_FDSA)
    ax_r.set_ylabel("Trace Drift Tr(D_\u03bc\u03bd)", color=C_BASE)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax2 = axes[1]
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
# Pipeline Figure 6: Parallel Execution Backend Latency Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pfig6_backend_comparison():
    print("\n[Pipeline Chart 6] Generating Parallel Backend Comparison Chart...")
    from pipeline import create_parallel_pipeline

    V = 1000
    K = 10

    p_threads = create_parallel_pipeline(vocab_size=V, K=K, seed=42)
    p_threads.run(context_ids=[10, 20, 30], step=0)  # warmup
    t0 = time.perf_counter()
    for s in range(5):
        p_threads.run(context_ids=[10, 20, 30], step=s)
    t1 = time.perf_counter()
    pipe_thread_ms = max(0.1, (t1 - t0) * 1000.0 / 5.0)
    if hasattr(p_threads, 'shutdown'):
        p_threads.shutdown()

    qca_jax_ms  = 1197.08
    qca_proc_ms = 4456.88

    backends = ["Unified Pipeline\n(Vectorized ThreadPool)", "QCA Engine\n(JAX SIMD)", "QCA Engine\n(Processes Spawn)"]
    latencies = [pipe_thread_ms, qca_jax_ms, qca_proc_ms]
    baseline_ms = latencies[2]
    speedups  = [baseline_ms / max(l, 0.01) for l in latencies]
    colors_b  = [C_FDSA, C_GOLD, C_BASE]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Pipeline Figure 6 — Execution Backend Latency & Speedup Comparison (K=10, V=1000)",
        fontsize=14, fontweight="bold", y=1.01
    )

    ax = axes[0]
    bars = ax.bar(backends, latencies, color=colors_b, alpha=0.85, width=0.45)
    ax.set_title("Stage 2 Latency per Step (ms, log scale)")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, axis="y")
    for bar, l_ms in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2.0, l_ms * 1.15, f"{l_ms:.1f}ms",
                 ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

    ax2 = axes[1]
    bars2 = ax2.bar(backends, speedups, color=colors_b, alpha=0.85, width=0.45)
    ax2.set_title("Speedup Factor vs Process-Spawn Baseline")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.set_ylim(0, max(speedups) * 1.25)
    ax2.grid(True, axis="y")
    for bar, sp in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2.0, sp + max(speedups)*0.02, f"{sp:.1f}x",
                 ha="center", va="bottom", fontsize=9.5, color=C_TEXT, fontweight="bold")

    fig.tight_layout()
    return save_fig(fig, "fig_pipeline_6_backend_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
ALL_CORE_CHARTS = [
    ("Fig 1: Hallucination Resistance",       fig1_hallucination),
    ("Fig 2: Repetition Suppression",          fig2_repetition),
    ("Fig 3: Pre-Inference Speed",             fig3_speed),
    ("Fig 4: Search Space Scaling",            fig4_scaling),
    ("Fig 5: V3_U1 Valuation Trajectory",      fig5_v3u1_valuation),
    ("Fig 6: QCA Parallel Speedup",            fig6_qca_parallel_speedup),
    ("Fig 7: Architecture Comparison",         fig7_architecture_comparison),
    ("Fig 8: Latency Fix Analysis",            fig8_latency_fix_analysis),
]

ALL_PIPELINE_CHARTS = [
    ("Pipeline 1: Latency & Pruning",          pfig1_latency_and_pruning),
    ("Pipeline 2: FDSA Vocab Scaling",         pfig2_fdsa_vocab_scaling),
    ("Pipeline 3: Theorem 2.1 Ordering",       pfig3_theorem21_ordering),
    ("Pipeline 4: QCA Cluster Scaling",        pfig4_qca_cluster_scaling),
    ("Pipeline 5: Autoregressive Trajectory",  pfig5_autoregressive_trajectory),
    ("Pipeline 6: Backend Comparison",         pfig6_backend_comparison),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Master Visualization Generator — all 14 charts"
    )
    parser.add_argument("--core-only", action="store_true",
                        help="Generate only the 8 core engine charts (fig1-fig8)")
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Generate only the 6 pipeline charts (fig_pipeline_1-6)")
    parser.add_argument("--chart", type=int, nargs="+",
                        help="Generate specific chart(s) by number (1-14)")
    args = parser.parse_args()

    all_charts = ALL_CORE_CHARTS + ALL_PIPELINE_CHARTS

    print("=" * 70)
    print("  Actualizer_Engine_FDSA_QCA — Unified Master Chart Generator")
    print("  Framework: Computational Knowledge Theory (CKT V3_U1)")
    print(f"  Total charts available: {len(all_charts)}")
    print("=" * 70)

    # Determine which charts to run
    if args.core_only:
        charts_to_run = ALL_CORE_CHARTS
        print("  Mode: Core Engine charts only (8 charts)")
    elif args.pipeline_only:
        charts_to_run = ALL_PIPELINE_CHARTS
        print("  Mode: Pipeline charts only (6 charts)")
    elif args.chart:
        charts_to_run = []
        for idx in args.chart:
            if 1 <= idx <= len(all_charts):
                charts_to_run.append(all_charts[idx - 1])
            else:
                print(f"  [WARN] Chart #{idx} does not exist (valid: 1-{len(all_charts)})")
        print(f"  Mode: Selected charts ({len(charts_to_run)} charts)")
    else:
        charts_to_run = all_charts
        print(f"  Mode: ALL charts ({len(charts_to_run)} charts)")

    print()
    generated_paths = []
    failed = []

    for name, func in charts_to_run:
        try:
            path = func()
            generated_paths.append(path)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append((name, str(e)))

    print("\n" + "=" * 70)
    print(f"  [COMPLETE] Generated {len(generated_paths)}/{len(charts_to_run)} charts:")
    print("=" * 70)
    for p in generated_paths:
        print(f"  \u2713 {p}")
    if failed:
        print(f"\n  [FAILED] {len(failed)} chart(s) failed:")
        for name, err in failed:
            print(f"  \u2717 {name}: {err}")
