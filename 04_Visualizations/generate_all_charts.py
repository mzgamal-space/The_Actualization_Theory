"""
generate_all_charts.py — Master Visualization Generator
=========================================================
Runs all 4 test modules and produces 4 publication-quality PNG charts.
All charts use a dark, premium design consistent with modern ML papers.

Output files (saved to 04_Visualizations/):
  fig1_hallucination_comparison.png
  fig2_repetition_suppression.png
  fig3_speed_comparison.png
  fig4_search_space_scaling.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_Tests_and_Benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared style ───────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
GRID_C   = "#30363d"
C_BASE   = "#f85149"    # red  — baseline
C_FDSA   = "#3fb950"    # green — FDSA / Actualizer
C_ACCENT = "#58a6ff"    # blue  — accent
C_GOLD   = "#e3b341"    # gold  — diversity / highlight
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
    "axes.titlesize"    : 14,
    "axes.labelsize"    : 12,
    "legend.facecolor"  : PANEL,
    "legend.edgecolor"  : GRID_C,
    "legend.fontsize"   : 10,
})

def save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {fname}")
    return path

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Hallucination Resistance
# ═══════════════════════════════════════════════════════════════════════════
def fig1_hallucination():
    print("\n[Fig 1] Running hallucination test…")
    import test_01_hallucination as t1
    r = t1.run(vocab_size=500, steps=30)

    steps     = r["steps"]
    base_cum  = [sum(r["base_grounded"][:i+1]) / (i+1) * 100 for i in range(len(steps))]
    fdsa_cum  = [sum(r["fdsa_grounded"][:i+1]) / (i+1) * 100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Hallucination Resistance Under Distractor Bait",
                 fontsize=15, fontweight='bold', y=1.01)

    # Left — per-step groundedness
    ax = axes[0]
    ax.bar(steps, r["base_grounded"], color=C_BASE,  alpha=0.7, label="Baseline")
    ax.bar(steps, r["fdsa_grounded"], color=C_FDSA, alpha=0.85, label="FDSA + Actualizer", bottom=[0]*len(steps))
    ax.set_title("Per-Step Factual Groundedness")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Grounded (1) / Hallucinated (0)")
    ax.set_ylim(-0.1, 1.3)
    ax.legend()
    ax.grid(True, axis='y')

    # Right — cumulative groundedness rate
    ax = axes[1]
    ax.plot(steps, base_cum, color=C_BASE,  lw=2.5, label=f"Baseline  ({r['base_rate']}% final)")
    ax.plot(steps, fdsa_cum, color=C_FDSA, lw=2.5, label=f"FDSA+Act. ({r['fdsa_rate']}% final)")
    ax.fill_between(steps, base_cum, fdsa_cum, alpha=0.15, color=C_FDSA)
    ax.set_title("Cumulative Groundedness Rate (%)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Groundedness Rate (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True)
    ax.axhline(100, color=C_FDSA, lw=1, ls='--', alpha=0.4)

    fig.tight_layout()
    return save(fig, "fig1_hallucination_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Repetition Suppression
# ═══════════════════════════════════════════════════════════════════════════
def fig2_repetition():
    print("\n[Fig 2] Running repetition stress test…")
    import test_02_repetition_stress as t2
    r = t2.run(vocab_size=300, steps=40)

    steps = r["steps"]
    # Cumulative repeat rate
    base_cum_rep = [sum(r["base_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]
    fdsa_cum_rep = [sum(r["fdsa_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 — Repetition Loop Suppression (Order Prime)",
                 fontsize=15, fontweight='bold', y=1.01)

    # Left — cumulative repeat rate
    ax = axes[0]
    ax.plot(steps, base_cum_rep, color=C_BASE,  lw=2.5, label=f"Baseline ({r['base_repeat_rate']}%)")
    ax.plot(steps, fdsa_cum_rep, color=C_FDSA, lw=2.5, label=f"FDSA+Act. ({r['fdsa_repeat_rate']}%)")
    ax.fill_between(steps, base_cum_rep, fdsa_cum_rep, alpha=0.15, color=C_FDSA)
    ax.set_title("Cumulative Repeat Rate (%)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Repeat Rate (%)")
    ax.legend()
    ax.grid(True)

    # Right — token diversity (unique tokens in last 10 steps)
    ax = axes[1]
    ax.plot(steps, r["base_diversity"], color=C_BASE,  lw=2.5, label="Baseline Diversity")
    ax.plot(steps, r["fdsa_diversity"], color=C_GOLD,  lw=2.5, label="FDSA+Act. Diversity")
    ax.set_title("Token Diversity (Unique Tokens / 10-Step Window)")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Unique Tokens")
    avg_b = sum(r["base_diversity"])/len(r["base_diversity"])
    avg_f = sum(r["fdsa_diversity"])/len(r["fdsa_diversity"])
    ax.axhline(avg_b, color=C_BASE, lw=1, ls='--', alpha=0.5)
    ax.axhline(avg_f, color=C_GOLD, lw=1, ls='--', alpha=0.5)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return save(fig, "fig2_repetition_suppression.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Pre-Inference Speed Sweep
# ═══════════════════════════════════════════════════════════════════════════
def fig3_speed():
    print("\n[Fig 3] Running pre-inference speed sweep (V = 1k → 100k)…")
    import test_03_pre_inference_speed as t3
    r = t3.run(trials=30)

    V      = r["vocab_sizes"]
    V_lbls = [f"{v//1000}k" for v in V]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 3 — Pre-Inference Speed Comparison: Standard Softmax vs FDSA Pruning",
                 fontsize=15, fontweight='bold', y=1.01)

    x = np.arange(len(V))
    w = 0.38

    # Left — latency bar chart
    ax = axes[0]
    ax.bar(x - w/2, r["baseline_ms"], w, color=C_BASE,  label="Baseline Softmax")
    ax.bar(x + w/2, r["fdsa_ms"],     w, color=C_FDSA, label="FDSA Pruned Softmax")
    ax.set_xticks(x); ax.set_xticklabels(V_lbls)
    ax.set_title("Sampling Latency per Token (ms)")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(True, axis='y')

    # Middle — speedup factor
    ax = axes[1]
    bars = ax.bar(x, r["speedup"], color=C_ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(V_lbls)
    ax.set_title("Throughput Speedup Factor (×)")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Speedup (×)")
    ax.axhline(1.0, color=C_MUTED, lw=1, ls='--')
    for bar, sv in zip(bars, r["speedup"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{sv:.1f}×", ha='center', va='bottom', fontsize=10, color=C_TEXT)
    ax.grid(True, axis='y')

    # Right — pruning rate
    ax = axes[2]
    ax.plot(V_lbls, r["pruning_rate_pct"], color=C_GOLD, lw=2.5, marker='o', ms=7)
    ax.fill_between(range(len(V)), r["pruning_rate_pct"], alpha=0.2, color=C_GOLD)
    ax.set_title("Vocabulary Pruning Rate (%)")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Pruning Rate (%)")
    ax.set_ylim(95, 100.5)
    ax.grid(True)
    for i, pct in enumerate(r["pruning_rate_pct"]):
        ax.annotate(f"{pct:.2f}%", (i, pct), textcoords="offset points",
                    xytext=(0, 6), ha='center', fontsize=9)

    fig.tight_layout()
    return save(fig, "fig3_speed_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Search Space Scaling
# ═══════════════════════════════════════════════════════════════════════════
def fig4_scaling():
    print("\n[Fig 4] Running search space scaling analysis (N = 4 → 18)…")
    import test_04_search_space_scaling as t4
    r = t4.run()

    N     = r["n_values"]
    c_log = np.log10([max(v, 1) for v in r["combinatorial_nodes"]])
    f_log = np.log10([max(v, 1) for v in r["fdsa_nodes"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4 — Search Space Scaling: Combinatorial O(M^N) vs FDSA O(N^D)",
                 fontsize=15, fontweight='bold', y=1.01)

    # Left — log-scale node counts
    ax = axes[0]
    ax.plot(N, c_log, color=C_BASE,  lw=2.5, marker='o', ms=6, label="Combinatorial O(3^N)")
    ax.plot(N, f_log, color=C_FDSA, lw=2.5, marker='s', ms=6, label=f"FDSA O(N^D), k=0.35")
    ax.fill_between(N, f_log, c_log, alpha=0.12, color=C_FDSA, label="Pruned Region")
    ax.set_title("Search Space Size (log₁₀ scale)")
    ax.set_xlabel("Problem Size N (tasks)")
    ax.set_ylabel("log₁₀(Search Nodes)")
    ax.legend()
    ax.grid(True)

    # Right — reduction %
    ax = axes[1]
    bars = ax.bar(N, r["reduction_pct"], color=C_ACCENT, alpha=0.85)
    ax.set_title("Search Space Reduction (%)")
    ax.set_xlabel("Problem Size N")
    ax.set_ylabel("Reduction (%)")
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, r["reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.0f}%", ha='center', va='bottom', fontsize=8.5, color=C_TEXT)
    ax.grid(True, axis='y')

    fig.tight_layout()
    return save(fig, "fig4_search_space_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Actualizer Engine — Master Chart Generator")
    print("=" * 60)
    paths = []
    paths.append(fig1_hallucination())
    paths.append(fig2_repetition())
    paths.append(fig3_speed())
    paths.append(fig4_scaling())
    print("\n✓ All charts generated:")
    for p in paths:
        print(f"    {p}")
