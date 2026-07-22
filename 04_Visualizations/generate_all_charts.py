"""
generate_all_charts.py — Master Visualization Generator  [V3_U1 updated]
=========================================================
Runs all test modules and produces 5 publication-quality PNG charts.
All charts use a dark, premium design consistent with modern ML papers.

Output files (saved to 04_Visualizations/):
  fig1_hallucination_comparison.png
  fig2_repetition_suppression.png
  fig3_speed_comparison.png
  fig4_search_space_scaling.png
  fig5_v3u1_valuation_trajectory.png  [NEW in V3_U1]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_Tests_and_Benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared style ───────────────────────────────────────────────────────────
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

# ===========================================================================
# Figure 1 - Hallucination Resistance
# ===========================================================================
def fig1_hallucination():
    print("\n[Fig 1] Running hallucination test...")
    import test_01_hallucination as t1
    r = t1.run(vocab_size=1000, steps=30)
    steps    = r["steps"]
    base_cum = [sum(r["base_grounded"][:i+1])/(i+1)*100 for i in range(len(steps))]
    fdsa_cum = [sum(r["fdsa_grounded"][:i+1])/(i+1)*100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 - Hallucination Resistance Under Distractor Bait",
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
    return save(fig, "fig1_hallucination_comparison.png")


# ===========================================================================
# Figure 2 - Repetition Suppression
# ===========================================================================
def fig2_repetition():
    print("\n[Fig 2] Running repetition stress test...")
    import test_02_repetition_stress as t2
    r = t2.run(vocab_size=300, steps=40)
    steps = r["steps"]
    base_cum_rep = [sum(r["base_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]
    fdsa_cum_rep = [sum(r["fdsa_repeat_counts"][:i+1])/(i+1)*100 for i in range(len(steps))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 - Repetition Loop Suppression (Order Prime)",
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
    return save(fig, "fig2_repetition_suppression.png")


# ===========================================================================
# Figure 3 - Pre-Inference Speed Sweep
# ===========================================================================
def fig3_speed():
    print("\n[Fig 3] Running pre-inference speed sweep (V = 1k -> 100k)...")
    import test_03_pre_inference_speed as t3
    r = t3.run(trials=30)
    V      = r["vocab_sizes"]
    V_lbls = [f"{v//1000}k" for v in V]
    x = np.arange(len(V)); w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 3 - Pre-Inference Speed: Standard Softmax vs FDSA Pruning",
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
    return save(fig, "fig3_speed_comparison.png")


# ===========================================================================
# Figure 4 - Search Space Scaling
# ===========================================================================
def fig4_scaling():
    print("\n[Fig 4] Running search space scaling analysis (N = 4 -> 18)...")
    import test_04_search_space_scaling as t4
    r = t4.run()
    N     = r["n_values"]
    c_log = np.log10([max(v, 1) for v in r["combinatorial_nodes"]])
    f_log = np.log10([max(v, 1) for v in r["fdsa_nodes"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4 - Search Space Scaling: Combinatorial O(M^N) vs FDSA O(N^D)",
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
    return save(fig, "fig4_search_space_scaling.png")


# ===========================================================================
# Figure 5 - V3_U1: nu_t Valuation Trajectory + Bifurcation Compliance
# NEW in V3_U1 - visualizes Theorem 3.3 and Section 3.3.1-A
# ===========================================================================
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
    return save(fig, "fig5_v3u1_valuation_trajectory.png")


# ===========================================================================
# Figure 6 - QCA Parallel Engine Speedup
# ===========================================================================
def fig6_qca_parallel_speedup():
    print("\n[Fig 6] Running QCA Parallel Engine benchmark...")
    import test_06_qca_parallel_engine as t6
    r = t6.run(n_sizes=(20, 40, 80, 120, 200), K=5, vocab_size=1000)
    n_sizes = r["n_sizes"]
    t_seq   = r["sequential_ms"]
    t_par   = r["parallel_ms"]
    speedup = r["speedup"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 6 - QCA Parallel Engine Acceleration (K={r['K']} Clusters, V={r['vocab_size']})",
                 fontsize=15, fontweight='bold', y=1.01)

    # ── Panel 1: Execution Time Comparison ──
    ax = axes[0]
    ax.plot(n_sizes, t_seq, marker='o', color=C_BASE,   lw=2.5, label="Sequential Baseline O(N²)")
    ax.plot(n_sizes, t_par, marker='s', color=C_FDSA,   lw=2.5, label=f"QCA Parallel Engine O(N²/{r['K']})")
    ax.fill_between(n_sizes, t_par, t_seq, color=C_FDSA, alpha=0.15, label="Time Saved by QCA Partitioning")

    # Stacked breakdown bars for parallel execution
    qca_t   = r["qca_ms"]
    worker_t = r["parallel_worker_ms"]
    synth_t = r["synthesis_ms"]
    
    ax.set_title("Execution Latency vs Problem Size (N)")
    ax.set_xlabel("Dataset Size N (Number of Nodes)")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend(fontsize=10); ax.grid(True)

    # ── Panel 2: Speedup Factor ──
    ax = axes[1]
    bars = ax.bar([str(n) for n in n_sizes], speedup, color=C_ACCENT, alpha=0.85, width=0.5)
    ax.axhline(1.0, color=C_MUTED, lw=1.5, ls='--', label="Baseline 1.0×")
    ax.axhline(float(r['K']), color=C_GOLD, lw=2.0, ls=':', label=f"Theoretical Upper Bound (K={r['K']}×)")

    for bar, s in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.05,
                f"{s:.2f}×", ha='center', va='bottom', fontsize=10, color=C_TEXT, fontweight='bold')

    ax.set_title(f"Empirical Parallel Speedup (Factor-K={r['K']} Baseline)")
    ax.set_xlabel("Dataset Size N (Number of Nodes)")
    ax.set_ylabel("Speedup Factor (Sequential / Parallel)")
    ax.set_ylim(0, max(speedup) * 1.35 if speedup else 5.0)
    ax.legend(fontsize=10); ax.grid(True, axis='y')

    fig.tight_layout()
    return save(fig, "fig6_qca_parallel_speedup.png")


# ===========================================================================
# Figure 7 — Three-Way Architecture Comparison
# ===========================================================================
def fig7_architecture_comparison():
    """
    Three-panel comparison chart from test_07_architecture_comparison empirical results.

    Panel 1: Grounding / Hallucination / Repetition rates (quality metrics bar chart)
    Panel 2: Latency vs vocabulary size V — Baseline vs Actualizer
    Panel 3: QCA Parallel speedup vs dataset size N (K=5, V=2000)
    """
    # ── Empirical results (from test_07 run: V=500, n_steps=30, distractor=+8.0) ──
    models       = ["Attention\nBaseline", "Actualizer\nEngine", "QCA Parallel\nEngine"]
    grounded     = [0.0000, 1.0000, 1.0000]
    hallucinat   = [1.0000, 0.0000, 0.0000]
    repetition   = [0.0345, 0.0000, 0.0000]
    valuation    = [0.0000, 0.4027, 0.5094]   # 0 for baseline = no mechanism
    actualized   = [0.0000, 1.0000, 1.0000]   # 0 for baseline = no mechanism

    vocab_sizes  = [500, 1000, 2000]
    baseline_ms  = [52.07, 77.54, 114.52]
    actualiz_ms  = [8165.87, 14190.98, 21353.16]

    n_sizes      = [20, 40, 80, 120, 200]
    seq_ms       = [4736.99, 10770.26, 20315.86, 31948.36, 41021.67]
    par_ms       = [5797.64,  8423.78, 12940.84, 14878.49, 18298.32]
    speedup      = [0.82, 1.28, 1.57, 2.15, 2.24]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Three-Way Architecture Comparison: Attention Baseline vs Actualizer Engine vs QCA Parallel Engine",
        fontsize=13, fontweight='bold', color=C_TEXT, y=1.01
    )

    # ── Panel 1: Quality metrics grouped bar chart ────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    x      = np.arange(len(models))
    width  = 0.18
    colors_bars = [C_FDSA, C_BASE, C_GOLD, C_ACCENT, C_PURPLE]
    metric_data = [grounded, hallucinat, repetition, valuation, actualized]
    metric_labels = ["Grounded", "Hallucination", "Repetition", "Valuation ν_t", "Actualized"]

    for i, (data, label, col) in enumerate(zip(metric_data, metric_labels, colors_bars)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, data, width, label=label, color=col, alpha=0.85, edgecolor='none')
        # Annotate non-zero bars
        for bar, val in zip(bars, data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=7.5,
                        color=C_TEXT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Rate / Score  [0–1]", color=C_TEXT)
    ax.set_title("Quality Metrics (V=500, 30 steps, distractor +8.0)", fontsize=11, color=C_TEXT)
    ax.legend(fontsize=8, loc='upper right', fancybox=True, framealpha=0.3)
    ax.grid(True, axis='y', alpha=0.5)

    # Baseline annotation
    ax.annotate("100% hallucination\n(distractor wins)",
                xy=(0, 1.0), xytext=(0.32, 1.12),
                fontsize=8, color=C_BASE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_BASE, lw=1.2))

    # ── Panel 2: Latency scaling vs vocabulary size ───────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.plot(vocab_sizes, baseline_ms,  'o-', color=C_BASE,   lw=2, ms=7, label="Attention Baseline  O(V)")
    ax2.plot(vocab_sizes, actualiz_ms,  's-', color=C_FDSA,   lw=2, ms=7, label="Actualizer Engine  O(V×iters)")
    ax2.fill_between(vocab_sizes, baseline_ms, actualiz_ms, alpha=0.08, color=C_FDSA)

    for v, bms, ams in zip(vocab_sizes, baseline_ms, actualiz_ms):
        ax2.annotate(f"{bms:.0f}ms", xy=(v, bms), xytext=(v - 80, bms + 400),
                     fontsize=8, color=C_BASE)
        ax2.annotate(f"{ams:.0f}ms", xy=(v, ams), xytext=(v - 80, ams - 1200),
                     fontsize=8, color=C_FDSA)

    ax2.set_xlabel("Vocabulary Size V", color=C_TEXT)
    ax2.set_ylabel("Latency (ms, 30 steps)", color=C_TEXT)
    ax2.set_title("Latency vs Vocabulary Size\n(Baseline vs Actualizer Engine)", fontsize=11, color=C_TEXT)
    ax2.legend(fontsize=9, fancybox=True, framealpha=0.3)
    ax2.grid(True, alpha=0.5)

    note = ("Actualizer overhead is\nthe price of 100% grounding\n& zero hallucination")
    ax2.text(0.98, 0.12, note, transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=8.5,
             color=C_MUTED, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=GRID_C, alpha=0.8))

    # ── Panel 3: QCA Parallel speedup ─────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(PANEL)
    ax3_r = ax3.twinx()
    ax3_r.set_facecolor(PANEL)

    ax3.plot(n_sizes, seq_ms, 'o--', color=C_BASE,   lw=2, ms=7, label="Sequential  O(N²)")
    ax3.plot(n_sizes, par_ms, 's-',  color=C_FDSA,   lw=2.5, ms=7, label="QCA Parallel  O(N²/K)")
    ax3.fill_between(n_sizes, par_ms, seq_ms, alpha=0.12, color=C_FDSA, label="Parallel savings")

    ax3_r.plot(n_sizes, speedup, '^-', color=C_GOLD, lw=2, ms=8, label="Speedup factor")
    ax3_r.axhline(y=5.0, color=C_PURPLE, lw=1, ls=':', alpha=0.7, label="K=5× theoretical max")

    for n, sp in zip(n_sizes, speedup):
        ax3_r.annotate(f"{sp:.2f}×", xy=(n, sp), xytext=(n + 3, sp + 0.06),
                       fontsize=8.5, color=C_GOLD, fontweight='bold')

    ax3.set_xlabel("Dataset Size N", color=C_TEXT)
    ax3.set_ylabel("Execution Time (ms)", color=C_TEXT)
    ax3_r.set_ylabel("Speedup Factor", color=C_GOLD)
    ax3_r.tick_params(axis='y', colors=C_GOLD)
    ax3_r.set_ylim(0, 6.5)
    ax3.set_title(f"QCA Parallel Engine Speedup\n(K=5, V=2000)", fontsize=11, color=C_TEXT)

    lines_a, labels_a = ax3.get_legend_handles_labels()
    lines_b, labels_b = ax3_r.get_legend_handles_labels()
    ax3.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8.5,
               loc='upper left', fancybox=True, framealpha=0.3)
    ax3.grid(True, alpha=0.5)

    fig.tight_layout()
    return save(fig, "fig7_architecture_comparison.png")


# ===========================================================================
# Main
# ===========================================================================
def fig8_latency_fix_analysis():
    """
    Four-panel latency root cause + fix analysis chart.
    Empirical data from test_08_latency_jax_comparison (V=500, 30 steps, distractor=+8.0).

    Panel 1: Latency comparison of all 5 approaches (log scale bar chart)
    Panel 2: Speedup factor vs vocabulary size V (NumPy and FDSA+NumPy)
    Panel 3: Grounding quality maintained across all fixes (quality is preserved)
    Panel 4: JAX backend QCA vs processes comparison
    """
    # ── Phase 1 data (V=500, 30 steps) ──────────────────────────────────────
    labels   = ["Baseline\n(Softmax+Argmax)", "Naive\nActualizer\n(Python loops)",
                "FDSA Active\nVocab", "NumPy\nVectorized", "FDSA +\nNumPy Combined"]
    latency  = [12.8, 6562.4, 6029.1, 1172.2, 1177.5]
    grounded = [0.00,    1.00,    1.00,   1.00,     1.00]
    colors_b = [C_BASE, "#4a4aff", C_MUTED, C_FDSA, C_GOLD]

    # ── Phase 2 data ─────────────────────────────────────────────────────────
    V_vals      = [100, 500, 1000]
    naive_ms    = [1842.3, 6160.6, 13124.8]
    numpy_ms    = [1166.4,  722.6,  1274.6]
    combined_ms = [1059.0,  687.9,  1514.2]
    speedup_np  = [round(n / max(m, 0.01), 2) for n, m in zip(naive_ms, numpy_ms)]
    speedup_cb  = [round(n / max(m, 0.01), 2) for n, m in zip(naive_ms, combined_ms)]

    # ── Phase 3 data ─────────────────────────────────────────────────────────
    backends   = ["JAX\n(vectorized)", "Processes\n(spawn)"]
    qca_ms_bck = [6335.1, 12940.84]   # JAX measured, Processes from test_06 N=80
    qca_spd    = [round(12940.84 / 6335.1, 2), 1.0]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Latency Root Cause Analysis & Fix Benchmark  |  V3_U1 ActualizerEngine",
        fontsize=13, fontweight='bold', color=C_TEXT, y=1.02
    )

    # ── Panel 1: Latency bar chart (log scale) ───────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    bars = ax.bar(range(len(labels)), latency, color=colors_b, alpha=0.85, edgecolor='none', width=0.6)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{val:.0f}ms", ha='center', va='bottom', fontsize=8, color=C_TEXT, fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (ms, log scale)", color=C_TEXT)
    ax.set_title("Approach Latency — V=500, 30 Steps\n(log scale; lower = better)", fontsize=10, color=C_TEXT)
    ax.grid(True, axis='y', alpha=0.4)

    # Annotate speedups
    ax.annotate("5.6x faster →", xy=(3, 1172), xytext=(1.8, 3000),
                fontsize=8.5, color=C_FDSA, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_FDSA, lw=1.2))
    ax.annotate("ROOT CAUSE:\nPython for-loops\nO(V×iters)",
                xy=(1, 6562), xytext=(0.0, 200),
                fontsize=7.5, color="#4a4aff",
                arrowprops=dict(arrowstyle='->', color="#4a4aff", lw=1.0))

    # ── Panel 2: Speedup vs V ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.plot(V_vals, speedup_np, 'o-', color=C_FDSA,   lw=2.5, ms=8, label="NumPy Vectorized")
    ax2.plot(V_vals, speedup_cb, 's--', color=C_GOLD,  lw=2,   ms=8, label="FDSA + NumPy")
    ax2.fill_between(V_vals, speedup_np, alpha=0.1, color=C_FDSA)

    for v, sp_n, sp_c in zip(V_vals, speedup_np, speedup_cb):
        ax2.annotate(f"{sp_n:.1f}×", xy=(v, sp_n), xytext=(v - 50, sp_n + 0.5),
                     fontsize=8.5, color=C_FDSA, fontweight='bold')

    ax2.set_xlabel("Vocabulary Size V", color=C_TEXT)
    ax2.set_ylabel("Speedup vs Naive Python Loops", color=C_TEXT)
    ax2.set_title("Speedup Factor vs V\n(vs naive Python-loop engine)", fontsize=10, color=C_TEXT)
    ax2.legend(fontsize=9, fancybox=True, framealpha=0.3)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(0, max(max(speedup_np), max(speedup_cb)) * 1.4)

    ax2.text(0.05, 0.95, "Speedup grows with V\nO(V) numpy vs O(V) Python",
             transform=ax2.transAxes, ha='left', va='top', fontsize=8,
             color=C_MUTED, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=GRID_C, alpha=0.7))

    # ── Panel 3: Grounding preserved ─────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(PANEL)
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
    ax3.set_ylabel("Grounded Rate  [0–1]", color=C_TEXT)
    ax3.set_title("Grounding Quality Preserved\n(100% across all fixes)", fontsize=10, color=C_TEXT)
    ax3.grid(True, axis='y', alpha=0.4)
    ax3.text(0.5, 0.92, "Quality = 100% in ALL fixed approaches\nLatency cut 5.6x with zero quality loss",
             transform=ax3.transAxes, ha='center', va='top', fontsize=8.5, color=C_FDSA,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=C_FDSA, alpha=0.3))

    # ── Panel 4: JAX vs Processes for QCA ────────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor(PANEL)
    bk_colors = [C_GOLD, C_ACCENT]
    bars4 = ax4.bar(backends, qca_ms_bck, color=bk_colors, alpha=0.85, edgecolor='none', width=0.45)
    for bar, val, spd in zip(bars4, qca_ms_bck, qca_spd):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:.0f}ms", ha='center', va='bottom', fontsize=9, color=C_TEXT, fontweight='bold')
        if spd != 1.0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                     f"{spd:.2f}×\nfaster", ha='center', va='center', fontsize=9,
                     color='black', fontweight='bold')

    ax4.set_ylabel("Latency (ms, N=80, K=5, V=1000)", color=C_TEXT)
    ax4.set_title("QCA Parallel: JAX vs Processes\n(N=80, K=5, V=1000)", fontsize=10, color=C_TEXT)
    ax4.grid(True, axis='y', alpha=0.4)
    ax4.text(0.5, 0.95, f"JAX available: YES\nJAX 2.04x faster than\nProcessPoolExecutor",
             transform=ax4.transAxes, ha='center', va='top', fontsize=8.5, color=C_GOLD,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL, edgecolor=C_GOLD, alpha=0.3))

    fig.tight_layout()
    return save(fig, "fig8_latency_fix_analysis.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  Actualizer Engine - Master Chart Generator  [V3_U1]")
    print("=" * 60)
    paths = []
    paths.append(fig1_hallucination())
    paths.append(fig2_repetition())
    paths.append(fig3_speed())
    paths.append(fig4_scaling())
    paths.append(fig5_v3u1_valuation())
    paths.append(fig6_qca_parallel_speedup())
    paths.append(fig7_architecture_comparison())
    paths.append(fig8_latency_fix_analysis())
    print(f"\nAll {len(paths)} charts generated:")
    for p in paths:
        print(f"    {p}")


