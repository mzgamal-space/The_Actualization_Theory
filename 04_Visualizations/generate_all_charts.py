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
    r = t1.run(vocab_size=500, steps=30)
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
# Main
# ===========================================================================
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
    print(f"\nAll {len(paths)} charts generated:")
    for p in paths:
        print(f"    {p}")
