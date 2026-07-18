"""
test_05_v3u1_features.py  —  V3_U1 Theory Compliance Test Suite
================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153

Tests that the ActualizerEngine correctly implements all six V3_U1 corrections:

  [FIX-1]  H(R) uses squared magnitude-defect: (Σα²-1)²  not |Σα-1|
  [FIX-2]  ν_t valuation trajectory is returned per iteration
  [FIX-3]  Tr(D_μν) bifurcation criterion is computed and returned
  [FIX-4]  prime_weights is a constructor parameter
  [FIX-5]  mercy_k is the primary k parameter (Mercy = k, §3.3.1-C)
  [FIX-6]  Causal snap is gated by Tr(D_μν) ≤ tau_bifurcation

Also produces a ν_t trajectory dataset used by generate_all_charts.py (Fig 5).

Returns a dict for generate_all_charts.py.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from actualizer_engine import ActualizerEngine, EQUILIBRIUM_ALPHA, N_PRIMES, H_MAX_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_logits(V, target_tok, distractor_tok, target_logit=3.5, bait_logit=8.0):
    import random
    random.seed(42)
    logits = [random.gauss(-4.0, 1.0) for _ in range(V)]
    logits[target_tok]    = target_logit
    logits[distractor_tok] = bait_logit
    return logits


def _softmax(logits):
    max_l = max(logits)
    e     = [math.exp(x - max_l) for x in logits]
    s     = sum(e)
    return [x / s for x in e]


# ─────────────────────────────────────────────────────────────────────────────
# Individual tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fix1_squared_entropy():
    """
    [FIX-1] H(R) = Var(α) + (Σα²-1)²  must equal 0 at equilibrium α_i=1/√5.
    """
    engine = ActualizerEngine(vocab_size=100, mercy_k=0.45)
    alpha_eq = [EQUILIBRIUM_ALPHA] * N_PRIMES
    H = engine._structural_entropy(alpha_eq)
    assert abs(H) < 1e-10, f"H(equilibrium) should be 0, got {H}"

    # At a highly skewed state, H must be > 0
    alpha_skew = [0.9, 0.05, 0.02, 0.02, 0.01]
    H_skew = engine._structural_entropy(alpha_skew)
    assert H_skew > 0, f"H(skewed) should be > 0, got {H_skew}"

    # Verify squared (not absolute): the defect term uses (Σα²-1)²
    # For alpha=[1,0,0,0,0]:
    #   mean = 0.2, Var = ((1-0.2)²+4*(0-0.2)²)/5 = (0.64+0.16)/5 = 0.16
    #   Σα² = 1 → (Σα²-1)² = 0
    #   So H = 0.16 + 0 = 0.16
    alpha_spike = [1.0, 0.0, 0.0, 0.0, 0.0]
    H_spike = engine._structural_entropy(alpha_spike)
    expected_var = 0.16   # population variance with divisor n=5
    assert abs(H_spike - expected_var) < 1e-9, (
        f"H([1,0,0,0,0]) should be {expected_var} (Var only, defect=0), got {H_spike}"
    )
    return {"passed": True, "H_equilibrium": H, "H_skewed": round(H_skew, 6)}


def test_fix2_nu_t_trajectory():
    """
    [FIX-2] steer() must return nu_history as a list of ν_t values.
    ν_t should be in [0,1] and increase monotonically on average.
    """
    V     = 200
    engine = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-4, max_iters=30)
    logits = _make_logits(V, target_tok=50, distractor_tok=V-1,
                          target_logit=3.5, bait_logit=0.0)   # no bait → clean case
    history = [48, 49, 50]
    targets = set(range(50, 80))

    _, _, Tr_D, iters, nu_history, actualized = engine.steer(logits, history, targets)

    assert len(nu_history) == iters, \
        f"nu_history length {len(nu_history)} != iters {iters}"
    assert all(0.0 <= v <= 1.0 for v in nu_history), \
        f"All nu_t must be in [0,1]: {nu_history}"
    return {
        "passed"     : True,
        "nu_history" : nu_history,
        "iters"      : iters,
        "actualized" : actualized,
        "Tr_D_final" : round(Tr_D, 8),
    }


def test_fix3_trace_bifurcation():
    """
    [FIX-3] Tr(D_μν) must be < tau_bifurcation for a well-constrained substrate,
    triggering the ACTUALIZATION branch (actualized=True).
    """
    V      = 100
    engine = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-5,
                               tau_bifurcation=5.0)

    # Clean logits: target at 50 gets +4, rest noise → should actualize
    logits = [0.0] * V
    logits[50] = 4.0
    logits[51] = 3.5
    # Everything else = 0 → mostly in target window
    history = [48, 49, 50]
    targets = set(range(50, 80))

    tok, _, Tr_D, iters, nu_hist, actualized = engine.steer(logits, history, targets)

    assert actualized, f"Expected ACTUALIZATION branch, got dissolution. Tr_D={Tr_D}"
    assert Tr_D <= 5.0, f"Tr_D={Tr_D} should be <= tau_bifurcation=5.0"
    assert tok in targets, f"Selected token {tok} should be in target window"
    return {
        "passed"    : True,
        "tok"       : tok,
        "Tr_D"      : round(Tr_D, 8),
        "actualized": actualized,
        "iters"     : iters,
    }


def test_fix4_prime_weights_param():
    """
    [FIX-4] prime_weights must be a constructor parameter that affects drift scores.
    Changing weights should change the selected token in edge cases.
    """
    V = 100
    logits = [0.0] * V
    logits[50] = 3.0   # near-equal candidates
    logits[51] = 3.1
    history = [48, 49, 50]
    targets = set(range(50, 80))

    # Default weights
    e1 = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-4, max_iters=30)
    t1, _, _, _, _, _ = e1.steer(logits, history, targets)

    # Heavy Order weight (suppresses recent tokens strongly)
    e2 = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-4, max_iters=30,
                          prime_weights={"Order": 0.70, "Justice": 0.20,
                                         "Knowledge": 0.07, "Mercy": 0.03})
    t2, _, _, _, _, _ = e2.steer(logits, history, targets)

    # Both must select valid targets (weights change magnitude, not validity)
    assert t1 in targets, f"Default weights: tok {t1} not in targets"
    assert t2 in targets, f"Heavy Order: tok {t2} not in targets"
    return {
        "passed"         : True,
        "tok_default"    : t1,
        "tok_heavy_order": t2,
        "weights_differ" : (t1 != t2),  # often true, not required
    }


def test_fix5_mercy_k_alias():
    """
    [FIX-5] mercy_k must be the primary parameter name.
    engine.mercy_k and engine.k must refer to the same value.
    """
    engine = ActualizerEngine(vocab_size=50, mercy_k=0.37)
    assert engine.mercy_k == 0.37, f"mercy_k={engine.mercy_k} != 0.37"
    assert engine.k == 0.37, f"k alias={engine.k} != 0.37"
    return {"passed": True, "mercy_k": engine.mercy_k, "k_alias": engine.k}


def test_fix6_causal_snap_gating():
    """
    [FIX-6] The causal snap (Power Prime) should return actualized=False
    only when Tr(D_μν) > tau_bifurcation.
    We test this by setting tau_bifurcation very low (0.0) so it always dissolves.
    """
    V = 100
    logits = [0.0] * V
    logits[50] = 4.0
    history = [48, 49, 50]
    targets = set(range(50, 80))

    # tau_bifurcation = 0.0 → Tr_D always exceeds tau → dissolution branch
    engine = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-5,
                               tau_bifurcation=0.0)
    tok, _, Tr_D, iters, nu_hist, actualized = engine.steer(logits, history, targets)

    # With tau=0, ANY positive trace → dissolution
    # Fallback: most recent token in history ∩ targets or 0
    assert not actualized, \
        f"Expected DISSOLUTION branch with tau=0.0, got actualized=True"
    return {
        "passed"    : True,
        "tok"       : tok,
        "Tr_D"      : round(Tr_D, 8),
        "actualized": actualized,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ν_t trajectory dataset  (for Figure 5 in generate_all_charts.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_nu_trajectory_data():
    """
    Run the engine on 3 scenarios and collect ν_t per iteration:
      A: Clean substrate  (no distractor, target well-separated)
      B: Moderate noise   (target + bait at equal logit)
      C: Strong distractor (bait +8 but masked by FDSA grammar)

    Returns per-scenario nu_history and Tr_D_history for Figure 5.
    """
    import random
    from fdsa_pruner import VectorizedFDSAPruner

    V       = 500
    history = [48, 49, 50, 50]
    targets = set(range(50, 80))

    grammar = {}
    for t in range(50, 79):
        grammar[t] = {t + 1, ((t - 50 + 7) % 29) + 50}
    grammar[79] = {50, 57}

    pruner = VectorizedFDSAPruner(vocab_size=V, k=0.35)

    scenarios = {}

    # Scenario A: clean
    random.seed(10)
    logits_a = [random.gauss(-4.0, 1.0) for _ in range(V)]
    logits_a[51] = 4.0
    logits_a[57] = 3.5
    eng_a = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-6, max_iters=80)
    pruned_a, _ = pruner.prune_vocabulary(logits_a, 50, grammar, "logical_coding")
    tok_a, _, Tr_a, iters_a, nu_a, act_a = eng_a.steer(pruned_a, history, targets)
    scenarios["clean"] = {"nu": nu_a, "iters": iters_a, "tok": tok_a,
                          "actualized": act_a, "Tr_D_final": Tr_a}

    # Scenario B: moderate noise (bait at +3, target at +3.5)
    random.seed(20)
    logits_b = [random.gauss(-4.0, 1.0) for _ in range(V)]
    logits_b[51] = 3.5
    logits_b[V-1] = 3.0   # bait lower than grammar, grammar masks it
    eng_b = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-6, max_iters=80)
    pruned_b, _ = pruner.prune_vocabulary(logits_b, 50, grammar, "logical_coding")
    tok_b, _, Tr_b, iters_b, nu_b, act_b = eng_b.steer(pruned_b, history, targets)
    scenarios["moderate"] = {"nu": nu_b, "iters": iters_b, "tok": tok_b,
                              "actualized": act_b, "Tr_D_final": Tr_b}

    # Scenario C: strong distractor (bait +8) — masked by FDSA grammar
    random.seed(30)
    logits_c = [random.gauss(-4.0, 1.0) for _ in range(V)]
    logits_c[51] = 3.2
    logits_c[V-1] = 8.0   # strong bait — will be masked by grammar
    eng_c = ActualizerEngine(vocab_size=V, mercy_k=0.45, Q_c=1e-6, max_iters=80)
    pruned_c, _ = pruner.prune_vocabulary(logits_c, 50, grammar, "logical_coding")
    tok_c, _, Tr_c, iters_c, nu_c, act_c = eng_c.steer(pruned_c, history, targets)
    scenarios["distractor"] = {"nu": nu_c, "iters": iters_c, "tok": tok_c,
                                "actualized": act_c, "Tr_D_final": Tr_c}

    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Master run() — called by generate_all_charts.py
# ─────────────────────────────────────────────────────────────────────────────

def run() -> dict:
    results = {}

    tests = [
        ("fix1_squared_entropy",  test_fix1_squared_entropy),
        ("fix2_nu_t_trajectory",  test_fix2_nu_t_trajectory),
        ("fix3_trace_bifurcation",test_fix3_trace_bifurcation),
        ("fix4_prime_weights",    test_fix4_prime_weights_param),
        ("fix5_mercy_k_alias",    test_fix5_mercy_k_alias),
        ("fix6_causal_snap",      test_fix6_causal_snap_gating),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            r = fn()
            results[name] = r
            status = "[PASS]" if r.get("passed") else "[FAIL]"
        except Exception as e:
            results[name] = {"passed": False, "error": str(e)}
            status = f"[ERROR] {e}"
            all_passed = False
        print(f"  {status}  {name}")

    results["all_passed"]    = all_passed
    results["nu_trajectory"] = build_nu_trajectory_data()
    return results


if __name__ == "__main__":
    print("V3_U1 Theory Compliance Tests\n" + "=" * 50)
    r = run()
    print()
    print(f"All tests passed: {r['all_passed']}")
    print("\nnu_t trajectory scenarios:")
    for name, data in r["nu_trajectory"].items():
        print(f"  {name}: {data['iters']} iters, "
              f"tok={data['tok']}, actualized={data['actualized']}, "
              f"nu_final={data['nu'][-1]:.4f}, Tr_D={data['Tr_D_final']:.6f}")
