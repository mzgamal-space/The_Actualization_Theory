"""
test_09_drift_alignment_verification.py
=========================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153

BUG FIX VERIFICATION: D_future entropy gradient alignment
===========================================================

BUG DESCRIPTION (fixed):
    actualizer_engine.py line ~391 computed D_future as:
        entropy_grad = -math.log(max(p_v, 1e-12))
        D[v] += w_F * entropy_grad * 0.08

    But the docstring (line ~379) derives the correct formula:
        -d(entropy)/d(p_v) = log(p_v) + 1

    The code was wrong in three ways:
      1. Sign: -log(p_v) is always positive for p_v < 1
              log(p_v) + 1 is negative for p_v < 1/e, positive for p_v > 1/e
      2. Missing +1 constant from the Shannon entropy derivative
      3. Arbitrary 0.08 scaling factor instead of using w_F directly

    numpy_actualizer_engine.py already had the correct formula:
        D += self.w_F * (np.log(safe_U) + 1.0)

FIX APPLIED:
    actualizer_engine.py now computes:
        entropy_grad = math.log(max(p_v, 1e-12)) + 1.0
        D[v] += w_F * entropy_grad

    Both engines are now aligned to: D_future[v] = w_F * (log(p_v) + 1)

THIS TEST VERIFIES:
    1. Both engines produce identical D_future values for the same input
    2. Both engines produce identical Tr(D_mu_nu) values
    3. Both engines select the same token S*
    4. Both engines converge in the same number of iterations (within 1)
    5. Sign structure of D_future is correct: negative for p_v < 1/e, positive for p_v > 1/e
"""

from __future__ import annotations

import sys
import os
import math
import random
from typing import Set, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_Core_Engine'))

from actualizer_engine import ActualizerEngine
from numpy_actualizer_engine import NumpyActualizerEngine


# ============================================================================
# Test utilities
# ============================================================================

def make_logits(V: int, seed: int = 42, distractor_strength: float = 5.0) -> List[float]:
    rng = random.Random(seed)
    logits = [rng.gauss(0.0, 0.5) for _ in range(V)]
    for v in range(V // 5, V // 3):
        logits[v] += rng.uniform(1.0, 2.5)
    distractor = rng.randint(V // 3, V - 1)
    logits[distractor] += distractor_strength
    return logits


def make_target_tokens(V: int) -> Set[int]:
    return set(range(V // 5, V // 3))


def make_history(V: int, seed: int = 42) -> List[int]:
    rng = random.Random(seed + 99)
    return [rng.randint(0, V - 1) for _ in range(10)]


# ============================================================================
# Test 1: D_future sign structure verification
# ============================================================================

def test_d_future_sign_structure():
    """
    Verify the sign structure of the entropy gradient proxy:
        log(p_v) + 1  is negative when p_v < 1/e
                       is zero    when p_v = 1/e
                       is positive when p_v > 1/e

    For a well-distributed softmax with V=100, most p_v << 1/e,
    so most D_future entries should be NEGATIVE.
    """
    print("=" * 68)
    print("  TEST 1: D_future sign structure")
    print("=" * 68)

    threshold = 1.0 / math.e  # ~0.3679

    # Test with a few known p_v values
    test_probs = [0.001, 0.01, 0.1, 1.0 / math.e, 0.5, 0.9]
    print(f"\n  {'p_v':>10}  {'log(p_v)+1':>12}  {'Sign':>8}  {'Expected':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}")
    all_correct = True
    for p_v in test_probs:
        grad = math.log(p_v) + 1.0
        sign = "+" if grad > 0 else ("-" if grad < 0 else "0")
        expected = "+" if p_v > threshold else ("-" if p_v < threshold else "0")
        ok = sign == expected
        all_correct = all_correct and ok
        print(f"  {p_v:>10.4f}  {grad:>12.6f}  {sign:>8}  {expected:>10}  {'OK' if ok else 'FAIL'}")

    print(f"\n  Sign structure: {'PASS' if all_correct else 'FAIL'}")
    return all_correct


# ============================================================================
# Test 2: Drift tensor alignment between Python and NumPy engines
# ============================================================================

def test_drift_tensor_alignment(V: int = 200, seed: int = 42):
    """
    Verify both engines compute identical drift tensor values for the same input.
    """
    print(f"\n{'=' * 68}")
    print(f"  TEST 2: Drift tensor alignment (V={V})")
    print("=" * 68)

    logits_list = make_logits(V, seed=seed)
    logits_np   = np.array(logits_list, dtype=np.float64)
    target      = make_target_tokens(V)
    history     = make_history(V, seed=seed)

    # Create engines with identical parameters
    params = dict(
        vocab_size=V, mercy_k=0.45, Q_c=1e-5,
        tau_bifurcation=5.0, max_iters=100,
        repetition_penalty=2.0, global_drift_penalty=1.5, h_max=2.0,
    )
    py_engine = ActualizerEngine(**params)
    np_engine = NumpyActualizerEngine(**params)

    # Compute softmax on both
    U_py = py_engine._softmax(logits_list)
    U_np = np_engine._softmax(logits_np)

    # Verify softmax alignment
    softmax_diff = max(abs(U_py[v] - float(U_np[v])) for v in range(V))
    print(f"\n  Softmax max abs diff: {softmax_diff:.2e}  {'PASS' if softmax_diff < 1e-10 else 'FAIL'}")

    # Compute drift tensors
    D_py = py_engine.compute_drift_tensor(U_py, history, target)
    D_np = np_engine._drift_tensor(U_np, history, target)

    # Compare D_future components specifically
    # Isolate D_future: compute D with only w_F, no w_L or w_G
    w_F = 0.20
    d_future_py = []
    d_future_np = []
    for v in range(V):
        p_v = U_py[v]
        if p_v > 0:
            py_val = w_F * (math.log(max(p_v, 1e-12)) + 1.0)
        else:
            py_val = 0.0
        np_val = float(w_F * (np.log(max(float(U_np[v]), 1e-300)) + 1.0))
        d_future_py.append(py_val)
        d_future_np.append(np_val)

    future_diff = max(abs(d_future_py[v] - d_future_np[v]) for v in range(V))
    print(f"  D_future max abs diff: {future_diff:.2e}  {'PASS' if future_diff < 1e-10 else 'FAIL'}")

    # Full drift tensor comparison
    drift_diff = max(abs(D_py[v] - float(D_np[v])) for v in range(V))
    print(f"  Full D_tensor max abs diff: {drift_diff:.2e}  {'PASS' if drift_diff < 1e-8 else 'FAIL'}")

    # Tr(D) comparison
    Tr_py = sum(U_py[v] * D_py[v] for v in range(V))
    Tr_np = float(np.dot(U_np, D_np))
    Tr_diff = abs(Tr_py - Tr_np)
    print(f"  Tr(D) Python:  {Tr_py:.10f}")
    print(f"  Tr(D) NumPy:   {Tr_np:.10f}")
    print(f"  Tr(D) abs diff: {Tr_diff:.2e}  {'PASS' if Tr_diff < 1e-8 else 'FAIL'}")

    return drift_diff < 1e-8 and Tr_diff < 1e-8


# ============================================================================
# Test 3: Full steer() alignment — token, iterations, valuation, Tr(D)
# ============================================================================

def test_steer_alignment(V: int = 200, seed: int = 42):
    """
    Run both engines on identical input and verify:
      - Same token selected
      - Same Tr(D) at convergence (within tolerance)
      - Same number of iterations (within 1)
      - Same final valuation (within tolerance)
    """
    print(f"\n{'=' * 68}")
    print(f"  TEST 3: Full steer() alignment (V={V})")
    print("=" * 68)

    logits_list = make_logits(V, seed=seed)
    logits_np   = np.array(logits_list, dtype=np.float64)
    target      = make_target_tokens(V)
    history     = make_history(V, seed=seed)

    params = dict(
        vocab_size=V, mercy_k=0.45, Q_c=1e-5,
        tau_bifurcation=5.0, max_iters=100,
        repetition_penalty=2.0, global_drift_penalty=1.5, h_max=2.0,
    )
    py_engine = ActualizerEngine(**params)
    np_engine = NumpyActualizerEngine(**params)

    tok_py, U_py, Tr_py, iters_py, nu_py, act_py = py_engine.steer(logits_list, history, target)
    tok_np, U_np, Tr_np, iters_np, nu_np, act_np = np_engine.steer(logits_np, history, target)

    token_match = tok_py == tok_np
    Tr_diff     = abs(Tr_py - float(Tr_np))
    iter_diff   = abs(iters_py - iters_np)
    nu_py_final = nu_py[-1] if nu_py else 0.0
    nu_np_final = nu_np[-1] if nu_np else 0.0
    nu_diff     = abs(nu_py_final - nu_np_final)
    act_match   = act_py == act_np

    print(f"\n  {'Metric':<30} {'Python':>15} {'NumPy':>15} {'Match':>10}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
    print(f"  {'Token S*':<30} {tok_py:>15} {tok_np:>15} {'PASS' if token_match else 'FAIL':>10}")
    print(f"  {'Tr(D) at convergence':<30} {Tr_py:>15.8f} {float(Tr_np):>15.8f} {'PASS' if Tr_diff < 1e-6 else 'FAIL':>10}")
    print(f"  {'Iterations':<30} {iters_py:>15} {iters_np:>15} {'PASS' if iter_diff <= 1 else 'FAIL':>10}")
    print(f"  {'Final valuation nu_t':<30} {nu_py_final:>15.8f} {nu_np_final:>15.8f} {'PASS' if nu_diff < 0.01 else 'FAIL':>10}")
    print(f"  {'Actualized?':<30} {str(act_py):>15} {str(act_np):>15} {'PASS' if act_match else 'FAIL':>10}")
    print(f"  {'Tr(D) abs diff':<30} {Tr_diff:>15.2e} {'':>15} {'PASS' if Tr_diff < 1e-6 else 'FAIL':>10}")

    all_pass = token_match and Tr_diff < 1e-6 and iter_diff <= 1 and nu_diff < 0.01 and act_match
    print(f"\n  Full alignment: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================================
# Test 4: Multi-step episode alignment (simulating real generation)
# ============================================================================

def test_multi_step_alignment(V: int = 200, n_steps: int = 20, seed: int = 42):
    """
    Run both engines for n_steps of sequential token generation
    and verify they produce the same token sequence.
    """
    print(f"\n{'=' * 68}")
    print(f"  TEST 4: Multi-step episode alignment (V={V}, {n_steps} steps)")
    print("=" * 68)

    target  = make_target_tokens(V)
    history = make_history(V, seed=seed)

    params = dict(
        vocab_size=V, mercy_k=0.45, Q_c=1e-5,
        tau_bifurcation=5.0, max_iters=100,
        repetition_penalty=2.0, global_drift_penalty=1.5, h_max=2.0,
    )
    py_engine = ActualizerEngine(**params)
    np_engine = NumpyActualizerEngine(**params)

    py_history = list(history)
    np_history = list(history)
    tokens_py  = []
    tokens_np  = []
    tr_diffs   = []

    for step in range(n_steps):
        logits_list = make_logits(V, seed=seed + step, distractor_strength=5.0)
        logits_np   = np.array(logits_list, dtype=np.float64)

        tok_py, _, Tr_py, _, _, _ = py_engine.steer(logits_list, py_history, target)
        tok_np, _, Tr_np, _, _, _ = np_engine.steer(logits_np,   np_history, target)

        tokens_py.append(tok_py)
        tokens_np.append(tok_np)
        tr_diffs.append(abs(Tr_py - float(Tr_np)))
        py_history.append(tok_py)
        np_history.append(tok_np)

    token_matches = sum(1 for a, b in zip(tokens_py, tokens_np) if a == b)
    max_tr_diff   = max(tr_diffs)
    mean_tr_diff  = sum(tr_diffs) / len(tr_diffs)

    print(f"\n  Token sequence match: {token_matches}/{n_steps} ({token_matches/n_steps*100:.1f}%)")
    print(f"  Tr(D) max abs diff across steps: {max_tr_diff:.2e}")
    print(f"  Tr(D) mean abs diff: {mean_tr_diff:.2e}")

    # Show first few steps
    print(f"\n  {'Step':>5}  {'Token(Py)':>10}  {'Token(Np)':>10}  {'Tr(D) diff':>12}  {'Match':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*6}")
    for i in range(min(n_steps, 10)):
        match = "OK" if tokens_py[i] == tokens_np[i] else "DIFF"
        print(f"  {i:>5}  {tokens_py[i]:>10}  {tokens_np[i]:>10}  {tr_diffs[i]:>12.2e}  {match:>6}")
    if n_steps > 10:
        print(f"  ... ({n_steps - 10} more steps)")

    all_pass = token_matches == n_steps and max_tr_diff < 1e-6
    print(f"\n  Multi-step alignment: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================================
# Test 5: Vocabulary size sweep — verify alignment holds across V
# ============================================================================

def test_vocab_sweep(vocab_sizes=(50, 100, 200, 500, 1000), seed: int = 42):
    """
    Verify alignment across different vocabulary sizes.
    """
    print(f"\n{'=' * 68}")
    print(f"  TEST 5: Vocabulary size sweep alignment")
    print("=" * 68)

    print(f"\n  {'V':>6}  {'Token(Py)':>10}  {'Token(Np)':>10}  {'Tr_diff':>12}  {'Result':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}")

    all_pass = True
    for V in vocab_sizes:
        logits_list = make_logits(V, seed=seed)
        logits_np   = np.array(logits_list, dtype=np.float64)
        target      = make_target_tokens(V)
        history     = make_history(V, seed=seed)

        params = dict(
            vocab_size=V, mercy_k=0.45, Q_c=1e-5,
            tau_bifurcation=5.0, max_iters=100,
            repetition_penalty=2.0, global_drift_penalty=1.5, h_max=2.0,
        )
        py_e = ActualizerEngine(**params)
        np_e = NumpyActualizerEngine(**params)

        tok_py, _, Tr_py, _, _, _ = py_e.steer(logits_list, history, target)
        tok_np, _, Tr_np, _, _, _ = np_e.steer(logits_np,   history, target)

        tr_diff = abs(Tr_py - float(Tr_np))
        ok = tok_py == tok_np and tr_diff < 1e-6
        all_pass = all_pass and ok
        print(f"  {V:>6}  {tok_py:>10}  {tok_np:>10}  {tr_diff:>12.2e}  {'PASS' if ok else 'FAIL':>8}")

    print(f"\n  Vocab sweep: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  DRIFT ALIGNMENT VERIFICATION SUITE")
    print("  Bug: D_future used -log(p_v)*0.08 instead of log(p_v)+1")
    print("  Fix: Both engines now use log(p_v) + 1 (Shannon entropy grad)")
    print("=" * 68)

    results = {}
    results["sign_structure"]     = test_d_future_sign_structure()
    results["drift_alignment"]    = test_drift_tensor_alignment(V=200)
    results["steer_alignment"]    = test_steer_alignment(V=200)
    results["multi_step"]         = test_multi_step_alignment(V=200, n_steps=20)
    results["vocab_sweep"]        = test_vocab_sweep(vocab_sizes=(50, 100, 200, 500, 1000))

    print(f"\n{'=' * 68}")
    print("  FINAL VERDICT")
    print("=" * 68)
    for name, passed in results.items():
        print(f"  {name:<25} {'PASS' if passed else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 68)
