"""
tests/test_ckt_core.py — Full Test Suite for 07_CKT_Core
==========================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Covers:
  1.  DIEPTState — phase angle, linguistic markers.
  2.  CausalDensityFunction — depth-weighted evaluation.
  3.  InformationCurvatureLoss — cycle detection.
  4.  QuestionOperatorParser — Stage A0 mapping.
  5.  ReferenceDomain / MCE — construction and k derivation.
  6.  FractalDeductionSearch — isomorphic anchoring and fractal dimension.
  7.  EpistemicVerificationSuite — Pipeline A (Justice collapse).
  8.  EpistemicVerificationSuite — Pipeline C (Negentropy quarantine).
  9.  EpistemicVerificationSuite — C(R) cost functional.
  10. UpgradedActualizerEngine — CAKI Theorem 7 bound.
  11. UpgradedActualizerEngine — Crystallization into MCE.
  12. UpgradedActualizerEngine — Dynamic isomorphic anchoring.
  13. UpgradedActualizerEngine — process_query end-to-end lifecycle.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import unittest

from diept import (
    DIEPTState,
    CausalDensityFunction,
    InformationCurvatureLoss,
    QuestionOperatorParser,
)
from mce import ReferenceDomain, MCE
from thought import CandidateThought
from fdsa import FractalDeductionSearch, VectorizedFDSAPruner
from filters import EpistemicVerificationSuite
from engine import UpgradedActualizerEngine


# ---------------------------------------------------------------------------
# Helper: build a standard test engine
# ---------------------------------------------------------------------------

def make_engine(vocab_size=1000, caki_threshold=0.50, delta_finite=0.5):
    cwf = {
        (1, 2): 0.0,
        (2, 3): 0.0,
        (2, 9): float("inf"),   # Forbidden causal transition
    }
    return UpgradedActualizerEngine(
        vocab_size=vocab_size,
        cwf_penalty_matrix=cwf,
        k_contractive=0.45,
        Q_c=1e-5,
        theta_target=0.70,
        caki_threshold=caki_threshold,
        delta_finite=delta_finite,
    )


# ===========================================================================
class TestDIEPTState(unittest.TestCase):

    def test_purely_grounded(self):
        s = DIEPTState([1.0, 1.0], [0.0, 0.0])
        self.assertAlmostEqual(s.theta, 0.0)

    def test_purely_speculative(self):
        s = DIEPTState([0.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(s.theta, math.pi / 2)

    def test_equal_subspaces(self):
        s = DIEPTState([1.0, 1.0], [1.0, 1.0])
        self.assertAlmostEqual(s.theta, math.pi / 4)

    def test_linguistic_marker_factual(self):
        s = DIEPTState([10.0], [0.1])   # very grounded
        self.assertEqual(s.get_linguistic_marker(0.70), "Factual / Confirmed:")

    def test_linguistic_marker_speculative(self):
        s = DIEPTState([1.0], [1.0])    # theta ~ 0.785, above 0.70
        self.assertEqual(s.get_linguistic_marker(0.70), "[Speculative / Noise]")

    def test_mismatched_dimensions_raise(self):
        with self.assertRaises(ValueError):
            DIEPTState([1.0], [1.0, 2.0])


# ===========================================================================
class TestCausalDensityFunction(unittest.TestCase):

    def test_depth_zero(self):
        cdf = CausalDensityFunction(base_weight=1.0, depth_scaling=1.1)
        self.assertAlmostEqual(cdf.evaluate(0), 1.0)

    def test_depth_one(self):
        cdf = CausalDensityFunction(base_weight=1.0, depth_scaling=1.1)
        self.assertAlmostEqual(cdf.evaluate(1), 1.1)

    def test_depth_two(self):
        cdf = CausalDensityFunction(base_weight=1.0, depth_scaling=1.1)
        self.assertAlmostEqual(cdf.evaluate(2), 1.21)


# ===========================================================================
class TestInformationCurvatureLoss(unittest.TestCase):

    def test_no_cycles(self):
        self.assertEqual(
            InformationCurvatureLoss.detect_residues([10, 20, 30, 40]), 0.0
        )

    def test_one_cycle(self):
        self.assertEqual(
            InformationCurvatureLoss.detect_residues([10, 20, 30, 20, 40]), 1.0
        )

    def test_two_cycles(self):
        self.assertEqual(
            InformationCurvatureLoss.detect_residues([10, 20, 10, 30, 20]), 2.0
        )


# ===========================================================================
class TestQuestionOperatorParser(unittest.TestCase):

    def test_why_operator(self):
        domain, primes, theta = QuestionOperatorParser.parse_operator("why")
        self.assertEqual(domain, "Causal_Law")
        self.assertAlmostEqual(theta, 0.15)
        self.assertEqual(primes[1], 1.0)   # Justice dominance

    def test_what_if_operator(self):
        domain, primes, theta = QuestionOperatorParser.parse_operator("what if")
        self.assertEqual(domain, "OpenCI_Speculation")
        self.assertAlmostEqual(theta, 1.20)

    def test_should_operator(self):
        domain, primes, theta = QuestionOperatorParser.parse_operator("should")
        self.assertEqual(domain, "Wisdom_Evaluation")
        self.assertAlmostEqual(theta, 0.60)

    def test_case_insensitive(self):
        domain, _, _ = QuestionOperatorParser.parse_operator("WHY")
        self.assertEqual(domain, "Causal_Law")

    def test_default_operator(self):
        domain, primes, theta = QuestionOperatorParser.parse_operator("hello")
        self.assertEqual(domain, "General_Purpose")
        self.assertAlmostEqual(theta, 0.70)


# ===========================================================================
class TestReferenceDomainMCE(unittest.TestCase):

    def test_reference_domain_k_bounds(self):
        with self.assertRaises(ValueError):
            ReferenceDomain("bad", [1.0] * 5, k=0.0)
        with self.assertRaises(ValueError):
            ReferenceDomain("bad", [1.0] * 5, k=1.0)

    def test_mce_k_derived(self):
        mce = MCE(
            "Test", [1, 2, 3],
            prime_profile=[0.9, 0.9, 0.9, 0.9, 0.9],
            mass=3.0, complexity=0.9, entropy=0.05
        )
        self.assertTrue(0.15 <= mce.k <= 0.65)

    def test_mce_inherits_reference_domain(self):
        mce = MCE(
            "Physics", [10, 20, 30],
            prime_profile=[0.8, 0.9, 0.2, 0.8, 0.7],
            mass=5.0, complexity=0.5, entropy=0.0
        )
        self.assertIsInstance(mce, ReferenceDomain)


# ===========================================================================
class TestFDSA(unittest.TestCase):

    def test_isomorphic_anchoring(self):
        fdsa = FractalDeductionSearch()
        domain, sim = fdsa.isomorphic_anchoring([0.60, 0.10, 0.10, 0.10, 0.10])
        self.assertEqual(domain.name, "Fermat_Least_Time")
        self.assertGreater(sim, 0.95)

    def test_dynamic_mce_anchoring(self):
        fdsa = FractalDeductionSearch()
        mce  = MCE(
            "Physics_Gravity", [1, 2, 3],
            prime_profile=[0.95, 0.95, 0.05, 0.90, 0.85],
            mass=3.0, complexity=0.85, entropy=0.0
        )
        fdsa.add_reference_domain(mce)
        domain, sim = fdsa.isomorphic_anchoring([0.92, 0.94, 0.08, 0.88, 0.83])
        self.assertEqual(domain.name, "Physics_Gravity")
        self.assertGreater(sim, 0.98)

    def test_fractal_dimension_range(self):
        D = FractalDeductionSearch.fractal_dimension(1000, 0.45)
        # D = ln(1000)/ln(1/0.45) ~ 8.7
        self.assertGreater(D, 5.0)
        self.assertLess(D, 20.0)


# ===========================================================================
class TestEpistemicVerificationSuite(unittest.TestCase):

    def setUp(self):
        cwf = {(1, 2): 0.0, (2, 3): 0.0, (2, 9): float("inf")}
        self.suite = EpistemicVerificationSuite(cwf_penalty_matrix=cwf)

    # Pipeline A

    def test_valid_chain_passes_pipeline_a(self):
        t = CandidateThought([1, 2, 3], propensity=1.0)
        self.assertTrue(self.suite.run_pipeline_a(t))
        self.assertGreater(t.propensity, 0.0)
        self.assertGreater(t.primes["Justice"], 0.0)

    def test_causal_violation_collapses(self):
        t = CandidateThought([1, 2, 9], propensity=1.0)
        self.assertFalse(self.suite.run_pipeline_a(t))
        self.assertEqual(t.propensity, 0.0)
        self.assertEqual(t.primes["Justice"], 0.0)
        self.assertEqual(t.primes["Power"], 0.0)

    # Pipeline C

    def test_grounded_thought_passes_pipeline_c(self):
        t = CandidateThought([1, 2, 3], propensity=1.0,
                              diept_state=DIEPTState([5.0, 5.0], [0.1, 0.1]))
        self.assertTrue(self.suite.run_pipeline_c(t))
        self.assertGreater(t.primes["Mercy"], 0.0)

    def test_speculative_thought_quarantined(self):
        # theta = arctan(sqrt(8)/sqrt(2)) = arctan(2) ~ 1.1 > 0.70
        t = CandidateThought([1, 2, 3], propensity=1.0,
                              diept_state=DIEPTState([1.0, 1.0], [2.0, 2.0]))
        self.assertFalse(self.suite.run_pipeline_c(t))
        self.assertEqual(t.propensity, 0.0)
        self.assertEqual(t.primes["Mercy"], 0.0)

    # Justice Dominance

    def test_justice_dominance_enforced(self):
        cwf = {}
        with self.assertRaises(Exception):
            EpistemicVerificationSuite(cwf, lambda_R=0.5, lambda_L=0.3, lambda_D=0.2)

    # C(R) cost

    def test_cr_cost_collapsed_thought(self):
        t = CandidateThought([1, 2, 9], propensity=0.0)
        cost = self.suite.evaluate_cost(t)
        self.assertEqual(cost, float("inf"))


# ===========================================================================
class TestUpgradedActualizerEngine(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine()

    # CAKI Theorem 7 bound

    def test_caki_never_reaches_one(self):
        t = CandidateThought([1, 2, 3], propensity=1.0)
        caki, _, _, _ = self.engine.calculate_caki(t)
        self.assertLess(caki, 1.0, "CAKI must never reach 1.0 (Theorem 7)")

    def test_caki_zero_loss_near_one(self):
        # Perfect thought — should approach but not reach 1.0
        t     = CandidateThought(list(range(20)), propensity=1.0)
        caki, L, R, _ = self.engine.calculate_caki(t)
        self.assertEqual(L, 0.0)
        self.assertEqual(R, 1.0)
        self.assertLess(caki, 1.0)
        self.assertGreater(caki, 0.90)  # should be high for long unique chain

    def test_caki_decreases_with_redundancy(self):
        t_clean = CandidateThought([1, 2, 3], propensity=1.0)
        t_redun = CandidateThought([1, 2, 2], propensity=1.0)
        c_clean, _, _, _ = self.engine.calculate_caki(t_clean)
        c_redun, _, _, _ = self.engine.calculate_caki(t_redun)
        self.assertGreater(c_clean, c_redun)

    # Crystallization

    def test_crystallization_success(self):
        thought = CandidateThought(
            [1, 2, 3], propensity=1.0,
            diept_state=DIEPTState([5.0] * 4, [0.05] * 4)
        )
        n_before = len(self.engine.fdsa_search.library)
        is_cryst, mce, caki = self.engine.verify_and_crystallize(
            thought=thought, delta_C_R=-0.3,
            domain_name="TestDomain", description="Unit test."
        )
        self.assertTrue(is_cryst)
        self.assertIsNotNone(mce)
        self.assertIsInstance(mce, MCE)
        self.assertEqual(len(self.engine.fdsa_search.library), n_before + 1)
        self.assertLess(caki, 1.0)

    def test_crystallization_fails_on_violation(self):
        thought = CandidateThought([1, 2, 9], propensity=1.0)
        is_cryst, mce, _ = self.engine.verify_and_crystallize(
            thought=thought, delta_C_R=-0.3, domain_name="Bad"
        )
        self.assertFalse(is_cryst)
        self.assertIsNone(mce)

    # process_query lifecycle

    def test_process_query_factual(self):
        marker, is_cryst, mce, caki = self.engine.process_query(
            query="Why did the system drop the object?",
            initial_history=[10],
            target_tokens={30},
            simulated_diept_a=[5.0, 5.0, 4.0],
            simulated_diept_b=[0.2, 0.1, 0.2],
        )
        self.assertIn("Factual", marker)
        self.assertLess(caki, 1.0)

    def test_process_query_speculative(self):
        marker, is_cryst, mce, caki = self.engine.process_query(
            query="What if gravity inverted?",
            initial_history=[99],
            target_tokens={30},
            simulated_diept_a=[1.0, 1.0, 1.0],
            simulated_diept_b=[3.0, 4.0, 3.5],
        )
        self.assertIn("Speculative", marker)


# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
