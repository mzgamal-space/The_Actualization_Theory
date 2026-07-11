"""
test_upgraded_engine.py — Unit Tests for Upgraded Actualizer Engine & MCE Class
================================================================================
Author : Antigravity (Advanced Agentic Coding)

Unit tests to verify:
  1. Causal collapse in Pipeline A (Justice).
  2. Speculative quarantine in Pipeline C (Negentropy/DIEPT).
  3. CAKI index calculation correctness.
  4. Crystallization into the MCE class.
  5. Isomorphic anchoring matching against dynamic MCE objects.
"""

from __future__ import annotations
import math
import unittest
from ckt_actualizer.models.mce import MCE, ReferenceDomain
from ckt_actualizer.models.thought import CandidateThought
from ckt_actualizer.core.filters import EpistemicVerificationSuite
from ckt_actualizer.core.fdsa import FractalDeductionSearch, VectorizedFDSAPruner
from ckt_actualizer.core.engine import UpgradedActualizerEngine

class TestUpgradedActualizerEngine(unittest.TestCase):
    def setUp(self) -> None:
        # Causal rules
        self.cwf = {
            (1, 2): 0.0,
            (2, 3): 0.0,
            (2, 9): float('inf')  # Forbidden causal transition
        }
        self.vocab_size = 1000
        self.engine = UpgradedActualizerEngine(
            vocab_size=self.vocab_size,
            cwf_penalty_matrix=self.cwf,
            k_contractive=0.45,
            Q_c=1e-5,
            theta_target=0.70,
            caki_threshold=0.80
        )

    def test_pipeline_a_causal_collapse(self) -> None:
        # Valid thought
        valid_thought = CandidateThought(causal_chain=[1, 2, 3], propensity=1.0)
        passed_valid = self.engine.verifier.run_pipeline_a(valid_thought)
        self.assertTrue(passed_valid)
        self.assertGreater(valid_thought.propensity, 0.0)
        self.assertGreater(valid_thought.primes["Justice"], 0.0)

        # Invalid thought (violates CWF)
        invalid_thought = CandidateThought(causal_chain=[1, 2, 9], propensity=1.0)
        passed_invalid = self.engine.verifier.run_pipeline_a(invalid_thought)
        self.assertFalse(passed_invalid)
        self.assertEqual(invalid_thought.propensity, 0.0)
        self.assertEqual(invalid_thought.primes["Justice"], 0.0)
        self.assertEqual(invalid_thought.primes["Power"], 0.0)

    def test_pipeline_c_negentropy_quarantine(self) -> None:
        # Grounded thought: low speculative component
        grounded_thought = CandidateThought(
            causal_chain=[1, 2, 3],
            propensity=1.0,
            grounded_subspace=[1.0, 1.0],
            speculative_subspace=[0.1, 0.1]
        )
        passed_grounded = self.engine.verifier.run_pipeline_c(grounded_thought)
        self.assertTrue(passed_grounded)
        self.assertGreater(grounded_thought.propensity, 0.0)
        self.assertGreater(grounded_thought.primes["Mercy"], 0.0)

        # Speculative thought: speculative components dominate grounded ones
        speculative_thought = CandidateThought(
            causal_chain=[1, 2, 3],
            propensity=1.0,
            grounded_subspace=[1.0, 1.0],
            speculative_subspace=[2.0, 2.0]  # phase angle theta = arctan(sqrt(8)/sqrt(2)) = arctan(2) ~ 1.1 rad > 0.70 rad
        )
        passed_speculative = self.engine.verifier.run_pipeline_c(speculative_thought)
        self.assertFalse(passed_speculative)
        self.assertEqual(speculative_thought.propensity, 0.0)
        self.assertEqual(speculative_thought.primes["Mercy"], 0.0)
        self.assertEqual(speculative_thought.primes["Knowledge"], 0.0)

    def test_caki_calculation(self) -> None:
        thought = CandidateThought(causal_chain=[1, 2, 3], propensity=1.0)
        caki, L_viol, R, I_eff = self.engine.calculate_caki(thought)
        
        # Zero violations, zero duplicate redundancy
        self.assertEqual(L_viol, 0.0)
        self.assertEqual(R, 1.0)
        self.assertEqual(I_eff, 3.0)
        # K_acc = (3.0 / 1.0) * exp(0.0) = 3.0
        # normalized caki = 3.0 / 3.0 = 1.0
        self.assertEqual(caki, 1.0)

        # Add duplicate token (redundancy)
        redundant_thought = CandidateThought(causal_chain=[1, 2, 2], propensity=1.0)
        caki_r, L_viol_r, R_r, I_eff_r = self.engine.calculate_caki(redundant_thought)
        self.assertEqual(L_viol_r, 0.0)
        # duplicates = 1, R = 1 + 1/3 = 1.3333
        self.assertAlmostEqual(R_r, 1.3333333333)
        self.assertEqual(I_eff_r, 3.0)
        # K_acc = 3.0 / 1.3333 = 2.25
        # caki = 2.25 / 3.0 = 0.75
        self.assertAlmostEqual(caki_r, 0.75)

    def test_crystallization_into_mce(self) -> None:
        # Fully grounded and verified thought
        thought = CandidateThought(
            causal_chain=[1, 2, 3],
            propensity=1.0,
            grounded_subspace=[1.0] * 8,
            speculative_subspace=[0.0] * 8
        )
        
        # Verify and crystallize (simulating a negative delta_C_R cost reduction)
        library_size_before = len(self.engine.fdsa_search.library)
        is_cryst, mce_obj, caki = self.engine.verify_and_crystallize(
            thought=thought,
            delta_C_R=-0.2,
            domain_name="TestDomain",
            description="Testing crystallization logic."
        )
        
        self.assertTrue(is_cryst)
        self.assertIsNotNone(mce_obj)
        self.assertEqual(len(self.engine.fdsa_search.library), library_size_before + 1)
        self.assertIsInstance(mce_obj, MCE)
        self.assertIsInstance(mce_obj, ReferenceDomain)
        self.assertEqual(mce_obj.causal_chain, [1, 2, 3])
        self.assertGreater(mce_obj.mass, 0.0)
        self.assertGreater(mce_obj.complexity, 0.0)
        self.assertEqual(mce_obj.entropy, 0.0)
        
        # Derived k contraction constraint check
        self.assertTrue(0.15 <= mce_obj.k <= 0.65)

    def test_isomorphic_anchoring_to_dynamic_mce(self) -> None:
        # Add dynamic MCE to library
        chain = [1, 2, 3]
        mce_mock = MCE(
            name="MCE_Physics_Test",
            causal_chain=chain,
            prime_profile=[0.90, 0.90, 0.10, 0.90, 0.85],  # Custom physics profile
            mass=3.0,
            complexity=0.85,
            entropy=0.05
        )
        self.engine.fdsa_search.add_reference_domain(mce_mock)

        # Prompt profile matching the mock MCE
        P_prompt = [0.88, 0.89, 0.15, 0.92, 0.80]
        best_domain, similarity = self.engine.fdsa_search.isomorphic_anchoring(P_prompt)
        
        self.assertEqual(best_domain.name, "MCE_Physics_Test")
        self.assertGreater(similarity, 0.98)
        self.assertEqual(best_domain.k, mce_mock.k)

if __name__ == "__main__":
    unittest.main()
