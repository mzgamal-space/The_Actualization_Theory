"""
test_diept.py — Unit Tests for DIEPT and Question Operators
=============================================================
"""

import math
import unittest
from ckt_actualizer.core.diept import (
    DIEPTState,
    CausalDensityFunction,
    InformationCurvatureLoss,
    QuestionOperatorParser
)

class TestDIEPT(unittest.TestCase):
    
    def test_diept_state_phase_angle(self):
        # Purely grounded
        state1 = DIEPTState([1.0, 1.0], [0.0, 0.0])
        self.assertEqual(state1.theta, 0.0)
        self.assertEqual(state1.get_linguistic_marker(0.70), "Factual / Confirmed:")
        
        # Equal grounding and speculation (theta = arctan(1) = pi/4 ~ 0.785)
        state2 = DIEPTState([1.0, 1.0], [1.0, 1.0])
        self.assertAlmostEqual(state2.theta, math.pi / 4)
        
        # Purely speculative (theta = pi/2)
        state3 = DIEPTState([0.0, 0.0], [1.0, 1.0])
        self.assertEqual(state3.theta, math.pi / 2)
        
    def test_linguistic_marker(self):
        theta_target = 0.70
        state1 = DIEPTState([1.0], [0.1]) # ~0.1 rad
        self.assertEqual(state1.get_linguistic_marker(theta_target), "Factual / Confirmed:")
        
        state2 = DIEPTState([1.0], [0.6]) # ~0.54 rad
        self.assertEqual(state2.get_linguistic_marker(theta_target), "Hypothesis / Let's explore:")
        
        state3 = DIEPTState([1.0], [1.0]) # ~0.785 rad
        self.assertEqual(state3.get_linguistic_marker(theta_target), "[Speculative / Noise]")
        
    def test_causal_density_function(self):
        cdf = CausalDensityFunction(base_weight=1.0, depth_scaling=1.1)
        self.assertAlmostEqual(cdf.evaluate(0), 1.0)
        self.assertAlmostEqual(cdf.evaluate(1), 1.1)
        self.assertAlmostEqual(cdf.evaluate(2), 1.21)

    def test_information_curvature_loss(self):
        chain_no_cycles = [10, 20, 30, 40]
        self.assertEqual(InformationCurvatureLoss.detect_residues(chain_no_cycles), 0.0)
        
        chain_with_cycles = [10, 20, 30, 20, 40]
        self.assertEqual(InformationCurvatureLoss.detect_residues(chain_with_cycles), 1.0)
        
    def test_question_operator_parser(self):
        # 'Why' should require Justice and low theta_target
        domain, primes, theta = QuestionOperatorParser.parse_operator("why")
        self.assertEqual(domain, "Causal_Law")
        self.assertEqual(theta, 0.15)
        
        # 'What if' should require Creativity and high theta_target
        domain, primes, theta = QuestionOperatorParser.parse_operator("what if")
        self.assertEqual(domain, "OpenCI_Speculation")
        self.assertEqual(theta, 1.20)

if __name__ == '__main__':
    unittest.main()
