"""
diept.py — Dynamic Information and Energy Phase Transformation (DIEPT)
======================================================================
Author : Antigravity (Advanced Agentic Coding)

Implements Stage 3 of the C(R) architecture (DIEPT).
Formalizes the inference phase angle and handles linguistic markers mapping
from Question Operators as described in the CKT framework.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict

class DIEPTState:
    """
    The Complex Justification State Z = A + iB.
    A represents the grounded (verified) knowledge subspace.
    B represents the speculative (imaginary) subspace.
    """
    def __init__(self, A: List[float], B: List[float]) -> None:
        if len(A) != len(B):
            raise ValueError("Subspaces A and B must have the same dimension.")
        self.A = A
        self.B = B
        
    @property
    def norm_A(self) -> float:
        return math.sqrt(sum(a**2 for a in self.A))
        
    @property
    def norm_B(self) -> float:
        return math.sqrt(sum(b**2 for b in self.B))
        
    @property
    def theta(self) -> float:
        """
        Computes the phase angle theta = arctan(||B|| / ||A||).
        This serves as the Mercy compliance signal.
        """
        nA = self.norm_A
        if nA == 0:
            return math.pi / 2.0 if self.norm_B > 0 else 0.0
        return math.atan(self.norm_B / nA)

    def get_linguistic_marker(self, theta_target: float) -> str:
        """
        Maps the phase angle to a linguistic marker based on the Mercy threshold.
        """
        t = self.theta
        if t <= theta_target * 0.25:
            return "Factual / Confirmed:"
        elif t <= theta_target * 0.75:
            return "Likely / High Confidence:"
        elif t <= theta_target:
            return "Hypothesis / Let's explore:"
        else:
            return "[Speculative / Noise]"


class CausalDensityFunction:
    """
    Partial Justice Prime compliance proxy at inference.
    C(x) = logical weight required to justify information at causal depth x.
    """
    def __init__(self, base_weight: float = 1.0, depth_scaling: float = 1.1) -> None:
        self.base_weight = base_weight
        self.depth_scaling = depth_scaling
        
    def evaluate(self, depth_x: int) -> float:
        """Returns the logical justification weight required at depth x."""
        return self.base_weight * (self.depth_scaling ** depth_x)


class InformationCurvatureLoss:
    """
    Order Prime Enforcement: discrete approximation of the contour integral
    to detect unresolved logical contradictions (cycles or backtracking).
    """
    @staticmethod
    def detect_residues(causal_chain: List[int]) -> float:
        """
        A discrete proxy for the contour integral over the inference space.
        If a token appears more than once, it suggests a cycle/contradiction
        in the causal derivation, yielding a non-zero residue.
        """
        residue = 0.0
        seen = set()
        for token in causal_chain:
            if token in seen:
                # Contradiction/cycle detected
                residue += 1.0
            seen.add(token)
        return residue

class QuestionOperatorParser:
    """
    Stage A0 Pre-processor: Maps Question Operator to Prime Architecture.
    f(QuestionOperator) -> (DomainLens, PrimeVector, theta_target)
    """
    @staticmethod
    def parse_operator(operator: str) -> Tuple[str, List[float], float]:
        op = operator.strip().lower()
        if op == "why":
            # Justice dominance, near-zero theta_target
            return ("Causal_Law", [0.8, 1.0, 0.1, 0.9, 0.8], 0.15)
        elif op == "what if":
            # Creativity mode (K -> M), elevated theta_target
            return ("OpenCI_Speculation", [0.4, 0.2, 0.9, 0.8, 0.5], 1.20)
        elif op == "should":
            # Wisdom mode (all primes active)
            return ("Wisdom_Evaluation", [0.9, 0.9, 0.9, 0.9, 0.9], 0.60)
        else:
            # Default General Purpose
            return ("General_Purpose", [0.5, 0.5, 0.5, 0.5, 0.5], 0.70)
