"""
diept.py — Dynamic Information and Epistemic Phase Transformation (DIEPT)
==========================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT — Prime-Compliant Standard v4)
Code   : Antigravity (Advanced Agentic Coding)

Implements Stage 3 of the C(R) architecture (DIEPT).

Mathematical foundations
------------------------
  Complex Justification State  Z = A + iB
  Grounded subspace            A  (verified, real axis)
  Speculative subspace         iB (uncertain, imaginary axis)

  Phase angle (Mercy compliance signal)
      theta = arctan(||B|| / ||A||)           [radians, in [0, pi/2]]

  Theorem 7 bound (Unsimulability of Reality):
      CAKI = K_acc / I_max  ->  [0, 1)   (never reaches 1.0)

  Question Operator -> Epistemic Architecture Triple
      f(QuestionOperator) -> (DomainLens, PrimeVector, theta_target)
"""

from __future__ import annotations
import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# DIEPTState
# ---------------------------------------------------------------------------

class DIEPTState:
    """
    The Complex Justification State Z = A + iB.

    A represents the grounded (verified) knowledge subspace.
    B represents the speculative (imaginary) subspace.
    """

    def __init__(self, A: List[float], B: List[float]) -> None:
        if len(A) != len(B):
            raise ValueError("Subspaces A and B must have the same dimension.")
        self.A = list(A)
        self.B = list(B)

    # ---- norms ----

    @property
    def norm_A(self) -> float:
        return math.sqrt(sum(a ** 2 for a in self.A))

    @property
    def norm_B(self) -> float:
        return math.sqrt(sum(b ** 2 for b in self.B))

    # ---- phase angle ----

    @property
    def theta(self) -> float:
        """
        theta = arctan(||B|| / ||A||)
        Mercy compliance signal.  theta -> 0 = fully grounded;
                                  theta -> pi/2 = fully speculative.
        """
        nA = self.norm_A
        if nA == 0:
            return math.pi / 2.0 if self.norm_B > 0 else 0.0
        return math.atan(self.norm_B / nA)

    # ---- linguistic modality ----

    def get_linguistic_marker(self, theta_target: float) -> str:
        """
        Maps the phase angle to a linguistic marker based on the Mercy threshold.

        theta_target is set by the Question Operator (Stage A0).
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

    def __repr__(self) -> str:
        return (
            f"DIEPTState(||A||={self.norm_A:.4f}, ||B||={self.norm_B:.4f}, "
            f"theta={self.theta:.4f} rad)"
        )


# ---------------------------------------------------------------------------
# CausalDensityFunction  (Justice Prime proxy)
# ---------------------------------------------------------------------------

class CausalDensityFunction:
    """
    Partial Justice Prime compliance proxy at inference.

    C(x) = base_weight * depth_scaling^x

    Provides depth-weighted causal justification scoring that is computable
    from the inference architecture without requiring external ground truth.
    """

    def __init__(self, base_weight: float = 1.0, depth_scaling: float = 1.1) -> None:
        self.base_weight = base_weight
        self.depth_scaling = depth_scaling

    def evaluate(self, depth_x: int) -> float:
        """Returns the logical justification weight required at causal depth x."""
        return self.base_weight * (self.depth_scaling ** depth_x)


# ---------------------------------------------------------------------------
# InformationCurvatureLoss  (Order Prime enforcement)
# ---------------------------------------------------------------------------

class InformationCurvatureLoss:
    """
    Order Prime Enforcement: discrete approximation of the contour integral
    
        L_IC = sum(Distance) + |∮_C J(z) dz|

    The contour integral detects unresolved logical contradictions (cycles
    or backtracking) in the causal chain.  Non-zero residues flag Order
    Prime violations before C(R) scoring.

    Note: The continuous Cauchy Residue Theorem form is an open theoretical
    problem (PCS v4, §8).  This module implements the discrete proxy:
    repeated tokens in the causal chain signal a closed-loop contradiction.
    """

    @staticmethod
    def detect_residues(causal_chain: List[int]) -> float:
        """
        Discrete proxy for ∮_C J(z) dz.
        Returns 0.0 when no contradictions (ideal chain).
        Returns > 0.0 for every repeated token (cycle detected).
        """
        residue = 0.0
        seen: set = set()
        for token in causal_chain:
            if token in seen:
                residue += 1.0   # Contradiction / cycle
            seen.add(token)
        return residue


# ---------------------------------------------------------------------------
# QuestionOperatorParser  (Stage A0 Pre-processor)
# ---------------------------------------------------------------------------

class QuestionOperatorParser:
    """
    Stage A0 Pre-processor.

    Maps a question operator string to the Epistemic Architecture Triple:
        f(QuestionOperator) -> (DomainLens, PrimeVector, theta_target)

    Based on: Question_Operators_Prime_Selectors_Noureldin_2026
              (Conciseness Framework Series)
    """

    @staticmethod
    def parse_operator(operator: str) -> Tuple[str, List[float], float]:
        """
        Returns (domain_lens, prime_vector, theta_target).

        prime_vector indices: [Order, Justice, Mercy, Knowledge, Power]
        """
        op = operator.strip().lower()
        if op == "why":
            # Justice dominance; near-zero theta (causal-law anchor)
            return ("Causal_Law", [0.8, 1.0, 0.1, 0.9, 0.8], 0.15)
        elif op in ("what if", "what"):
            # Creativity mode (K -> M); elevated theta (OpenCI speculation)
            return ("OpenCI_Speculation", [0.4, 0.2, 0.9, 0.8, 0.5], 1.20)
        elif op == "should":
            # Wisdom mode (all five Primes active simultaneously)
            return ("Wisdom_Evaluation", [0.9, 0.9, 0.9, 0.9, 0.9], 0.60)
        elif op == "how":
            # Knowledge + Order dominant; moderate theta
            return ("Procedural_Knowledge", [0.7, 0.6, 0.3, 0.9, 0.6], 0.40)
        else:
            # Default general-purpose
            return ("General_Purpose", [0.5, 0.5, 0.5, 0.5, 0.5], 0.70)
