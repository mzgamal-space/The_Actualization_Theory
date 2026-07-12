"""
thought.py — Candidate Thought representation under the CKT Framework
======================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Defines CandidateThought, representing a causal proposal C_{A -> B}
connecting a current state to a target state.

  - Maintains the unactualized propensity nu(D_i) in [0, 1].
  - Holds a DIEPTState (Z = A + iB) tracking grounded vs speculative subspaces.
  - Tracks the five Prime compliance scores, calibrated by filters.
"""

from __future__ import annotations
import math
from typing import List, Optional

from diept import DIEPTState


class CandidateThought:
    """
    A candidate causal path through the problem space.

    Attributes
    ----------
    causal_chain   : list[int]    — Token sequence forming the candidate path.
    propensity     : float        — nu(D_i) in [0, 1]; collapses to 0 on violation.
    diept_state    : DIEPTState   — Complex justification state Z = A + iB.
    primes         : dict         — Five Prime compliance scores in [0, 1].
    """

    def __init__(
        self,
        causal_chain: List[int],
        propensity: float = 1.0,
        diept_state: Optional[DIEPTState] = None,
    ) -> None:
        self.causal_chain = causal_chain
        self.propensity   = propensity

        # Default: fully grounded, zero speculation (size-8 vector).
        self.diept_state = diept_state if diept_state is not None \
            else DIEPTState([1.0] * 8, [0.0] * 8)

        # Five Primes — calibrated by filters during verification.
        self.primes = {
            "Order"    : 1.0,
            "Justice"  : 1.0,
            "Mercy"    : 1.0,
            "Knowledge": 1.0,
            "Power"    : 1.0,
        }

    # ---- phase angle (Mercy compliance signal) ----

    @property
    def phase_angle(self) -> float:
        """
        theta = arctan(||B|| / ||A||)
        Delegates to DIEPTState.
        """
        return self.diept_state.theta

    # ---- Order Prime compliance ----

    def compute_order_compliance(self, repetition_penalty: float = 0.25) -> float:
        """
        Evaluates the Order Prime score based on structural repetition.
        A perfectly non-repeating chain scores 1.0.
        """
        if not self.causal_chain:
            return 1.0
        unique    = set(self.causal_chain)
        duplicates = len(self.causal_chain) - len(unique)
        score = math.exp(-repetition_penalty * duplicates)
        self.primes["Order"] = score
        return score

    def __repr__(self) -> str:
        return (
            f"CandidateThought(chain={self.causal_chain}, "
            f"propensity={self.propensity:.4f}, "
            f"theta={self.phase_angle:.4f} rad, "
            f"Primes={ {k: round(v, 3) for k, v in self.primes.items()} })"
        )
