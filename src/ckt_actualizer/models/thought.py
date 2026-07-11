"""
thought.py — Candidate Thought representation under the CKT Framework
======================================================================
Author : Antigravity (Advanced Agentic Coding)

Defines CandidateThought, representing a causal proposal C_{A -> B} 
connecting a starting state to a target state. It maintains the 
unactualized propensity, the DIEPT hidden subspaces (grounded and speculative), 
and the five Prime compliance scores.
"""

from __future__ import annotations
import math
from typing import List, Optional
from ckt_actualizer.core.diept import DIEPTState

class CandidateThought:
    """
    CandidateThought represents a candidate causal path through the problem space.
    
    Attributes
    ----------
    causal_chain : List[int]
        The sequence of tokens/states forming this candidate path.
    propensity : float
        Unactualized propensity nu(D_i) in [0, 1]. Represented initially as 
        its raw transition probability.
    diept_state : DIEPTState
        The complex justification state Z = A + iB managing grounding vs speculation.
    primes : dict[str, float]
        Dictionary mapping each of the 5 Conceptual Primes to a compliance score in [0, 1].
    """
    def __init__(
        self,
        causal_chain: List[int],
        propensity: float = 1.0,
        diept_state: Optional[DIEPTState] = None,
    ) -> None:
        self.causal_chain = causal_chain
        self.propensity = propensity
        
        # Initialize DIEPT hidden state subspaces.
        # If not provided, we set default vectors of size 8 (fully grounded).
        self.diept_state = diept_state if diept_state is not None else DIEPTState([1.0]*8, [0.0]*8)
        
        # Initialize Prime profiles with default values.
        # These are calibrated by filters during the verification phase.
        self.primes = {
            "Order"    : 1.0,
            "Justice"  : 1.0,
            "Mercy"    : 1.0,
            "Knowledge": 1.0,
            "Power"    : 1.0,
        }

    @property
    def phase_angle(self) -> float:
        """
        Computes the DIEPT phase angle: theta = arctan(||B|| / ||A||).
        Delegates to DIEPTState.
        """
        return self.diept_state.theta

    def compute_order_compliance(self, repetition_penalty: float = 0.25) -> float:
        """
        Evaluates the Order Prime score based on structural repetition.
        A perfectly non-repeating chain scores 1.0.
        """
        if not self.causal_chain:
            return 1.0
        
        # Count duplicates in the chain
        unique_tokens = set(self.causal_chain)
        duplicates = len(self.causal_chain) - len(unique_tokens)
        
        # Apply exponential decay to compliance for duplicates
        score = math.exp(-repetition_penalty * duplicates)
        self.primes["Order"] = score
        return score

    def __repr__(self) -> str:
        return (
            f"CandidateThought(chain={self.causal_chain}, propensity={self.propensity:.4f}, "
            f"theta={self.phase_angle:.4f}, Primes={ {k: round(v, 2) for k, v in self.primes.items()} })"
        )
