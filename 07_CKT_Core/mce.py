"""
mce.py — MCE (Mass, Complexity, Entropy) Crystallized Knowledge Object
========================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

An MCE object represents a stabilized, crystallized "knowledge super-cluster"
derived from a verified candidate thought.  It inherits from ReferenceDomain
(defined in 02_Core_Engine/fdsa_pruner.py) so that it can be directly injected
into the FDSA Analogy Library for future Isomorphic Anchoring.

MCE Structural Metrics
-----------------------
  Mass (m)       : Information gravity / contextual justification.
                   Quantified by compliance with Justice (|J>) and Knowledge (|K>).
                   High-mass objects serve as unbreakable causal law anchors.

  Complexity (c) : Internal structural depth achieved by the thought.
                   By Theorem 1 (Reality-Complexity Equivalence):
                   Omega(S) = f(E,I) * min_i P_i(S).
                   Bounded by the weakest Prime compliance score.

  Entropy (e)    : Defect metric — internal contradiction, noise, cross-domain defects.
                   Corresponds to D(Omega) and the Total Path Drift Action H_drift.
                   Crystallization gate: e must stay below entropy ceiling.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Core_Engine"))

from typing import List

# ReferenceDomain is the single canonical definition in 02_Core_Engine.
from fdsa_pruner import ReferenceDomain


# ---------------------------------------------------------------------------
# MCE
# ---------------------------------------------------------------------------

class MCE(ReferenceDomain):
    """
    MCE Class: Frozen, immutable crystallized knowledge object.

    Extends ReferenceDomain so it can be fed directly back into the
    FDSA Analogy Library for dynamic isomorphic anchoring.

    Parameters
    ----------
    name          : str
    causal_chain  : List[int]   — The verified token sequence.
    prime_profile : List[float] — [Order, Justice, Mercy, Knowledge, Power] in [0, 1]
    mass          : float       — Information gravity (contextual justification).
    complexity    : float       — Structural depth (weakest-Prime bounded Omega(S)).
    entropy       : float       — Defect metric (D(Omega) + H_drift).
    description   : str
    """

    def __init__(
        self,
        name: str,
        causal_chain: List[int],
        prime_profile: List[float],
        mass: float,
        complexity: float,
        entropy: float,
        description: str = "",
    ) -> None:
        # Derive Banach constant k from the normalised complexity (min Prime score).
        # Highly complex, well-verified domains prune more aggressively (lower k).
        # Bounded to [0.15, 0.65] for numerical stability.
        min_prime = min(prime_profile) if prime_profile else 0.5
        k_derived = max(0.15, min(0.65, 0.60 - 0.45 * min_prime))

        full_description = (
            f"Crystallized MCE Knowledge Object. "
            f"Mass: {mass:.4f}, Complexity: {complexity:.4f}, Entropy: {entropy:.4f}. "
            f"Chain: {causal_chain}. {description}"
        )

        super().__init__(name, prime_profile, k_derived, full_description)
        self.causal_chain = causal_chain
        self.mass         = mass
        self.complexity   = complexity
        self.entropy      = entropy

    def __repr__(self) -> str:
        return (
            f"MCE('{self.name}', m={self.mass:.3f}, c={self.complexity:.3f}, "
            f"e={self.entropy:.3f}, k={self.k:.3f})"
        )
