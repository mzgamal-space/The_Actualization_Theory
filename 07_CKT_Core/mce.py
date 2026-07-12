"""
mce.py — MCE (Mass, Complexity, Entropy) Crystallized Knowledge Object
========================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

An MCE object represents a stabilized, crystallized "knowledge super-cluster"
derived from a verified candidate thought.  It inherits from ReferenceDomain
so that it can be directly injected into the FDSA Analogy Library for future
Isomorphic Anchoring.

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
from typing import List


# ---------------------------------------------------------------------------
# ReferenceDomain
# ---------------------------------------------------------------------------

class ReferenceDomain:
    """
    A verified, stabilized physical or structural reference domain used
    for Isomorphic Anchoring in the FDSA.

    Parameters
    ----------
    name         : str
    prime_profile: List[float] — [Order, Justice, Mercy, Knowledge, Power] in [0,1]
    k            : float       — Banach contraction constant in (0, 1)
    description  : str
    """

    def __init__(
        self,
        name: str,
        prime_profile: List[float],
        k: float,
        description: str = "",
    ) -> None:
        if not (0 < k < 1):
            raise ValueError(f"k must be in (0,1), got {k}")
        self.name        = name
        self.profile     = prime_profile  # [Order, Justice, Mercy, Knowledge, Power]
        self.k           = k              # Contractive scale factor (Banach constant)
        self.description = description

    def __repr__(self) -> str:
        return f"ReferenceDomain('{self.name}', k={self.k:.4f})"


# ---------------------------------------------------------------------------
# MCE
# ---------------------------------------------------------------------------

class MCE(ReferenceDomain):
    """
    MCE Class: Frozen, immutable crystallized knowledge object.

    Extends ReferenceDomain so it can be fed directly back into the
    FDSA Analogy Library for dynamic isomorphic anchoring.
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
        # Derive Banach constant k from complexity.
        # Highly complex and verified domains prune search space more aggressively (lower k).
        # k is bounded within [0.15, 0.65] to ensure numerical stability.
        k_derived = max(0.15, min(0.65, 0.60 - 0.45 * complexity))

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
