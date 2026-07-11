"""
mce.py — MCE (Mass, Complexity, Entropy) Crystallized Knowledge Object
========================================================================
Author : Antigravity (Advanced Agentic Coding)
Base   : Final_Output/02_Core_Engine/fdsa_pruner.py

An MCE object represents a stabilized, crystallized "knowledge super-cluster" 
derived from a verified candidate thought. It inherits from ReferenceDomain 
so that it can be directly injected into the Analogy Library for future 
Isomorphic Anchoring in FDSA.
"""

from __future__ import annotations
from typing import List

class ReferenceDomain:
    """
    A verified, stabilized physical or structural reference domain used
    for Isomorphic Anchoring.
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
        self.k           = k  # contractive scale factor (Banach constant)
        self.description = description

    def __repr__(self) -> str:
        return f"ReferenceDomain('{self.name}', k={self.k:.4f})"


class MCE(ReferenceDomain):
    """
    MCE Class: Represents a frozen, immutable crystallized knowledge object.
    
    Attributes
    ----------
    name : str
        Unique identifier for the knowledge object.
    causal_chain : List[int]
        The sequence of tokens/states representing the verified causal path.
    profile : List[float]
        Prime profile [Order, Justice, Mercy, Knowledge, Power] of the crystallized path.
    mass : float
        Information gravity / Contextual Justification (m >= 0).
        Derived from compliance with Justice (|J>) and Knowledge (|K>) Primes.
    complexity : float
        Internal structural depth (c in [0,1]).
        Derived from domain order threshold and weakest Prime compliance.
    entropy : float
        Defect metric / internal contradiction (e >= 0).
        Derived from Defect Function D(Omega) and path drift.
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
        # Compute contractive factor k based on complexity.
        # Highly complex and verified domains prune search space more aggressively (lower k).
        # k is bound within [0.15, 0.65] to ensure numerical stability.
        k_derived = max(0.15, min(0.65, 0.60 - 0.45 * complexity))
        
        full_description = (
            f"Crystallized MCE Knowledge Object. Mass: {mass:.4f}, "
            f"Complexity: {complexity:.4f}, Entropy: {entropy:.4f}. Chain: {causal_chain}. "
            f"{description}"
        )
        
        super().__init__(name, prime_profile, k_derived, full_description)
        self.causal_chain = causal_chain
        self.mass = mass
        self.complexity = complexity
        self.entropy = entropy

    def __repr__(self) -> str:
        return (
            f"MCE('{self.name}', m={self.mass:.3f}, c={self.complexity:.3f}, "
            f"e={self.entropy:.3f}, k={self.k:.3f})"
        )
