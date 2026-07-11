"""
fdsa.py — Upgraded Fractal Deduction Search Algorithm (FDSA)
============================================================
Author : Antigravity (Advanced Agentic Coding)
Base   : Final_Output/02_Core_Engine/fdsa_pruner.py

Implements isomorphic anchoring and dimensional truncation.
Supports dynamic lookup and similarity matching against crystallized MCE objects.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple
from ckt_actualizer.models.mce import ReferenceDomain, MCE
from ckt_actualizer.utils.exceptions import AnalogyLibraryEmptyError

# Built-in reference domain library
DEFAULT_LIBRARY: List[ReferenceDomain] = [
    ReferenceDomain(
        "Resistor_Equilibrium",
        [0.40, 0.30, 0.10, 0.10, 0.10],
        k=0.35,
        description=(
            "Electrical resistor network reaching Kirchhoff voltage equilibrium. "
            "High Order (current routing obeys strict laws) and Justice (load "
            "balancing across nodes). Best analogy for constraint-satisfaction, "
            "code generation, and logical reasoning tasks."
        ),
    ),
    ReferenceDomain(
        "Fermat_Least_Time",
        [0.60, 0.10, 0.10, 0.10, 0.10],
        k=0.45,
        description=(
            "Fermat's principle of least time (optics). Extreme Order — the path "
            "taken is always the globally optimal path. Best analogy for shortest-"
            "path problems, mathematical proof steps, and planning tasks."
        ),
    ),
    ReferenceDomain(
        "Cellular_Homeostasis",
        [0.20, 0.20, 0.30, 0.20, 0.10],
        k=0.50,
        description=(
            "Biological cell maintaining homeostatic equilibrium across membrane "
            "potentials. High Mercy (graceful adaptation) and Knowledge (predictive "
            "biochemical signalling). Best analogy for creative, open-ended, or "
            "narrative generation tasks."
        ),
    ),
]


class FractalDeductionSearch:
    """
    FDSA engine with dynamic isomorphic anchoring against physical reference domains
    and crystallized MCE knowledge objects.
    """
    def __init__(self, analogy_library: Optional[List[ReferenceDomain]] = None) -> None:
        self.library = list(analogy_library) if analogy_library is not None else list(DEFAULT_LIBRARY)

    def add_reference_domain(self, domain: ReferenceDomain) -> None:
        """Dynamically appends a new ReferenceDomain or MCE object to the library."""
        self.library.append(domain)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Computes cosine similarity between two Prime profile vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def isomorphic_anchoring(self, P_unknown: List[float]) -> Tuple[ReferenceDomain, float]:
        """
        Finds the best-matching reference domain or MCE for the given Prime profile.
        """
        if not self.library:
            raise AnalogyLibraryEmptyError("Analogy library is empty.")
            
        best_domain, best_sim = self.library[0], -1.0
        for domain in self.library:
            sim = self._cosine_similarity(P_unknown, domain.profile)
            if sim > best_sim:
                best_sim, best_domain = sim, domain
        return best_domain, best_sim

    @staticmethod
    def fractal_dimension(N: int, k: float) -> float:
        """
        Computes the Actualization Fractal Dimension D = ln(N) / ln(1/k).
        """
        if not (0 < k < 1):
            k = 0.45
        return math.log(N) / math.log(1.0 / k)


class VectorizedFDSAPruner:
    """
    High-performance pre-inference vocabulary pruner using FDSA.
    """
    CONTEXT_PROFILES: Dict[str, List[float]] = {
        "logical_coding"   : [0.50, 0.30, 0.05, 0.10, 0.05],
        "creative_dialogue": [0.10, 0.10, 0.40, 0.10, 0.30],
        "mathematical"     : [0.55, 0.25, 0.05, 0.10, 0.05],
        "factual_qa"       : [0.35, 0.35, 0.10, 0.15, 0.05],
        "general"          : [0.20, 0.20, 0.20, 0.20, 0.20],
    }

    def __init__(self, vocab_size: int, fdsa_search: FractalDeductionSearch) -> None:
        self.V = vocab_size
        self.fdsa = fdsa_search

    def prune_vocabulary(
        self,
        logits: List[float],
        last_token: int,
        grammar_rules: Dict[int, Set[int]],
        context_type: str = "general",
    ) -> Tuple[List[float], int, ReferenceDomain, float]:
        """
        Applies dimensional truncation and grammar masking to logits.
        """
        profile = self.CONTEXT_PROFILES.get(context_type, self.CONTEXT_PROFILES["general"])
        domain, similarity = self.fdsa.isomorphic_anchoring(profile)
        k_ref = domain.k
        D_limit = self.fdsa.fractal_dimension(self.V, k_ref)
        threshold = -D_limit * 1.5

        # Grammar successor set
        allowed: Optional[Set[int]] = (
            grammar_rules.get(last_token) if last_token in grammar_rules else None
        )

        pruned = list(logits)
        active = 0
        for v in range(self.V):
            valid = True
            if logits[v] < threshold:
                valid = False
            if allowed is not None and v not in allowed:
                valid = False
            if not valid:
                pruned[v] = -math.inf
            else:
                active += 1

        return pruned, active, domain, similarity
