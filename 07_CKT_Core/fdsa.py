"""
fdsa.py — Fractal Deduction Search Algorithm (FDSA)
====================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Wraps the canonical FDSA implementation in 02_Core_Engine/fdsa_pruner.py and
extends VectorizedFDSAPruner with:
  - Dynamic MCE-aware library (crystallized knowledge injected at runtime).
  - Extended prune_vocabulary() returning 4-tuple
    (pruned_logits, active_count, anchor_domain, similarity) for the CKT
    engine's steer_next_token() to consume.
  - Additional DIEPT-derived context profiles for Stage A0 domain lenses.

Mathematical foundations
------------------------
  Fractal Dimension   D = ln(N) / ln(1/k)
  Contraction factor  k  in (0, 1)   (from matched ReferenceDomain / MCE)
  Cosine similarity   sim(P_unknown, P_domain)  for isomorphic anchoring
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Core_Engine"))

import math
from typing import Dict, List, Optional, Set, Tuple

# Import canonical base classes from 02_Core_Engine (single source of truth).
from fdsa_pruner import ReferenceDomain, FractalDeductionSearch
from exceptions import AnalogyLibraryEmptyError

from mce import MCE


# ---------------------------------------------------------------------------
# CKTFractalDeductionSearch — extends base with MCE dynamic library support
# ---------------------------------------------------------------------------

class CKTFractalDeductionSearch(FractalDeductionSearch):
    """
    Extends the base FractalDeductionSearch with MCE-aware dynamic library.

    Crystallized MCE objects can be added at runtime via add_reference_domain(),
    making them immediately available as isomorphic anchors for future FDSA
    searches (dynamic knowledge accumulation).
    """

    def add_reference_domain(self, domain: ReferenceDomain) -> None:
        """Dynamically appends a new ReferenceDomain or MCE to the library."""
        self.library.append(domain)


# Re-export for backward compatibility with test imports.
FractalDeductionSearch = CKTFractalDeductionSearch


# ---------------------------------------------------------------------------
# CKTVectorizedFDSAPruner — extends base with 4-tuple return and DIEPT profiles
# ---------------------------------------------------------------------------

class VectorizedFDSAPruner:
    """
    CKT-extended pre-inference vocabulary pruner.

    Wraps the base VectorizedFDSAPruner logic but uses a CKTFractalDeductionSearch
    instance (so the MCE library is shared with the engine) and returns a 4-tuple:
        (pruned_logits, active_count, anchor_domain, similarity)

    Additional context profiles for DIEPT Stage A0 domain lenses are included.
    """

    # Unified context profiles: base + DIEPT Stage A0 domain lenses.
    CONTEXT_PROFILES: Dict[str, List[float]] = {
        "logical_coding"      : [0.50, 0.30, 0.05, 0.10, 0.05],
        "creative_dialogue"   : [0.10, 0.10, 0.40, 0.10, 0.30],
        "mathematical"        : [0.55, 0.25, 0.05, 0.10, 0.05],
        "factual_qa"          : [0.35, 0.35, 0.10, 0.15, 0.05],
        "general"             : [0.20, 0.20, 0.20, 0.20, 0.20],
        # DIEPT Stage A0 domain lenses
        "Causal_Law"          : [0.40, 0.50, 0.02, 0.05, 0.03],
        "OpenCI_Speculation"  : [0.10, 0.05, 0.50, 0.25, 0.10],
        "Wisdom_Evaluation"   : [0.20, 0.20, 0.20, 0.20, 0.20],
        "Procedural_Knowledge": [0.35, 0.30, 0.15, 0.15, 0.05],
        "General_Purpose"     : [0.20, 0.20, 0.20, 0.20, 0.20],
    }

    def __init__(self, vocab_size: int, fdsa_search: CKTFractalDeductionSearch) -> None:
        self.V    = vocab_size
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

        Returns (pruned_logits, active_count, anchor_domain, similarity).
        """
        profile = self.CONTEXT_PROFILES.get(
            context_type, self.CONTEXT_PROFILES["general"]
        )
        domain, similarity = self.fdsa.isomorphic_anchoring(profile)
        k_ref     = domain.k
        D_limit   = self.fdsa.fractal_dimension(self.V, k_ref)
        threshold = -D_limit * 1.5

        allowed: Optional[Set[int]] = (
            grammar_rules.get(last_token)
            if last_token in grammar_rules
            else None
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
