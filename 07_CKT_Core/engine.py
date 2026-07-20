"""
engine.py — Upgraded Actualizer Engine (CKT Core)
==================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Extends the canonical ActualizerEngine (02_Core_Engine/actualizer_engine.py)
with the full CKT Cognitive Life Cycle:

  Stage A0 — Question Operator Parsing (DIEPT Epistemic Architecture Triple)
  Stage 1  — Thought Generation via FDSA (pre-inference pruning + Banach steer)
  Stage 2  — Epistemic Verification (Pipeline A: Justice, Pipeline C: Negentropy)
  Stage 3  — Crystallization into MCE Class + Linguistic Modality Assignment

Architecture
------------
  UpgradedActualizerEngine wraps ActualizerEngine from 02_Core_Engine:
    - steer_next_token() delegates the Banach contraction loop to
      ActualizerEngine.steer() — no duplication of drift/vacuum/contraction logic.
    - Pre-inference FDSA pruning is delegated to VectorizedFDSAPruner (fdsa.py).
    - Crystallization logic (CAKI, MCE construction) is CKT-exclusive.

Mathematical bounds enforced
-----------------------------
  Theorem 7 (Unsimulability of Reality):
    CAKI = K_acc / I_max  in [0, 1)   (asymptotically approaches, never reaches 1.0)
    I_max = I_in + delta_finite        (delta_finite > 0 encodes the unsimulable boundary)

  Banach Contraction Principle:
    Delegated entirely to ActualizerEngine.steer().

  Justice Dominance Constraint:
    lambda_L > lambda_R  and  lambda_L > lambda_D  (enforced in EpistemicVerificationSuite)

  CAKI (Concise Accumulated Knowledge Index):
    K_acc = (I_eff / R) * exp(-L / (T + eps))
    CAKI  = K_acc / (I_in + delta_finite)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Core_Engine"))

import math
from typing import Dict, List, Optional, Set, Tuple

# Canonical base engine — Banach contraction loop lives here.
from actualizer_engine import ActualizerEngine

from mce import MCE
from fdsa import CKTFractalDeductionSearch, VectorizedFDSAPruner
from thought import CandidateThought
from filters import EpistemicVerificationSuite
from diept import QuestionOperatorParser, DIEPTState


class UpgradedActualizerEngine:
    """
    CKT Cognitive Life Cycle — wraps ActualizerEngine + adds MCE crystallization.

    Parameters
    ----------
    vocab_size         : int    — Vocabulary size V.
    cwf_penalty_matrix : dict   — Causal transitions penalty matrix for Pipeline A.
    k_contractive      : float  — Default Banach contraction constant. Default 0.45.
    Q_c                : float  — L2 convergence threshold. Default 1e-5.
    tau                : float  — Vacuum Brake decay temperature. Default 1.0.
    tau_bifurcation   : float  — Bifurcation threshold for Tr(D_μν) ≤ τ criterion. Default 5.0.
    theta_target       : float  — DIEPT Mercy threshold (radians). Default 0.70.
    caki_threshold     : float  — CAKI gate for crystallization. Default 0.80.
    delta_finite       : float  — Theorem 7 boundary gap. Default 0.5.
    """

    def __init__(
        self,
        vocab_size: int,
        cwf_penalty_matrix: Dict[Tuple[int, int], float],
        k_contractive: float = 0.45,
        Q_c: float = 1e-5,
        tau: float = 1.0,
        tau_bifurcation: float = 5.0,
        theta_target: float = 0.70,
        caki_threshold: float = 0.80,
        delta_finite: float = 0.5,
    ) -> None:
        self.V              = vocab_size
        self.caki_threshold = caki_threshold
        self.delta_finite   = delta_finite

        # Canonical Banach contraction engine (02_Core_Engine)
        self._base_engine = ActualizerEngine(
            vocab_size=vocab_size,
            mercy_k=k_contractive,
            Q_c=Q_c,
            tau=tau,
            tau_bifurcation=tau_bifurcation,
        )

        # CKT-exclusive components
        self.fdsa_search = CKTFractalDeductionSearch()
        self.pruner      = VectorizedFDSAPruner(vocab_size, self.fdsa_search)
        self.verifier    = EpistemicVerificationSuite(
            cwf_penalty_matrix=cwf_penalty_matrix,
            theta_target=theta_target,
            lambda_R=0.35,
            lambda_L=0.45,
            lambda_D=0.20,
        )

    # ------------------------------------------------------------------
    # Core Steering Logic — delegates to ActualizerEngine
    # ------------------------------------------------------------------

    def steer_next_token(
        self,
        logits: List[float],
        history: List[int],
        target_tokens: Set[int],
        context_type: str = "general",
    ) -> Tuple[int, List[float], float, int, object, float]:
        """
        Pre-inference FDSA pruning → Banach contractive actualization (delegated).

        Returns (selected_token, U_final, final_drift, iterations, anchor_domain, similarity).
        """
        # Phase 1: FDSA Vocabulary Pruning (CKT-exclusive, 4-tuple return)
        pruned_logits, _, anchor_domain, similarity = self.pruner.prune_vocabulary(
            logits, history[-1] if history else -1, {}, context_type
        )

        # Phase 2+3: Full Banach contraction loop — delegated to canonical engine.
        # The base engine uses the anchor domain's k via its mercy_k property.
        saved_mercy_k = self._base_engine.mercy_k
        self._base_engine.mercy_k = anchor_domain.k
        self._base_engine.k = anchor_domain.k
        
        selected_token, U_final, final_drift, iterations, nu_history, actualized = self._base_engine.steer(
            pruned_logits, history, target_tokens
        )
        
        self._base_engine.mercy_k = saved_mercy_k
        self._base_engine.k = saved_mercy_k

        return selected_token, U_final, final_drift, iterations, anchor_domain, similarity

    # ------------------------------------------------------------------
    # CAKI — Concise Accumulated Knowledge Index
    # ------------------------------------------------------------------

    def calculate_caki(
        self, thought: CandidateThought
    ) -> Tuple[float, float, float, float]:
        """
        K_acc = (I_eff / R) * exp(-L / (T + eps))
        CAKI  = K_acc / (I_in + delta_finite)      [Theorem 7 bound: never reaches 1.0]

        Returns (caki, L_violations, R, I_eff).
        """
        chain = thought.causal_chain
        I_in  = len(chain)
        if I_in == 0:
            return 0.0, 0.0, 1.0, 0.0

        # CWF accumulated penalty (Loss term)
        L_violations = sum(
            self.verifier.cwf.get((chain[i], chain[i + 1]), 0.0)
            for i in range(I_in - 1)
        )

        duplicates = I_in - len(set(chain))
        R          = 1.0 + (duplicates / I_in)
        I_eff      = max(0.0, float(I_in) - L_violations)

        T   = 0.5
        eps = 1e-6
        K_acc = (I_eff / R) * math.exp(-L_violations / (T + eps))

        # Theorem 7: I_max = I_in + delta_finite  ->  CAKI in [0, 1)
        I_max = float(I_in) + self.delta_finite
        caki  = K_acc / I_max

        return caki, L_violations, R, I_eff

    # ------------------------------------------------------------------
    # Verification and Crystallization
    # ------------------------------------------------------------------

    def verify_and_crystallize(
        self,
        thought: CandidateThought,
        delta_C_R: float,
        domain_name: str,
        description: str = "",
    ) -> Tuple[bool, Optional[MCE], float]:
        """
        Pipeline A → Pipeline C → CAKI → Crystallization gate.

        Crystallization conditions (Knowledge Accumulation Law):
          1. CAKI >= caki_threshold   (stable, concise thought)
          2. delta_C_R <= 0           (system entropy reduced)

        Returns (is_crystallized, mce_obj | None, caki).
        """
        # Pipeline A: Justice — Causal chain check
        if not self.verifier.run_pipeline_a(thought):
            return False, None, 0.0

        # Pipeline C: Mercy — Negentropy (DIEPT) quarantine
        if not self.verifier.run_pipeline_c(thought):
            return False, None, 0.0

        caki, L_viol, R, I_eff = self.calculate_caki(thought)

        if caki >= self.caki_threshold and delta_C_R <= 0.0:
            # MCE structural metrics (from class MCE.docx definitions)
            mass       = (thought.primes["Justice"]
                          * thought.primes["Knowledge"]
                          * len(thought.causal_chain))
            complexity = min(thought.primes.values())
            entropy    = (R - 1.0) + (
                L_viol / max(1, len(thought.causal_chain))
            )

            mce_obj = MCE(
                name=f"MCE_{domain_name}_{len(self.fdsa_search.library) + 1}",
                causal_chain=thought.causal_chain,
                prime_profile=[
                    thought.primes["Order"],
                    thought.primes["Justice"],
                    thought.primes["Mercy"],
                    thought.primes["Knowledge"],
                    thought.primes["Power"],
                ],
                mass=mass,
                complexity=complexity,
                entropy=entropy,
                description=description,
            )

            # Inject crystallized MCE into the FDSA library for future anchoring.
            self.fdsa_search.add_reference_domain(mce_obj)
            return True, mce_obj, caki

        return False, None, caki

    # ------------------------------------------------------------------
    # Full PBI Cognitive Life Cycle — process_query
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        initial_history: List[int],
        target_tokens: Set[int],
        simulated_diept_a: List[float],
        simulated_diept_b: List[float],
        delta_c_r_sim: float = -0.5,
    ) -> Tuple[str, bool, Optional[MCE], float]:
        """
        End-to-end PBI Cognitive Life Cycle:

        Stage A0 — Parse Question Operator -> Epistemic Architecture Triple.
        Stage 1  — Generate causal chain via FDSA + ActualizerEngine.steer().
        Stage 2  — Verify via Pipelines A & C; compute C(R) and CAKI.
        Stage 3  — Crystallize into MCE; assign Linguistic Modality.

        Returns (linguistic_marker, is_crystallized, mce_obj | None, caki).
        """
        # Stage A0
        domain, prime_vector, theta_target = QuestionOperatorParser.parse_operator(
            query.split()[0]
        )
        self.verifier.theta_target = theta_target

        # Stage 1 — FDSA pruning + Banach steer (2 steps)
        chain            = list(initial_history)
        total_propensity = 1.0
        for _ in range(2):
            logits = [1.0] * self.V
            token, U, _, _, _, _ = self.steer_next_token(
                logits, chain, target_tokens, context_type=domain
            )
            chain.append(token)
            total_propensity *= U[token]

        # Build CandidateThought with DIEPT state
        diept_state = DIEPTState(simulated_diept_a, simulated_diept_b)
        thought     = CandidateThought(
            causal_chain=chain,
            propensity=total_propensity,
            diept_state=diept_state,
        )

        # Stage 2 + 3
        is_cryst, mce_obj, caki = self.verify_and_crystallize(
            thought=thought,
            delta_C_R=delta_c_r_sim,
            domain_name=domain,
            description=f"Generated from query: '{query}'",
        )

        # Linguistic Modality from DIEPT
        marker = thought.diept_state.get_linguistic_marker(theta_target)
        return marker, is_cryst, mce_obj, caki
