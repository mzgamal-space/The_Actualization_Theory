"""
engine.py — Upgraded Actualizer Engine (CKT Core)
==================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Implements the full PBI Cognitive Life Cycle:

  Stage A0 — Question Operator Parsing (DIEPT Epistemic Architecture Triple)
  Stage 1  — Thought Generation via FDSA (Fractal Deduction Search)
  Stage 2  — Epistemic Verification (Pipeline A: Justice, Pipeline C: Negentropy)
  Stage 3  — Crystallization into MCE Class + Linguistic Modality Assignment

Mathematical bounds enforced
-----------------------------
  Theorem 7 (Unsimulability of Reality):
    CAKI = K_acc / I_max  in [0, 1)   (asymptotically approaches, never reaches 1.0)
    I_max = I_in + delta_finite        (delta_finite > 0 encodes the unsimulable boundary)

  Banach Contraction Principle:
    Each FDSA steering step applies a k-contractive mapping on U until
    the L2-norm delta <= Q_c (convergence).

  Justice Dominance Constraint:
    lambda_L > lambda_R  and  lambda_L > lambda_D  (enforced in EpistemicVerificationSuite)

  CAKI (Concise Accumulated Knowledge Index):
    K_acc = (I_eff / R) * exp(-L / (T + eps))
    CAKI  = K_acc / (I_in + delta_finite)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

from mce import MCE, ReferenceDomain
from thought import CandidateThought
from filters import EpistemicVerificationSuite
from fdsa import FractalDeductionSearch, VectorizedFDSAPruner
from diept import QuestionOperatorParser, DIEPTState


class UpgradedActualizerEngine:
    """
    Upgraded Actualizer Engine — full CKT Cognitive Lifecycle.

    Parameters
    ----------
    vocab_size         : int    — Vocabulary size V.
    cwf_penalty_matrix : dict   — Causal transitions penalty matrix for Pipeline A.
    k_contractive      : float  — Default Banach contraction constant. Default 0.45.
    Q_c                : float  — L2 convergence threshold. Default 1e-5.
    tau                : float  — Vacuum Brake decay temperature. Default 1.0.
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
        theta_target: float = 0.70,
        caki_threshold: float = 0.80,
        delta_finite: float = 0.5,
    ) -> None:
        self.V              = vocab_size
        self.k              = k_contractive
        self.Q_c            = Q_c
        self.tau            = tau
        self.caki_threshold = caki_threshold
        self.delta_finite   = delta_finite

        self.fdsa_search = FractalDeductionSearch()
        self.pruner      = VectorizedFDSAPruner(self.V, self.fdsa_search)
        self.verifier    = EpistemicVerificationSuite(
            cwf_penalty_matrix=cwf_penalty_matrix,
            theta_target=theta_target,
            lambda_R=0.35,
            lambda_L=0.45,
            lambda_D=0.20,
        )

    # ------------------------------------------------------------------
    # Core Steering Logic — Contractive Actualization Loop
    # ------------------------------------------------------------------

    def _softmax(self, logits: List[float]) -> List[float]:
        max_l  = max(x for x in logits if x != -math.inf)
        exp_l, valid_idx = [], []
        for i, x in enumerate(logits):
            if x != -math.inf:
                exp_l.append(math.exp(x - max_l))
                valid_idx.append(i)
        total  = sum(exp_l) or 1.0
        probs  = [0.0] * self.V
        for j, i in enumerate(valid_idx):
            probs[i] = exp_l[j] / total
        return probs

    def compute_drift_tensor(
        self,
        U: List[float],
        history: List[int],
        target_tokens: Set[int],
    ) -> List[float]:
        """
        Drift weights: Order (w_L), Justice (w_G), Knowledge (w_F).
        """
        w_L, w_G, w_F = 0.35, 0.35, 0.20
        D = [0.0] * self.V

        # Local Recency Drift (Order Prime)
        for step_back, tok in enumerate(reversed(history[-8:])):
            if 0 <= tok < self.V:
                D[tok] += w_L * 2.0 * math.exp(-0.4 * step_back)

        # Global Drift (Justice) + Future Drift (Knowledge)
        for v in range(self.V):
            if U[v] == 0.0:
                continue
            if v not in target_tokens:
                D[v] += w_G * 1.5
            D[v] += w_F * (-math.log(max(U[v], 1e-12)) * 0.08)

        return D

    def apply_vacuum_brake(
        self, U: List[float], D: List[float]
    ) -> List[float]:
        decay    = [math.exp(-d / self.tau) for d in D]
        U_braked = [U[i] * decay[i] for i in range(self.V)]
        total    = sum(U_braked) or 1.0
        return [x / total for x in U_braked]

    def steer_next_token(
        self,
        logits: List[float],
        history: List[int],
        target_tokens: Set[int],
        context_type: str = "general",
    ) -> Tuple[int, List[float], float, int, ReferenceDomain, float]:
        """
        Pre-inference FDSA pruning -> contractive actualization loop.
        Returns (selected_token, U, final_drift, iterations, anchor_domain, similarity).
        """
        # Stage 1-a: FDSA Vocabulary Pruning
        pruned_logits, _, anchor_domain, similarity = self.pruner.prune_vocabulary(
            logits, history[-1] if history else -1, {}, context_type
        )

        # Stage 1-b: Contractive Banach mapping
        U       = self._softmax(pruned_logits)
        k_step  = anchor_domain.k

        for iteration in range(1, 21):
            U_prev = U[:]
            D      = self.compute_drift_tensor(U, history, target_tokens)
            U_b    = self.apply_vacuum_brake(U, D)
            U      = [k_step * U_b[v] + (1.0 - k_step) * U_prev[v]
                      for v in range(self.V)]
            delta  = math.sqrt(sum((U[v] - U_prev[v]) ** 2 for v in range(self.V)))
            if delta <= self.Q_c:
                break

        selected_token = max(range(self.V), key=lambda v: U[v])
        final_drift    = self.compute_drift_tensor(U, history, target_tokens)[selected_token]
        return selected_token, U, final_drift, iteration, anchor_domain, similarity

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
        Pipeline A -> Pipeline C -> CAKI -> Crystallization gate.

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
            # MCE metrics
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
        Stage 1  — Generate causal chain via FDSA contractive loop.
        Stage 2  — Verify via Pipelines A & C; compute C(R) and CAKI.
        Stage 3  — Crystallize into MCE; assign Linguistic Modality.

        Returns (linguistic_marker, is_crystallized, mce_obj | None, caki).
        """
        # Stage A0
        domain, prime_vector, theta_target = QuestionOperatorParser.parse_operator(
            query.split()[0]
        )
        self.verifier.theta_target = theta_target

        # Stage 1 — FDSA token steering (2 steps)
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
