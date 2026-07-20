"""
filters.py — Epistemic Verification Filters (Pipelines A and C)
===============================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Implements the three-stage C(R) verification architecture:

  Pipeline A (Justice) — Causation-Chain Justifier
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Checks each transition in the causal chain against the Causation Wave
  Function (CWF) penalty matrix.  Infinite penalties collapse propensity
  to 0.0 (hard causal violation).  Graded penalties reduce Justice score.
  Depth-weighted via CausalDensityFunction C(x).

  Pipeline C (Negentropy / Mercy) — DIEPT Quarantine
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Evaluates the DIEPT phase angle theta.  If theta > theta_target, the
  thought is flagged as ungrounded noise and quarantined (propensity 0.0).
  Graded compliance proportional to how far theta is below the target.

  Epistemic Selection Loop — 3-term Conciseness Cost Functional C(R)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  C(R) = lambda_R * R  +  lambda_L * L  +  lambda_D * D
  Justice Dominance Constraint:  lambda_L > lambda_R  and  lambda_L > lambda_D.
  Information Curvature Loss penalizes Order Prime for detected cycles.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple

from thought import CandidateThought
from exceptions import InvalidPrimeProfileError
from diept import CausalDensityFunction, InformationCurvatureLoss


class EpistemicVerificationSuite:
    """
    Coordinates Pipeline A, Pipeline C, and the Epistemic Cost Functional.

    Parameters
    ----------
    cwf_penalty_matrix : dict[(int, int), float]
        Token-transition penalty map.  float('inf') = hard causal violation.
    theta_target : float
        Mercy threshold for DIEPT phase angle (radians).  Default 0.70.
    lambda_R : float
        Weight for Redundancy cost (Order Prime).  Default 0.35.
    lambda_L : float
        Weight for Loss cost (Justice + Knowledge).  Default 0.45.
    lambda_D : float
        Weight for Decision Cost (Mercy + Power).    Default 0.20.
    """

    def __init__(
        self,
        cwf_penalty_matrix: Dict[Tuple[int, int], float],
        theta_target: float = 0.70,
        lambda_R: float = 0.35,
        lambda_L: float = 0.45,
        lambda_D: float = 0.20,
    ) -> None:
        # Justice Dominance Constraint
        if not (lambda_L > lambda_R and lambda_L > lambda_D):
            raise InvalidPrimeProfileError(
                f"Justice Dominance Constraint violated! "
                f"Require lambda_L > lambda_R and lambda_L > lambda_D. "
                f"Got: R={lambda_R}, L={lambda_L}, D={lambda_D}"
            )
        self.cwf          = cwf_penalty_matrix
        self.theta_target = theta_target
        self.lambda_R     = lambda_R
        self.lambda_L     = lambda_L
        self.lambda_D     = lambda_D
        self.causal_density = CausalDensityFunction()

    # ------------------------------------------------------------------
    # Pipeline A — Justice: Causation-Chain Justifier
    # ------------------------------------------------------------------

    def run_pipeline_a(self, thought: CandidateThought) -> bool:
        """
        Evaluates causal transitions via CWF penalty matrix.
        Collapses propensity to 0.0 on causal violation.
        Updates primes["Justice"] and primes["Power"].
        Returns True if the thought survives (no hard violation).
        """
        chain = thought.causal_chain
        if len(chain) < 2:
            thought.primes["Justice"] = 1.0
            return True

        total_penalty     = 0.0
        has_violation     = False

        for i in range(len(chain) - 1):
            transition = (chain[i], chain[i + 1])
            penalty    = self.cwf.get(transition, 0.0)

            # Depth-weighted penalty via CausalDensityFunction C(x)
            weight  = self.causal_density.evaluate(i)
            penalty *= weight

            if math.isinf(penalty) or penalty >= 100.0:
                has_violation = True
                total_penalty = float("inf")
                break
            total_penalty += penalty

        if has_violation:
            thought.propensity       = 0.0
            thought.primes["Justice"] = 0.0
            thought.primes["Power"]   = 0.0  # Paralysis: unable to act
            return False

        # Graded compliance based on accumulated penalty
        thought.primes["Justice"] = math.exp(-0.2 * total_penalty)
        return True

    # ------------------------------------------------------------------
    # Pipeline C — Negentropy Filter (DIEPT Quarantine)
    # ------------------------------------------------------------------

    def run_pipeline_c(self, thought: CandidateThought) -> bool:
        """
        Evaluates DIEPT phase angle against Mercy threshold.
        Quarantines (propensity = 0.0) if theta > theta_target.
        Updates primes["Mercy"] and primes["Knowledge"].
        Returns True if the thought survives.
        """
        theta = thought.phase_angle

        if theta > self.theta_target:
            thought.propensity         = 0.0
            thought.primes["Mercy"]    = 0.0
            thought.primes["Knowledge"] = 0.0
            return False

        # Graded Mercy compliance
        mercy_score              = 1.0 - (theta / self.theta_target)
        thought.primes["Mercy"]  = max(0.05, mercy_score)
        # Knowledge score: more grounded -> higher accuracy
        thought.primes["Knowledge"] = math.cos(theta)
        return True

    # ------------------------------------------------------------------
    # Epistemic Selection Loop — 3-term Conciseness Cost Functional
    # ------------------------------------------------------------------

    def evaluate_cost(self, thought: CandidateThought) -> float:
        """
        C(R) = lambda_R * R  +  lambda_L * L  +  lambda_D * D

        Returns float('inf') if the thought has been collapsed / quarantined.
        """
        if thought.propensity == 0.0:
            return float("inf")

        chain = thought.causal_chain
        I_in  = len(chain)
        if I_in == 0:
            return float("inf")

        # 1. Order Prime — Redundancy + Information Curvature Loss
        thought.compute_order_compliance()
        residues = InformationCurvatureLoss.detect_residues(chain)
        if residues > 0:
            thought.primes["Order"] *= math.exp(-0.5 * residues)

        duplicates = I_in - len(set(chain))
        R = 1.0 + (duplicates / I_in)

        # 2. Justice + Knowledge — Loss term
        P_J = thought.primes["Justice"]
        P_K = thought.primes["Knowledge"]
        L   = (1.0 - P_J) + (1.0 - P_K)

        # 3. Mercy + Power — Decision Cost
        P_M = thought.primes["Mercy"]
        P_P = thought.primes["Power"]
        D   = (1.0 - P_M) + (1.0 - P_P)

        return (self.lambda_R * R) + (self.lambda_L * L) + (self.lambda_D * D)
