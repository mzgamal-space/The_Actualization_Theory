"""
filters.py — Epistemic Verification Filters for Candidate Thoughts
===================================================================
Author : Antigravity (Advanced Agentic Coding)

Implements:
  1. Pipeline A (Justice): Causation-Chain Justifier using CWF penalty mapping.
  2. Pipeline C (Negentropy): DIEPT phase-angle quarantine.
  3. Epistemic Selection Loop: 3-term Conciseness Cost Functional C(R).
"""

from __future__ import annotations
import math
from typing import Dict, Set, Tuple
from ckt_actualizer.models.thought import CandidateThought
from ckt_actualizer.utils.exceptions import InvalidPrimeProfileError
from ckt_actualizer.core.diept import CausalDensityFunction, InformationCurvatureLoss

class EpistemicVerificationSuite:
    """
    Coordinates Pipeline A, Pipeline C, and Epistemic Cost Functional filters.
    
    Parameters
    ----------
    cwf_penalty_matrix : dict[Tuple[int, int], float]
        Dictionary mapping token transitions (t_i, t_j) to causal penalty values.
        A penalty of float('inf') represents a direct causal violation.
    theta_target : float
        Mercy threshold for DIEPT phase angle (in radians). Default 0.70.
    lambda_R : float
        Weight for Redundancy cost (Order). Default 0.35.
    lambda_L : float
        Weight for Loss cost (Justice + Knowledge). Default 0.45.
    lambda_D : float
        Weight for Decision Cost (Mercy + Power). Default 0.20.
    """
    def __init__(
        self,
        cwf_penalty_matrix: Dict[Tuple[int, int], float],
        theta_target: float = 0.70,
        lambda_R: float = 0.35,
        lambda_L: float = 0.45,
        lambda_D: float = 0.20,
    ) -> None:
        # Enforce Justice Dominance Constraint
        if not (lambda_L > lambda_R and lambda_L > lambda_D):
            raise InvalidPrimeProfileError(
                f"Justice Dominance Constraint Violated! Must satisfy "
                f"lambda_L > lambda_R and lambda_L > lambda_D. "
                f"Got: R={lambda_R}, L={lambda_L}, D={lambda_D}"
            )
            
        self.cwf = cwf_penalty_matrix
        self.theta_target = theta_target
        self.lambda_R = lambda_R
        self.lambda_L = lambda_L
        self.lambda_D = lambda_D
        self.causal_density = CausalDensityFunction()

    # ------------------------------------------------------------------
    # Pipeline A — Justice: Causation-Chain Justifier
    # ------------------------------------------------------------------
    def run_pipeline_a(self, thought: CandidateThought) -> bool:
        """
        Evaluates the causal transitions in the thought's chain.
        Collapses propensity to 0.0 if any transition violates causal laws.
        Updates self.primes["Justice"] and self.primes["Power"].
        """
        chain = thought.causal_chain
        if len(chain) < 2:
            thought.primes["Justice"] = 1.0
            return True
            
        total_penalty = 0.0
        has_causal_violation = False
        
        for i in range(len(chain) - 1):
            transition = (chain[i], chain[i+1])
            penalty = self.cwf.get(transition, 0.0)
            
            # Apply depth scaling from CausalDensityFunction (partial Justice compliance)
            weight = self.causal_density.evaluate(i)
            penalty *= weight
            
            # Direct causal violation collapses the state
            if math.isinf(penalty) or penalty >= 100.0:
                has_causal_violation = True
                total_penalty = float('inf')
                break
            total_penalty += penalty
            
        if has_causal_violation:
            thought.propensity = 0.0
            thought.primes["Justice"] = 0.0
            thought.primes["Power"] = 0.0  # Paralysis, unable to act
            return False
            
        # Graded compliance based on accumulated penalty
        justice_score = math.exp(-0.2 * total_penalty)
        thought.primes["Justice"] = justice_score
        return True

    # ------------------------------------------------------------------
    # Pipeline C — Negentropy Filter (DIEPT Quarantine)
    # ------------------------------------------------------------------
    def run_pipeline_c(self, thought: CandidateThought) -> bool:
        """
        Verifies if speculative entropy exceeds the Mercy threshold.
        Quarantines thoughts (propensity to 0.0) if phase angle theta > theta_target.
        Updates self.primes["Mercy"] and self.primes["Knowledge"].
        """
        theta = thought.phase_angle
        
        # If theta exceeds the target boundary, the thought is ungrounded noise
        if theta > self.theta_target:
            thought.propensity = 0.0
            thought.primes["Mercy"] = 0.0
            thought.primes["Knowledge"] = 0.0  # Distorts parametric knowledge
            return False
            
        # Graded Mercy compliance: higher score when further below threshold
        mercy_score = 1.0 - (theta / self.theta_target)
        thought.primes["Mercy"] = max(0.05, mercy_score)
        
        # Knowledge score matches groundedness (more grounded -> higher accuracy score)
        thought.primes["Knowledge"] = math.cos(theta)
        return True

    # ------------------------------------------------------------------
    # Epistemic Selection Loop (3-term Cost Functional)
    # ------------------------------------------------------------------
    def evaluate_cost(self, thought: CandidateThought) -> float:
        """
        Computes the 3-term Conciseness Cost Functional:
           C(R) = lambda_R * R + lambda_L * L + lambda_D * D
        
        If the thought is collapsed/quarantined (propensity = 0.0), 
        the cost diverges to infinity.
        """
        if thought.propensity == 0.0:
            return float('inf')
            
        chain = thought.causal_chain
        I_in = len(chain)
        if I_in == 0:
            return float('inf')
            
        # 1. Redundancy and Curvature Loss (governed by Order Prime)
        thought.compute_order_compliance()
        
        # Enforce Information Curvature Loss for logical contradictions/cycles
        residues = InformationCurvatureLoss.detect_residues(chain)
        if residues > 0:
            thought.primes["Order"] *= math.exp(-0.5 * residues)
            
        duplicates = I_in - len(set(chain))
        R = 1.0 + (duplicates / I_in)
        
        # 2. Loss (governed by Justice + Knowledge Primes)
        # Loss increases as Prime compliance drops
        P_J = thought.primes["Justice"]
        P_K = thought.primes["Knowledge"]
        L = (1.0 - P_J) + (1.0 - P_K)
        
        # 3. Decision Cost (governed by Mercy + Power Primes)
        P_M = thought.primes["Mercy"]
        P_P = thought.primes["Power"]
        D = (1.0 - P_M) + (1.0 - P_P)
        
        # Combine under C(R)
        C_R = (self.lambda_R * R) + (self.lambda_L * L) + (self.lambda_D * D)
        return C_R
