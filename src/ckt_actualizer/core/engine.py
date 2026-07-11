"""
engine.py — Upgraded Actualizer Engine with Thought Verification & Crystallization
==================================================================================
Author : Antigravity (Advanced Agentic Coding)

Coordinates the generation of candidate thoughts using the Actualizer steering loop,
applies the verification filters, calculates the CAKI index, and crystallizes 
highly concise thoughts into MCE objects.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple
from ckt_actualizer.models.mce import MCE, ReferenceDomain
from ckt_actualizer.models.thought import CandidateThought
from ckt_actualizer.core.filters import EpistemicVerificationSuite
from ckt_actualizer.core.fdsa import FractalDeductionSearch, VectorizedFDSAPruner
from ckt_actualizer.core.diept import QuestionOperatorParser, DIEPTState

class UpgradedActualizerEngine:
    """
    Upgraded Actualizer Engine that implements the full cognitive lifecycle:
    Generation -> Verification (Pipeline A/C) -> Crystallization (MCE Class).
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size V.
    cwf_penalty_matrix : dict
        Causal transitions penalty matrix for Pipeline A.
    k_contractive : float
        Default contraction constant. Default 0.45.
    Q_c : float
        Causal snap L2 norm convergence threshold. Default 1e-5.
    tau : float
        Vacuum Brake decay temperature. Default 1.0.
    theta_target : float
        DIEPT phase angle threshold. Default 0.70.
    caki_threshold : float
        The numerical threshold of CAKI that triggers crystallization. Default 0.80.
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
        self.V = vocab_size
        self.k = k_contractive
        self.Q_c = Q_c
        self.tau = tau
        self.caki_threshold = caki_threshold
        self.delta_finite = delta_finite
        
        # Initialize internal libraries and modules
        self.fdsa_search = FractalDeductionSearch()
        self.pruner = VectorizedFDSAPruner(self.V, self.fdsa_search)
        
        self.verifier = EpistemicVerificationSuite(
            cwf_penalty_matrix=cwf_penalty_matrix,
            theta_target=theta_target,
            lambda_R=0.35,
            lambda_L=0.45,
            lambda_D=0.20,
        )

    # ------------------------------------------------------------------
    # Core Steering Logic (based on actualizer_engine.py)
    # ------------------------------------------------------------------
    def _softmax(self, logits: List[float]) -> List[float]:
        max_l = max(x for x in logits if x != -math.inf)
        exp_l, probs = [], [0.0] * self.V
        valid_idx_map = []
        for i, x in enumerate(logits):
            if x != -math.inf:
                val = math.exp(x - max_l)
                exp_l.append(val)
                valid_idx_map.append(i)

        total = sum(exp_l) or 1.0
        for j, i in enumerate(valid_idx_map):
            probs[i] = exp_l[j] / total
        return probs

    def compute_drift_tensor(
        self,
        U: List[float],
        history: List[int],
        target_tokens: Set[int],
    ) -> List[float]:
        # Weights from CKT standards: Order, Justice, Knowledge
        w_L, w_G, w_F = 0.35, 0.35, 0.20
        D = [0.0] * self.V

        # Local Recency Drift (Order Prime)
        lookback = history[-8:]
        for step_back, tok in enumerate(reversed(lookback)):
            if 0 <= tok < self.V:
                recency_weight = math.exp(-0.4 * step_back)
                D[tok] += w_L * 2.0 * recency_weight

        # Global Drift (Justice) and Future Drift (Knowledge)
        for v in range(self.V):
            if U[v] == 0.0:
                continue
            if v not in target_tokens:
                D[v] += w_G * 1.5
            D[v] += w_F * (-math.log(max(U[v], 1e-12)) * 0.08)

        return D

    def apply_vacuum_brake(self, U: List[float], D: List[float]) -> List[float]:
        decay = [math.exp(-d / self.tau) for d in D]
        U_braked = [U[i] * decay[i] for i in range(self.V)]
        total = sum(U_braked) or 1.0
        return [x / total for x in U_braked]

    def steer_next_token(
        self,
        logits: List[float],
        history: List[int],
        target_tokens: Set[int],
        context_type: str = "general",
    ) -> Tuple[int, List[float], float, int, ReferenceDomain, float]:
        """
        Executes the pre-inference pruning and the contractive actualization loop
        to select the next single token.
        """
        # Step 1: Run pre-inference pruner (FDSA Phase)
        pruned_logits, active_size, anchor_domain, similarity = self.pruner.prune_vocabulary(
            logits, history[-1] if history else -1, {}, context_type
        )
        
        # Step 2: Initialize substrate
        U = self._softmax(pruned_logits)
        
        # Step 3: Run contraction loop
        k_step = anchor_domain.k  # Inherit contraction factor from anchor
        
        for iteration in range(1, 21):
            U_prev = U[:]
            D = self.compute_drift_tensor(U, history, target_tokens)
            U_b = self.apply_vacuum_brake(U, D)
            
            # Banach contractive mapping
            U = [k_step * U_b[v] + (1.0 - k_step) * U_prev[v] for v in range(self.V)]
            
            # L2 norm delta convergence
            delta = math.sqrt(sum((U[v] - U_prev[v]) ** 2 for v in range(self.V)))
            if delta <= self.Q_c:
                break
                
        selected_token = max(range(self.V), key=lambda v: U[v])
        final_drift = self.compute_drift_tensor(U, history, target_tokens)[selected_token]
        
        return selected_token, U, final_drift, iteration, anchor_domain, similarity

    # ------------------------------------------------------------------
    # Thought Verification & Crystallization
    # ------------------------------------------------------------------
    def calculate_caki(self, thought: CandidateThought) -> Tuple[float, float, float, float]:
        """
        Calculates the Concise Accumulated Knowledge Index (CAKI) for a thought.
        K_acc = (I_eff / R) * exp(-L / (T + eps))
        """
        chain = thought.causal_chain
        I_in = len(chain)
        if I_in == 0:
            return 0.0, 0.0, 1.0, 0.0
            
        # Calculate CWF violations (sum of causal penalties)
        L_violations = 0.0
        for i in range(I_in - 1):
            transition = (chain[i], chain[i+1])
            L_violations += self.verifier.cwf.get(transition, 0.0)
            
        # Redundancy R
        duplicates = I_in - len(set(chain))
        R = 1.0 + (duplicates / I_in)
        
        # Effective Information I_eff = max(0, I_in - L_violations)
        I_eff = max(0.0, float(I_in) - L_violations)
        
        # Exponential suppression
        T = 0.5  # Tolerance floor
        eps = 1e-6
        exponent = -L_violations / (T + eps)
        exp_factor = math.exp(exponent)
        
        K_acc = (I_eff / R) * exp_factor
        
        # Normalized CAKI: CAKI = K_acc / I_max.
        # Under Theorem 7 (Unsimulability of Reality), no finite simulation can contain the live
        # Prime-combination law, meaning CAKI must asymptotically approach but never reach 1.0.
        # We define I_max = I_in + delta_finite (where delta_finite > 0).
        I_max = float(I_in) + self.delta_finite
        caki = K_acc / I_max
        
        return caki, L_violations, R, I_eff

    def verify_and_crystallize(
        self,
        thought: CandidateThought,
        delta_C_R: float,
        domain_name: str,
        description: str = "",
    ) -> Tuple[bool, Optional[MCE], float]:
        """
        Applies Pipeline A and C, computes cost and CAKI, and decides whether
        to freeze the thought into an MCE super-cluster.
        
        Returns
        -------
        is_crystallized : bool
        mce_obj         : Optional[MCE]
        caki            : float
        """
        # 1. Pipeline A: Causation-Chain Check
        valid_a = self.verifier.run_pipeline_a(thought)
        if not valid_a:
            return False, None, 0.0
            
        # 2. Pipeline C: Negentropy (DIEPT) Check
        valid_c = self.verifier.run_pipeline_c(thought)
        if not valid_c:
            return False, None, 0.0
            
        # 3. Calculate CAKI
        caki, L_viol, R, I_eff = self.calculate_caki(thought)
        
        # 4. Check Crystallization Conditions:
        # - Thought must be stable/concise (CAKI >= threshold)
        # - Thought must reduce total system cost (delta_C_R < 0)
        # (For demonstration, if delta_C_R <= 0 we allow crystallization if CAKI is high)
        if caki >= self.caki_threshold and delta_C_R <= 0.0:
            # Construct crystallized MCE object
            # Mass: m = J_score * K_score * len(chain)
            mass = thought.primes["Justice"] * thought.primes["Knowledge"] * len(thought.causal_chain)
            
            # Complexity: c = min(Primes)
            complexity = min(thought.primes.values())
            
            # Entropy: e = Defect Function D(Omega) + path drift
            # We proxy this from R and L
            entropy = (R - 1.0) + (L_viol / len(thought.causal_chain))
            
            # Generate the immutable MCE object
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
            
            # Crystallization: dynamic insertion into FDSA Analogy Library
            self.fdsa_search.add_reference_domain(mce_obj)
            return True, mce_obj, caki
            
        return False, None, caki

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
        Executes the full PBI Cognitive Life Cycle end-to-end:
        Stage A0: Parse Question Operator & extract Epistemic Architecture Triple.
        Stage 1: Generate causal proposal (FDSA).
        Stage 2: Verification via Actualizer (Pipelines A/C & Cost loop).
        Stage 3: Crystallization into MCE Class + Linguistic Modality.
        """
        # Stage A0: Question Operator Parser
        # Maps to DomainLens, PrimeVector, and theta_target
        domain, prime_vector, theta_target = QuestionOperatorParser.parse_operator(query.split()[0])
        self.verifier.theta_target = theta_target
        
        # Stage 1: Thought Generation (FDSA)
        # Mocking a short 3-token chain generation for the lifecycle
        chain = list(initial_history)
        total_propensity = 1.0
        
        # We simulate steering the next two tokens
        for _ in range(2):
            # Mock uniform logits
            logits = [1.0] * self.V
            token, U, drift, iters, anchor, sim = self.steer_next_token(
                logits, chain, target_tokens, context_type=domain
            )
            chain.append(token)
            total_propensity *= U[token]
            
        # Initialize the Candidate Thought with DIEPT state
        diept_state = DIEPTState(simulated_diept_a, simulated_diept_b)
        thought = CandidateThought(
            causal_chain=chain,
            propensity=total_propensity,
            diept_state=diept_state
        )
        
        # Stage 2 & 3: Verification and Crystallization
        is_cryst, mce_obj, caki = self.verify_and_crystallize(
            thought=thought,
            delta_C_R=delta_c_r_sim,
            domain_name=domain,
            description=f"Generated from query: {query}"
        )
        
        # Linguistic Marker based on phase angle
        marker = thought.diept_state.get_linguistic_marker(theta_target)
        
        return marker, is_cryst, mce_obj, caki
