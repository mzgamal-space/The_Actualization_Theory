"""
global_actualizer.py — Actualizer + FDSA on Parallel MCE Results
=================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)
Module : Final_Output/08_QCA_Parallel_Actualizer

Stage 3 of the QCA → Parallel Actualizer pipeline.

Takes the collection of MCE sub-objects crystallized by the Parallel
Actualizer (one per QCA cluster) and runs a second-pass Actualizer +
FDSA over them to produce the final global solution.

The crystallized cluster MCEs are injected into the FDSA analogy library
*before* the final steer, so the global Banach loop is anchored to the
best-matching cluster solution rather than a generic reference domain.
This implements the "Actualizer + FDSA on Previous Results" step from
the pipeline specification.

Pipeline
--------
  MCE sub-objects (from Stage 2)
    → Inject all MCEs into FDSA library
    → FDSA isomorphic anchoring on aggregate Prime profile
    → Banach contractive steer (global chain extension)
    → Pipeline A + C verification
    → CAKI + crystallization gate
    → GlobalSolution (final MCE + full audit)
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "..", "02_Core_Engine")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from mce     import MCE
from thought  import CandidateThought
from filters   import EpistemicVerificationSuite
from diept     import DIEPTState

from actualizer_engine import ActualizerEngine
from fdsa_pruner       import FractalDeductionSearch, VectorizedFDSAPruner, DEFAULT_LIBRARY, ReferenceDomain

from parallel_actualizer import ClusterActualizerResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GlobalSolution:
    """
    Final output of the three-stage QCA → Parallel Actualizer pipeline.

    Attributes
    ----------
    is_crystallized    : bool           — Whether global CAKI threshold was met.
    final_mce          : Optional[MCE]  — Final crystallized knowledge object.
    global_caki        : float          — Global CAKI ∈ [0, 1).
    global_chain       : List[int]      — Final extended causal chain.
    n_cluster_mces     : int            — Number of cluster MCEs injected.
    anchor_domain      : str            — FDSA anchor used in global pass.
    anchor_similarity  : float          — Cosine similarity of anchor match.
    log                : List[str]      — Full audit trail.
    """
    is_crystallized:   bool
    final_mce:         Optional[MCE]
    global_caki:       float
    global_chain:      List[int]
    n_cluster_mces:    int
    anchor_domain:     str
    anchor_similarity: float
    log:               List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓ crystallized" if self.is_crystallized else "✗ not crystallized"
        return (
            f"GlobalSolution("
            f"CAKI={self.global_caki:.4f}, {status}, "
            f"cluster_MCEs={self.n_cluster_mces}, "
            f"anchor='{self.anchor_domain}')"
        )


# ---------------------------------------------------------------------------
# Global Actualizer
# ---------------------------------------------------------------------------

class GlobalActualizer:
    """
    Runs the final Actualizer + FDSA pass over all cluster MCE results.

    The key insight: crystallized cluster MCEs are injected into the FDSA
    analogy library as first-class reference domains.  The global steer
    therefore operates in a search space already constrained by validated
    sub-solutions — the Banach loop converges faster and anchors to
    problem-specific structure rather than generic physical analogies.

    Parameters
    ----------
    vocab_size         : int   — Token vocabulary size V.
    cwf_penalty_matrix : dict  — Causal penalty matrix for Pipeline A.
    k_contractive      : float — Default Banach k (overridden by anchor). 0.45.
    Q_c                : float — L2 convergence threshold. 1e-5.
    tau                : float — Vacuum Brake temperature. 1.0.
    theta_target       : float — DIEPT Mercy threshold (radians). 0.70.
    caki_threshold     : float — Minimum CAKI for crystallization. 0.80.
    delta_finite       : float — Theorem 7 unsimulability gap. 0.5.
    max_iterations     : int   — Maximum Banach steps. 20.
    n_steps            : int   — Number of global steer steps. 3.
    """

    def __init__(
        self,
        vocab_size:         int,
        cwf_penalty_matrix: Dict[Tuple[int, int], float],
        k_contractive:      float = 0.45,
        Q_c:                float = 1e-5,
        tau:                float = 1.0,
        theta_target:       float = 0.70,
        caki_threshold:     float = 0.80,
        delta_finite:       float = 0.5,
        max_iterations:     int   = 20,
        n_steps:            int   = 3,
    ) -> None:
        self.V              = vocab_size
        self.k_default      = k_contractive
        self.Q_c            = Q_c
        self.tau            = tau
        self.caki_threshold = caki_threshold
        self.delta_finite   = delta_finite
        self.max_iterations = max_iterations
        self.n_steps        = n_steps
        self._cwf           = cwf_penalty_matrix
        self._theta_target  = theta_target

    # ------------------------------------------------------------------
    # Helpers (same Banach machinery as ClusterActualizer — no inheritance
    # to keep modules fully self-contained and independently testable)
    # ------------------------------------------------------------------

    def _caki(
        self,
        chain: List[int],
        cwf:   Dict[Tuple[int, int], float],
    ) -> Tuple[float, float, float, float]:
        I_in = len(chain)
        if I_in == 0:
            return 0.0, 0.0, 1.0, 0.0
        L      = sum(cwf.get((chain[i], chain[i + 1]), 0.0) for i in range(I_in - 1))
        dups   = I_in - len(set(chain))
        R      = 1.0 + (dups / I_in)
        I_eff  = max(0.0, float(I_in) - L)
        T, eps = 0.5, 1e-6
        K_acc  = (I_eff / R) * math.exp(-L / (T + eps))
        caki   = K_acc / (float(I_in) + self.delta_finite)
        return caki, L, R, I_eff

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def run(
        self,
        cluster_results: List[ClusterActualizerResult],
        initial_chain:   List[int],
        target_tokens:   Set[int],
        diept_a:         List[float],
        diept_b:         List[float],
        delta_c_r:       float = -0.5,
    ) -> GlobalSolution:
        """
        Run the global Actualizer + FDSA over all cluster MCE results.

        Parameters
        ----------
        cluster_results : List[ClusterActualizerResult]
            Output of ParallelActualizer.run().
        initial_chain   : List[int]    — Seed history for global pass.
        target_tokens   : Set[int]     — Global target token set.
        diept_a         : List[float]  — Grounded DIEPT subspace.
        diept_b         : List[float]  — Speculative DIEPT subspace.
        delta_c_r       : float        — ΔC(R) gate for crystallization.

        Returns
        -------
        GlobalSolution
        """
        log: List[str] = ["[Global Actualizer] Starting final pass over cluster MCEs."]

        # --- Collect crystallized MCEs from cluster stage ---
        cluster_mces: List[MCE] = [
            r.mce for r in cluster_results if r.mce is not None
        ]
        log.append(
            f"[Global Actualizer] {len(cluster_mces)} / {len(cluster_results)} "
            f"clusters crystallized → injecting into FDSA library."
        )

        # --- Build global FDSA library: defaults + all cluster MCEs ---
        # Initialize FDSA directly with combined library
        fdsa = FractalDeductionSearch(list(DEFAULT_LIBRARY) + cluster_mces)
        
        # Initialize pruner and inject the updated fdsa engine
        pruner = VectorizedFDSAPruner(self.V)
        pruner.fdsa = fdsa

        verifier = EpistemicVerificationSuite(
            cwf_penalty_matrix=self._cwf,
            theta_target=self._theta_target,
            lambda_R=0.35,
            lambda_L=0.45,
            lambda_D=0.20,
        )
        log.append(
            f"[Global Actualizer] FDSA library size: {len(fdsa.library)} domains "
            f"({len(DEFAULT_LIBRARY)} built-in + {len(cluster_mces)} cluster MCEs)."
        )

        # --- Compute aggregate Prime profile over all cluster results ---
        n_res = len(cluster_results)
        if n_res > 0:
            agg_prime = [
                sum(r.prime_profile[p] for r in cluster_results) / n_res
                for p in range(5)
            ]
        else:
            agg_prime = [0.5] * 5

        # --- FDSA Isomorphic Anchoring on aggregate Prime ---
        anchor_domain, anchor_sim = fdsa.isomorphic_anchoring(agg_prime)
        k_step = anchor_domain.k
        log.append(
            f"[Global Actualizer] Aggregate Prime: [{', '.join(f'{p:.3f}' for p in agg_prime)}]"
        )
        log.append(
            f"[Global Actualizer] Anchor → '{anchor_domain.name}' "
            f"(sim={anchor_sim:.4f}, k={k_step:.4f})"
        )

        # --- Seed global chain from initial chain + cluster chain tails ---
        global_chain = list(initial_chain)
        for r in cluster_results:
            if r.causal_chain:
                # Append last token of each cluster chain as context seeds
                global_chain.append(r.causal_chain[-1])
        log.append(
            f"[Global Actualizer] Global chain seeded: {len(global_chain)} tokens."
        )

        # --- Banach steer: extend global chain n_steps further ---
        total_iters = 0

        engine = ActualizerEngine(
            vocab_size=self.V,
            mercy_k=k_step,
            Q_c=self.Q_c,
            tau=self.tau,
            max_iters=self.max_iterations,
        )

        for step in range(self.n_steps):
            logits            = [1.0] * self.V
            pruned, active = pruner.prune_vocabulary(
                logits,
                global_chain[-1] if global_chain else -1,
                {},
                "general",
            )
            token, U, Tr_D, iters, nu_hist, actualized = engine.steer(
                pruned, global_chain, target_tokens
            )
            global_chain.append(token)
            total_iters += iters
            log.append(
                f"[Global Actualizer] Step {step+1}: token={token}, "
                f"iters={iters}, active_vocab={active}, Tr_D={Tr_D:.4f}, actualized={actualized}"
            )

        # --- Build CandidateThought ---
        diept_state = DIEPTState(diept_a, diept_b)
        thought     = CandidateThought(
            causal_chain=global_chain,
            propensity=1.0,
            diept_state=diept_state,
        )

        # --- Pipeline A ---
        if not verifier.run_pipeline_a(thought):
            log.append("[Global Actualizer] ✗ Pipeline A (Justice) failed.")
            return GlobalSolution(
                is_crystallized=False,
                final_mce=None,
                global_caki=0.0,
                global_chain=global_chain,
                n_cluster_mces=len(cluster_mces),
                anchor_domain=anchor_domain.name,
                anchor_similarity=anchor_sim,
                log=log,
            )

        # --- Pipeline C ---
        if not verifier.run_pipeline_c(thought):
            log.append("[Global Actualizer] ✗ Pipeline C (Mercy/DIEPT) failed.")
            return GlobalSolution(
                is_crystallized=False,
                final_mce=None,
                global_caki=0.0,
                global_chain=global_chain,
                n_cluster_mces=len(cluster_mces),
                anchor_domain=anchor_domain.name,
                anchor_similarity=anchor_sim,
                log=log,
            )

        # --- Global CAKI ---
        global_caki, L_viol, R, I_eff = self._caki(global_chain, verifier.cwf)
        log.append(
            f"[Global Actualizer] Global CAKI={global_caki:.4f} "
            f"(L={L_viol:.3f}, R={R:.3f}, I_eff={I_eff:.3f})"
        )

        # --- Crystallization gate ---
        is_cryst = (global_caki >= self.caki_threshold and delta_c_r <= 0.0)
        final_mce: Optional[MCE] = None

        if is_cryst:
            mass       = (thought.primes["Justice"]
                          * thought.primes["Knowledge"]
                          * len(global_chain))
            complexity = min(thought.primes.values())
            entropy    = (R - 1.0) + (L_viol / max(1, len(global_chain)))

            final_mce = MCE(
                name=f"MCE_Global_Final_{len(fdsa.library)+1}",
                causal_chain=global_chain,
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
                description=(
                    f"Global MCE synthesized from {len(cluster_mces)} cluster MCEs. "
                    f"Anchor: '{anchor_domain.name}'. "
                    f"Global CAKI={global_caki:.4f}."
                ),
            )
            log.append(f"[Global Actualizer] ✓ Final crystallization → {final_mce}")
        else:
            log.append(
                f"[Global Actualizer] ✗ Global crystallization gate not met "
                f"(CAKI={global_caki:.4f} < {self.caki_threshold} "
                f"or ΔC(R)={delta_c_r:.3f} > 0)."
            )

        log.append("[Global Actualizer] Pipeline complete.")
        return GlobalSolution(
            is_crystallized=is_cryst,
            final_mce=final_mce,
            global_caki=global_caki,
            global_chain=global_chain,
            n_cluster_mces=len(cluster_mces),
            anchor_domain=anchor_domain.name,
            anchor_similarity=anchor_sim,
            log=log,
        )
