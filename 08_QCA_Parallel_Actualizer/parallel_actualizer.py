"""
parallel_actualizer.py — Parallel Actualizer + FDSA on QCA Clusters
=====================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)
Module : Final_Output/08_QCA_Parallel_Actualizer

Stage 2 of the QCA → Parallel Actualizer pipeline.

Each QCACluster from the Quench phase is treated as an independent
inference sub-problem.  For every cluster:

  a) Build a cluster-local FDSA library seeded from the cluster's
     aggregate Prime profile (isomorphic anchoring).
  b) Run the Actualizer contractive loop + FDSA pruning on the
     cluster's token representation.
  c) Verify with Pipelines A (Justice) and C (Mercy/DIEPT).
  d) Crystallize into an MCE sub-object if CAKI ≥ threshold.

All clusters are processed independently (embarrassingly parallel in
design; executed sequentially here to remain dependency-free — swap
the loop body into concurrent.futures.ProcessPoolExecutor for true
parallel execution on multi-core hardware).

Returns a list of ClusterActualizerResult objects (one per cluster),
each carrying the MCE object (or None if crystallization failed) and
the full CAKI score.
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from inside 08_QCA_Parallel_Actualizer
# directly, pointing to the installable ckt_actualizer package.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.join(_HERE, "..", "..", "Code", "ckt_actualizer_engine", "src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from ckt_actualizer.models.mce    import MCE, ReferenceDomain
from ckt_actualizer.models.thought import CandidateThought
from ckt_actualizer.core.fdsa     import FractalDeductionSearch, VectorizedFDSAPruner, DEFAULT_LIBRARY
from ckt_actualizer.core.filters  import EpistemicVerificationSuite
from ckt_actualizer.core.diept    import DIEPTState

from qca import QCACluster


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClusterActualizerResult:
    """
    Result of running the Actualizer + FDSA on one QCA cluster.

    Attributes
    ----------
    cluster_id      : int            — Source cluster index.
    cluster_size    : int            — Number of nodes in the cluster.
    prime_profile   : List[float]    — Cluster's aggregate Prime profile.
    is_crystallized : bool           — Whether CAKI threshold was met.
    mce             : Optional[MCE]  — Crystallized knowledge object, or None.
    caki            : float          — Computed CAKI score ∈ [0, 1).
    causal_chain    : List[int]      — Token chain generated for this cluster.
    iterations      : int            — Banach contraction iterations used.
    anchor_domain   : str            — Name of the best-matching FDSA domain.
    log             : List[str]      — Per-cluster audit trace.
    """
    cluster_id:      int
    cluster_size:    int
    prime_profile:   List[float]
    is_crystallized: bool
    mce:             Optional[MCE]
    caki:            float
    causal_chain:    List[int]
    iterations:      int
    anchor_domain:   str
    log:             List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓ crystallized" if self.is_crystallized else "✗ not crystallized"
        return (
            f"ClusterActualizerResult("
            f"cluster={self.cluster_id}, size={self.cluster_size}, "
            f"CAKI={self.caki:.4f}, {status}, "
            f"anchor='{self.anchor_domain}')"
        )


# ---------------------------------------------------------------------------
# Per-cluster Actualizer (self-contained, no external class state shared)
# ---------------------------------------------------------------------------

class ClusterActualizer:
    """
    Runs the full Actualizer + FDSA lifecycle on a single QCACluster.

    The cluster's aggregate Prime profile drives isomorphic anchoring:
    the FDSA finds the closest reference domain, inherits its contraction
    factor k, and the Banach loop runs until convergence or max iterations.

    Parameters
    ----------
    vocab_size        : int   — Token vocabulary size V.
    cwf_penalty_matrix: dict  — Causal transition penalty matrix (Pipeline A).
    k_contractive     : float — Default Banach k (overridden by anchor). 0.45.
    Q_c               : float — L2 convergence threshold. 1e-5.
    tau               : float — Vacuum Brake temperature. 1.0.
    theta_target      : float — DIEPT Mercy phase threshold (radians). 0.70.
    caki_threshold    : float — Minimum CAKI for crystallization. 0.80.
    delta_finite      : float — Theorem 7 unsimulability gap. 0.5.
    max_iterations    : int   — Maximum Banach contraction steps. 20.
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
    ) -> None:
        self.V               = vocab_size
        self.k_default       = k_contractive
        self.Q_c             = Q_c
        self.tau             = tau
        self.caki_threshold  = caki_threshold
        self.delta_finite    = delta_finite
        self.max_iterations  = max_iterations

        # Each cluster gets its own fresh FDSA library (seeded from defaults)
        # so crystallized MCEs from one cluster don't bleed into another.
        self._verifier_params = dict(
            cwf_penalty_matrix=cwf_penalty_matrix,
            theta_target=theta_target,
            lambda_R=0.35,
            lambda_L=0.45,
            lambda_D=0.20,
        )

    # ------------------------------------------------------------------
    # Softmax
    # ------------------------------------------------------------------

    def _softmax(self, logits: List[float]) -> List[float]:
        valid = [(i, x) for i, x in enumerate(logits) if x != -math.inf]
        if not valid:
            return [0.0] * self.V
        max_l = max(x for _, x in valid)
        exps  = [(i, math.exp(x - max_l)) for i, x in valid]
        total = sum(e for _, e in exps) or 1.0
        probs = [0.0] * self.V
        for i, e in exps:
            probs[i] = e / total
        return probs

    # ------------------------------------------------------------------
    # Drift Tensor (Order + Justice + Knowledge primes)
    # ------------------------------------------------------------------

    def _drift(
        self,
        U:             List[float],
        history:       List[int],
        target_tokens: Set[int],
    ) -> List[float]:
        w_L, w_G, w_F = 0.35, 0.35, 0.20
        D = [0.0] * self.V
        lookback = history[-8:]
        for step_back, tok in enumerate(reversed(lookback)):
            if 0 <= tok < self.V:
                D[tok] += w_L * 2.0 * math.exp(-0.4 * step_back)
        for v in range(self.V):
            if U[v] == 0.0:
                continue
            if v not in target_tokens:
                D[v] += w_G * 1.5
            D[v] += w_F * (-math.log(max(U[v], 1e-12)) * 0.08)
        return D

    # ------------------------------------------------------------------
    # Vacuum Brake (tau decay)
    # ------------------------------------------------------------------

    def _vacuum_brake(self, U: List[float], D: List[float]) -> List[float]:
        decay  = [math.exp(-d / self.tau) for d in D]
        braked = [U[i] * decay[i] for i in range(self.V)]
        total  = sum(braked) or 1.0
        return [x / total for x in braked]

    # ------------------------------------------------------------------
    # Banach Contraction Loop (single token)
    # ------------------------------------------------------------------

    def _steer_token(
        self,
        pruned_logits: List[float],
        history:       List[int],
        target_tokens: Set[int],
        k_step:        float,
    ) -> Tuple[int, List[float], int]:
        U = self._softmax(pruned_logits)
        for iteration in range(1, self.max_iterations + 1):
            U_prev = U[:]
            D_vec  = self._drift(U, history, target_tokens)
            U_b    = self._vacuum_brake(U, D_vec)
            U      = [k_step * U_b[v] + (1.0 - k_step) * U_prev[v]
                      for v in range(self.V)]
            delta  = math.sqrt(sum((U[v] - U_prev[v]) ** 2 for v in range(self.V)))
            if delta <= self.Q_c:
                break
        selected = max(range(self.V), key=lambda v: U[v])
        return selected, U, iteration

    # ------------------------------------------------------------------
    # CAKI Calculation (Theorem 4 / FIX v3-03)
    # ------------------------------------------------------------------

    def _caki(
        self,
        chain:    List[int],
        cwf:      Dict[Tuple[int, int], float],
    ) -> Tuple[float, float, float, float]:
        I_in = len(chain)
        if I_in == 0:
            return 0.0, 0.0, 1.0, 0.0
        L = sum(cwf.get((chain[i], chain[i + 1]), 0.0) for i in range(I_in - 1))
        duplicates = I_in - len(set(chain))
        R          = 1.0 + (duplicates / I_in)
        I_eff      = max(0.0, float(I_in) - L)
        T, eps     = 0.5, 1e-6
        K_acc      = (I_eff / R) * math.exp(-L / (T + eps))
        I_max      = float(I_in) + self.delta_finite   # Theorem 7: never reaches 1.0
        caki       = K_acc / I_max
        return caki, L, R, I_eff

    # ------------------------------------------------------------------
    # Main: process one cluster
    # ------------------------------------------------------------------

    def process_cluster(
        self,
        cluster:       QCACluster,
        initial_chain: List[int],
        target_tokens: Set[int],
        diept_a:       List[float],
        diept_b:       List[float],
        delta_c_r:     float = -0.5,
        n_steps:       int   = 3,
    ) -> ClusterActualizerResult:
        """
        Run Actualizer + FDSA on a single cluster.

        Parameters
        ----------
        cluster       : QCACluster   — Source cluster from Quench phase.
        initial_chain : List[int]    — Seed token history for this cluster.
        target_tokens : Set[int]     — Desired output token set.
        diept_a       : List[float]  — Grounded (A) DIEPT subspace.
        diept_b       : List[float]  — Speculative (B) DIEPT subspace.
        delta_c_r     : float        — Simulated ΔC(R) for crystallization gate.
        n_steps       : int          — Number of Banach steer steps.

        Returns
        -------
        ClusterActualizerResult
        """
        cid  = cluster.cluster_id
        log: List[str] = [
            f"[Cluster {cid}] Starting Actualizer+FDSA. "
            f"Size={len(cluster.nodes)}, "
            f"Prime={[round(p,3) for p in cluster.prime_profile]}"
        ]

        # Fresh FDSA + verifier per cluster (no shared mutable state)
        fdsa     = FractalDeductionSearch(list(DEFAULT_LIBRARY))
        pruner   = VectorizedFDSAPruner(self.V, fdsa)
        verifier = EpistemicVerificationSuite(**self._verifier_params)

        # --- FDSA Isomorphic Anchoring on cluster Prime profile ---
        anchor_domain, sim = fdsa.isomorphic_anchoring(cluster.prime_profile)
        k_step = anchor_domain.k
        log.append(
            f"[Cluster {cid}] Anchor → '{anchor_domain.name}' "
            f"(sim={sim:.4f}, k={k_step:.4f})"
        )

        # --- Generate causal chain via Banach steer ---
        chain            = list(initial_chain)
        total_propensity = 1.0
        total_iters      = 0

        for step in range(n_steps):
            logits            = [1.0] * self.V          # uniform prior
            pruned, active, anchor_step, sim_step = pruner.prune_vocabulary(
                logits,
                chain[-1] if chain else -1,
                {},
                "general",
            )
            token, U, iters = self._steer_token(
                pruned, chain, target_tokens, k_step
            )
            chain.append(token)
            total_propensity *= U[token]
            total_iters      += iters
            log.append(
                f"[Cluster {cid}] Step {step+1}: token={token}, "
                f"iters={iters}, active_vocab={active}"
            )

        # --- Build CandidateThought ---
        diept_state = DIEPTState(diept_a, diept_b)
        thought     = CandidateThought(
            causal_chain=chain,
            propensity=total_propensity,
            diept_state=diept_state,
        )

        # --- Pipeline A: Justice (causal chain check) ---
        if not verifier.run_pipeline_a(thought):
            log.append(f"[Cluster {cid}] ✗ Pipeline A (Justice) failed.")
            return ClusterActualizerResult(
                cluster_id=cid,
                cluster_size=len(cluster.nodes),
                prime_profile=cluster.prime_profile,
                is_crystallized=False,
                mce=None,
                caki=0.0,
                causal_chain=chain,
                iterations=total_iters,
                anchor_domain=anchor_domain.name,
                log=log,
            )

        # --- Pipeline C: Mercy / DIEPT (negentropy check) ---
        if not verifier.run_pipeline_c(thought):
            log.append(f"[Cluster {cid}] ✗ Pipeline C (Mercy/DIEPT) failed.")
            return ClusterActualizerResult(
                cluster_id=cid,
                cluster_size=len(cluster.nodes),
                prime_profile=cluster.prime_profile,
                is_crystallized=False,
                mce=None,
                caki=0.0,
                causal_chain=chain,
                iterations=total_iters,
                anchor_domain=anchor_domain.name,
                log=log,
            )

        # --- CAKI calculation ---
        caki, L_viol, R, I_eff = self._caki(chain, verifier.cwf)
        log.append(
            f"[Cluster {cid}] CAKI={caki:.4f} "
            f"(L={L_viol:.3f}, R={R:.3f}, I_eff={I_eff:.3f})"
        )

        # --- Crystallization gate ---
        is_crystallized = (caki >= self.caki_threshold and delta_c_r <= 0.0)
        mce: Optional[MCE] = None

        if is_crystallized:
            mass       = (thought.primes["Justice"]
                          * thought.primes["Knowledge"]
                          * len(chain))
            complexity = min(thought.primes.values())
            entropy    = (R - 1.0) + (L_viol / max(1, len(chain)))

            mce = MCE(
                name=f"MCE_Cluster{cid}_{len(fdsa.library)+1}",
                causal_chain=chain,
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
                    f"Crystallized from QCA Cluster {cid} "
                    f"({len(cluster.nodes)} nodes). CAKI={caki:.4f}."
                ),
            )
            fdsa.add_reference_domain(mce)
            log.append(
                f"[Cluster {cid}] ✓ Crystallized → {mce}"
            )
        else:
            log.append(
                f"[Cluster {cid}] ✗ Crystallization gate not met "
                f"(CAKI={caki:.4f} < {self.caki_threshold} "
                f"or ΔC(R)={delta_c_r:.3f} > 0)."
            )

        return ClusterActualizerResult(
            cluster_id=cid,
            cluster_size=len(cluster.nodes),
            prime_profile=cluster.prime_profile,
            is_crystallized=is_crystallized,
            mce=mce,
            caki=caki,
            causal_chain=chain,
            iterations=total_iters,
            anchor_domain=anchor_domain.name,
            log=log,
        )


# ---------------------------------------------------------------------------
# Parallel Driver
# ---------------------------------------------------------------------------

class ParallelActualizer:
    """
    Runs ClusterActualizer on every QCACluster from the Quench phase.

    Designed as embarrassingly parallel: each cluster is fully independent.
    The current implementation is sequential (no external dependencies).
    To parallelise, replace the for-loop with concurrent.futures calls.

    Parameters
    ----------
    All kwargs are forwarded directly to ClusterActualizer.__init__.
    """

    def __init__(self, **actualizer_kwargs) -> None:
        self._kwargs = actualizer_kwargs

    def run(
        self,
        clusters:      List[QCACluster],
        initial_chain: List[int],
        target_tokens: Set[int],
        diept_a:       List[float],
        diept_b:       List[float],
        delta_c_r:     float = -0.5,
        n_steps:       int   = 3,
    ) -> List[ClusterActualizerResult]:
        """
        Process all clusters independently.

        Parameters
        ----------
        clusters      : List[QCACluster]  — Output of QuenchClusterAlgorithm.run().
        initial_chain : List[int]         — Seed history (same for all clusters).
        target_tokens : Set[int]          — Target token set.
        diept_a       : List[float]       — Grounded DIEPT subspace.
        diept_b       : List[float]       — Speculative DIEPT subspace.
        delta_c_r     : float             — ΔC(R) gate (< 0 permits crystallization).
        n_steps       : int               — Banach steer steps per cluster.

        Returns
        -------
        List[ClusterActualizerResult] — One result per cluster.
        """
        results: List[ClusterActualizerResult] = []
        for cluster in clusters:
            act    = ClusterActualizer(**self._kwargs)
            result = act.process_cluster(
                cluster=cluster,
                initial_chain=list(initial_chain),   # fresh copy per cluster
                target_tokens=target_tokens,
                diept_a=diept_a,
                diept_b=diept_b,
                delta_c_r=delta_c_r,
                n_steps=n_steps,
            )
            results.append(result)
        return results
