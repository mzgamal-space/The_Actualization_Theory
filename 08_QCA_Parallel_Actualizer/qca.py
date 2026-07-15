"""
qca.py — Quench-Cluster Algorithm: Quench Phase Only
=====================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)
Module : Final_Output/08_QCA_Parallel_Actualizer

Implements the two-step Quench-only QCA as the crystallization front-end
for the Parallel Actualizer pipeline:

  Step 1 — Distance Matrix (Plasma substrate):  O(N²)
  Step 2 — Quench binding via T_q threshold:    O(N) per cluster

The canonical Quench Temperature is the RGG-derived form (Issue M-QCA-02):

    T_q^RGG = γ · √( A · ln(N/K) / (π · N) )

Each QCACluster produced here is an independent sub-problem passed directly
to the Parallel Actualizer + FDSA stage (parallel_actualizer.py).

Reference: CKT White Paper v3, §7.2 — Theorem 2 Corollary: K parallel
clusters each solve a sub-problem of size N/K at cost O((N/K)²);
aggregate = N²/K — a factor-K improvement over sequential solving.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class QCANode:
    """
    A single node in the problem space.

    Attributes
    ----------
    node_id       : int         — Unique identifier.
    coords        : List[float] — Spatial / embedding coordinates.
    prime_profile : List[float] — [Order, Justice, Mercy, Knowledge, Power].
    metadata      : dict        — Arbitrary problem-domain payload.
    """
    node_id:       int
    coords:        List[float]
    prime_profile: List[float] = field(default_factory=lambda: [0.5] * 5)
    metadata:      dict        = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"QCANode(id={self.node_id})"


@dataclass
class QCACluster:
    """
    A crystallization cluster produced by the Quench phase.

    Attributes
    ----------
    cluster_id    : int             — Cluster index.
    nodes         : List[QCANode]   — Member nodes.
    centroid      : List[float]     — Element-wise mean of member coordinates.
    prime_profile : List[float]     — Element-wise mean of member Prime profiles.
    """
    cluster_id:    int
    nodes:         List[QCANode]
    centroid:      List[float]  = field(default_factory=list)
    prime_profile: List[float]  = field(default_factory=lambda: [0.5] * 5)

    def __repr__(self) -> str:
        return f"QCACluster(id={self.cluster_id}, size={len(self.nodes)})"


@dataclass
class QuenchResult:
    """
    Output of a Quench-only QCA run.

    Attributes
    ----------
    clusters    : List[QCACluster] — K crystallized clusters.
    quench_temp : float            — T_q^RGG used.
    log         : List[str]        — Step-by-step audit trace.
    """
    clusters:    List[QCACluster]
    quench_temp: float
    log:         List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"QuenchResult(K={len(self.clusters)}, "
            f"T_q={self.quench_temp:.6f})"
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _build_distance_matrix(nodes: List[QCANode]) -> List[List[float]]:
    """Step 1 — O(N²) symmetric distance matrix."""
    N = len(nodes)
    D: List[List[float]] = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = _euclidean(nodes[i].coords, nodes[j].coords)
            D[i][j] = d
            D[j][i] = d
    return D


# ---------------------------------------------------------------------------
# Canonical Quench Temperature (RGG, Issue M-QCA-02)
# ---------------------------------------------------------------------------

def quench_temperature(
    N: int,
    K: int,
    A: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """
    T_q^RGG = γ · √( A · ln(N/K) / (π · N) )

    Parameters
    ----------
    N     : int   — Number of nodes.
    K     : int   — Target clusters.
    A     : float — Bounding-box area (default 1.0 for unit square).
    gamma : float — Coupling constant (default 1.0).
    """
    if K <= 0 or N <= K:
        raise ValueError(f"Must satisfy 0 < K < N; got N={N}, K={K}.")
    return gamma * math.sqrt(A * math.log(N / K) / (math.pi * N))


# ---------------------------------------------------------------------------
# Quench-Only QCA
# ---------------------------------------------------------------------------

class QuenchClusterAlgorithm:
    """
    Quench-phase-only QCA engine.

    Produces K crystallization clusters from N nodes using:
      - Step 1: Distance matrix construction (Plasma substrate).
      - Step 2: Farthest-point seed selection + nearest-seed assignment.

    Each cluster is an independent sub-problem ready for the
    Parallel Actualizer + FDSA stage.

    Parameters
    ----------
    K     : int           — Number of target clusters.
    A     : float         — Bounding-box area for T_q formula.
    gamma : float         — Coupling constant for T_q formula.
    seed  : Optional[int] — RNG seed for reproducibility.
    """

    def __init__(
        self,
        K: int = 5,
        A: float = 1.0,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.K     = K
        self.A     = A
        self.gamma = gamma
        self._rng  = random.Random(seed)

    # ------------------------------------------------------------------
    # Step 1: Distance Matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _step1_distance_matrix(
        nodes: List[QCANode],
    ) -> Tuple[List[List[float]], List[str]]:
        log = [
            f"[Step 1 — Distance Matrix] Building {len(nodes)}×{len(nodes)} matrix. "
            f"Complexity: O(N²) = O({len(nodes)**2})."
        ]
        D = _build_distance_matrix(nodes)
        log.append(f"  Matrix complete ({len(nodes)} nodes).")
        return D, log

    # ------------------------------------------------------------------
    # Step 2: Quench Binding
    # ------------------------------------------------------------------

    def _step2_quench(
        self,
        nodes: List[QCANode],
        D: List[List[float]],
        T_q: float,
    ) -> Tuple[List[QCACluster], List[str]]:
        """
        Farthest-point seed selection → nearest-seed node assignment.
        Nodes within T_q of a seed bind to it; remainder bind to nearest seed.
        Complexity: O(N·K) ≪ O(N²).
        """
        N   = len(nodes)
        log = [
            f"[Step 2 — Quench] T_q^RGG = {T_q:.8f}; "
            f"selecting {self.K} seeds via farthest-point sampling."
        ]

        # --- Farthest-point seed selection ---
        seeds: List[int] = [self._rng.randint(0, N - 1)]
        while len(seeds) < self.K:
            farthest = max(
                (i for i in range(N) if i not in seeds),
                key=lambda i: min(D[i][s] for s in seeds),
            )
            seeds.append(farthest)
        log.append(f"  Seed node indices: {seeds}")

        # --- Nearest-seed assignment ---
        assignments: Dict[int, List[int]] = {k: [] for k in range(self.K)}
        for node_idx in range(N):
            nearest_k = min(range(self.K), key=lambda k: D[node_idx][seeds[k]])
            assignments[nearest_k].append(node_idx)

        # --- Build QCACluster objects ---
        clusters: List[QCACluster] = []
        dim      = len(nodes[0].coords)
        n_primes = len(nodes[0].prime_profile)

        for k_idx, member_indices in assignments.items():
            if not member_indices:
                continue
            members  = [nodes[i] for i in member_indices]
            n        = len(members)

            centroid = [
                sum(m.coords[d]        for m in members) / n
                for d in range(dim)
            ]
            prime_agg = [
                sum(m.prime_profile[p] for m in members) / n
                for p in range(n_primes)
            ]

            clusters.append(QCACluster(
                cluster_id=k_idx,
                nodes=members,
                centroid=centroid,
                prime_profile=prime_agg,
            ))
            log.append(
                f"  Cluster {k_idx}: {n} nodes | "
                f"centroid=[{', '.join(f'{c:.3f}' for c in centroid)}] | "
                f"Prime=[{', '.join(f'{p:.3f}' for p in prime_agg)}]"
            )

        log.append(
            f"[Step 2 — Quench] Complete. "
            f"{len(clusters)} clusters formed from {N} nodes."
        )
        return clusters, log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, nodes: List[QCANode]) -> QuenchResult:
        """
        Execute Quench-only QCA.

        Parameters
        ----------
        nodes : List[QCANode] — Input problem nodes (N ≥ K).

        Returns
        -------
        QuenchResult — K clusters + quench temperature + audit log.
        """
        N = len(nodes)
        if N < self.K:
            raise ValueError(
                f"QCA requires N ({N}) ≥ K ({self.K}). "
                "Reduce K or provide more nodes."
            )

        log: List[str] = []
        log.append(
            f"[QCA — Quench Only] N={N}, K={self.K}, A={self.A}, γ={self.gamma}"
        )

        T_q = quench_temperature(N, self.K, self.A, self.gamma)
        log.append(f"[QCA] Canonical T_q^RGG = {T_q:.8f}")

        D, log1 = self._step1_distance_matrix(nodes)
        log.extend(log1)

        clusters, log2 = self._step2_quench(nodes, D, T_q)
        log.extend(log2)

        log.append(
            f"[QCA] Done. {len(clusters)} crystallization clusters "
            f"ready for Parallel Actualizer."
        )
        return QuenchResult(clusters=clusters, quench_temp=T_q, log=log)
