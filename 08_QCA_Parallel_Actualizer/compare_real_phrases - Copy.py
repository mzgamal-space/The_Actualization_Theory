"""
compare_real_phrases.py — Comparison between Sequential and Parallel Actualizer using Real Phrases
====================================================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)
Module : Final_Output/08_QCA_Parallel_Actualizer

This script evaluates the QCA Parallel Actualizer against a standard Sequential Actualizer
using a vocabulary of real phrases derived from CKT theory.

Both engines respond to a user PROMPT. The prompt is:
  1. Embedded into the vocabulary — matching phrases seed the causal chain.
  2. Translated into target tokens — phrases most relevant to the question
     become the steering goal for the Banach contraction loop.

Fixes applied vs. previous version:
  - PromptActualizer: adds a no-repeat window mask after each Banach step
    so the same phrase is never selected twice in a row.
  - Prompt embedding now deduplicates seed tokens and orders them by
    topic relevance.
  - Target tokens are derived from prompt keywords — not just the last token.
  - Theoretical speedup is shown alongside wall-clock timing.

Run:
    $env:PYTHONIOENCODING = "utf-8"; python compare_real_phrases.py
"""

import time
import math
from typing import List, Dict, Tuple, Set, Optional
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_PKG = os.path.join(_HERE, "..", "..", "Code", "ckt_actualizer_engine", "src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from qca import QuenchClusterAlgorithm, QCANode, QCACluster
from parallel_actualizer import ParallelActualizer, ClusterActualizer
from global_actualizer import GlobalActualizer

# ===========================================================================
# 1. Vocabulary — 24 real CKT phrases across 4 topics
# ===========================================================================

PHRASES = [
    # Topic 0: Ontology / Prime Base  (tokens 0–5)
    "Knowledge is our best justified, ordered map of reality.",           # 0
    "Wisdom is the lossless moral compression of knowledge.",             # 1
    "Order prevents chaos; Justice prevents bias.",                       # 2
    "Complexity is governed by the weakest Prime.",                       # 3
    "Energy and Information unfold through Time.",                        # 4
    "Consciousness requires self-referential anchoring.",                 # 5

    # Topic 1: CKT Dynamics  (tokens 6–11)
    "CAKI measures the ratio of accumulated knowledge to complexity.",    # 6
    "Crystallization requires a CAKI threshold to be met.",              # 7
    "FDSA anchors local problems to isomorphic global domains.",          # 8
    "The Banach contraction theorem guarantees convergence.",             # 9
    "The vacuum brake prevents infinite divergence of inference.",        # 10
    "Vectorized pruning reduces the search space logarithmically.",       # 11

    # Topic 2: QCA & Partitioning  (tokens 12–17)
    "QCA partitions the Plasma state into K clusters.",                   # 12
    "The Quench temperature is derived from Random Geometric Graphs.",    # 13
    "Parallel actualization accelerates inference significantly.",        # 14
    "Each cluster forms an independent problem space.",                   # 15
    "MCE sub-objects are injected into the global FDSA library.",        # 16
    "Global Actualizer synthesizes sub-objects into a final MCE.",       # 17

    # Topic 3: Ethics & Justice  (tokens 18–23)
    "Justice ensures fair allocation of computational resources.",        # 18
    "Mercy prevents catastrophic pruning of valid trajectories.",        # 19
    "A system without Order cannot sustain Knowledge.",                  # 20
    "Power is the capacity to actualize potential into reality.",        # 21
    "Truth is an asymptotic goal approached via continuous pruning.",     # 22
    "Ethics can be quantified as structural harmony.",                   # 23
]

VOCAB_SIZE = len(PHRASES)

# Keyword → (seed_token_id, [target_token_ids])
# seed: the phrase that most directly echoes the keyword in the prompt.
# target: the set of phrases that constitute a good *answer* to that keyword.
KEYWORD_MAP: Dict[str, Tuple[int, List[int]]] = {
    # prompt keyword   seed  targets (answer phrases)
    "knowledge":      (0,   [1, 2, 6]),
    "wisdom":         (1,   [0, 2]),
    "order":          (2,   [0, 20]),
    "justice":        (18,  [2, 20]),
    "prime":          (3,   [0, 2]),
    "caki":           (6,   [7, 9]),
    "crystallization":(7,   [6, 16]),
    "fdsa":           (8,   [9, 11, 16]),
    "banach":         (9,   [8, 10]),
    "convergence":    (9,   [10, 11]),
    "qca":            (12,  [13, 14, 15, 16, 17]),
    "quench":         (13,  [12, 14, 15]),
    "parallel":       (14,  [12, 15, 16, 17]),
    "cluster":        (15,  [12, 13, 14]),
    "inference":      (10,  [9, 11, 14]),
    "global":         (17,  [16, 8]),
    "mce":            (16,  [17, 7]),
    "mercy":          (19,  [18, 20]),
    "power":          (21,  [22, 0]),
    "truth":          (22,  [0, 1]),
    "ethics":         (23,  [18, 20]),
}


def get_topic(token_id: int) -> int:
    return token_id // 6


def get_topic_coords(topic_id: int) -> List[float]:
    return {0: [0.1, 0.1], 1: [0.1, 0.9], 2: [0.9, 0.1], 3: [0.9, 0.9]}[topic_id]


def build_nodes() -> List[QCANode]:
    import random
    rng = random.Random(42)
    nodes = []
    for i, phrase in enumerate(PHRASES):
        topic       = get_topic(i)
        base        = get_topic_coords(topic)
        coords      = [base[0] + rng.uniform(-0.05, 0.05),
                       base[1] + rng.uniform(-0.05, 0.05)]
        prime       = [0.5] * 5
        prime[topic] = 0.95
        nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime,
                             metadata={"phrase": phrase}))
    return nodes


def build_semantic_cwf() -> Dict[Tuple[int, int], float]:
    """Cross-topic transitions are penalised — same-topic chaining is free."""
    cwf: Dict[Tuple[int, int], float] = {}
    for i in range(VOCAB_SIZE):
        for j in range(VOCAB_SIZE):
            if get_topic(i) != get_topic(j):
                cwf[(i, j)] = 0.20
    return cwf


def embed_prompt(prompt: str) -> Tuple[List[int], Set[int]]:
    """
    Parse a natural-language prompt and return:
      seed_tokens   — token IDs that appear in the prompt (ordered, deduplicated).
      target_tokens — token IDs that constitute a good answer.
    """
    prompt_lower = prompt.lower()
    seed_ordered  : List[int] = []
    seen_seeds    : Set[int]  = set()
    target_set    : Set[int]  = set()

    for keyword, (seed_id, target_ids) in KEYWORD_MAP.items():
        if keyword in prompt_lower:
            if seed_id not in seen_seeds:
                seed_ordered.append(seed_id)
                seen_seeds.add(seed_id)
            for t in target_ids:
                target_set.add(t)

    # Fallback
    if not seed_ordered:
        seed_ordered = [0]
    if not target_set:
        target_set = {1, 2, 3}

    return seed_ordered, target_set


def decode_chain(chain: List[int], seed_len: int) -> str:
    """
    Decode a token chain to natural language.
    Skips the seed portion and deduplicates consecutive identical phrases.
    """
    response_tokens = chain[seed_len:]
    sentences: List[str] = []
    prev: Optional[int] = None
    for token in response_tokens:
        if token == prev:          # skip exact consecutive repeat
            continue
        if 0 <= token < VOCAB_SIZE:
            sentences.append(PHRASES[token])
        prev = token
    return " ".join(sentences) if sentences else "[no response generated]"


# ===========================================================================
# 2. PromptActualizer — extends ClusterActualizer with no-repeat masking
# ===========================================================================

class PromptActualizer(ClusterActualizer):
    """
    ClusterActualizer + no-repeat window mask.

    After the Banach loop converges to a probability vector U, any token
    that appears in the last `no_repeat_window` positions of the history
    is masked out before the argmax selection.  This prevents the loop from
    collapsing to a single dominant phrase.
    """

    def __init__(self, *args, no_repeat_window: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_repeat_window = no_repeat_window

    def _steer_token(
        self,
        pruned_logits: List[float],
        history:       List[int],
        target_tokens: Set[int],
        k_step:        float,
    ) -> Tuple[int, List[float], int]:
        token, U, iters = super()._steer_token(
            pruned_logits, history, target_tokens, k_step
        )
        # --- No-repeat mask ---
        recent = set(history[-self.no_repeat_window:])
        if token in recent:
            ranked = sorted(range(self.V), key=lambda v: -U[v])
            for candidate in ranked:
                if candidate not in recent and U[candidate] > 0.0:
                    token = candidate
                    break
        return token, U, iters


# ===========================================================================
# 3. PromptParallelActualizer — swaps in PromptActualizer per cluster
# ===========================================================================

class PromptParallelActualizer(ParallelActualizer):
    """Uses PromptActualizer (with no-repeat masking) instead of the base ClusterActualizer."""

    def run(
        self,
        clusters:       List[QCACluster],
        initial_chain:  List[int],
        target_tokens:  Set[int],
        diept_a:        List[float],
        diept_b:        List[float],
        delta_c_r:      float = -0.5,
        n_steps:        int   = 3,
    ):
        from parallel_actualizer import ClusterActualizerResult
        results = []
        for cluster in clusters:
            act    = PromptActualizer(**self._kwargs)
            result = act.process_cluster(
                cluster=cluster,
                initial_chain=list(initial_chain),
                target_tokens=target_tokens,
                diept_a=diept_a,
                diept_b=diept_b,
                delta_c_r=delta_c_r,
                n_steps=n_steps,
            )
            results.append(result)
        return results


# ===========================================================================
# 4. ANSI helpers
# ===========================================================================
BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def bar(title: str) -> None:
    line = "─" * 70
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}")


# ===========================================================================
# 5. Run helpers
# ===========================================================================

def run_sequential(
    seed_tokens:   List[int],
    target_tokens: Set[int],
    cwf:           Dict[Tuple[int, int], float],
    nodes:         List[QCANode],
) -> Tuple[float, float, str]:

    bar("Sequential Actualizer  (single cluster, full vocab)")

    giant = QCACluster(
        cluster_id=0,
        nodes=nodes,
        centroid=[0.5, 0.5],
        prime_profile=[0.5] * 5,
    )

    start = time.perf_counter()
    act   = PromptActualizer(
        vocab_size=VOCAB_SIZE,
        cwf_penalty_matrix=cwf,
        max_iterations=100,
        no_repeat_window=6,
    )
    result = act.process_cluster(
        cluster=giant,
        initial_chain=list(seed_tokens),
        target_tokens=target_tokens,
        diept_a=[0.9] * 5,
        diept_b=[0.1] * 5,
        n_steps=5,
    )
    elapsed = time.perf_counter() - start

    thought = decode_chain(result.causal_chain, len(seed_tokens))
    print(f"{YELLOW}  Time:  {RESET}{elapsed:.4f} s")
    print(f"{YELLOW}  CAKI:  {RESET}{result.caki:.4f}  (anchor: '{result.anchor_domain}')")
    print(f"{YELLOW}  Response:{RESET}")
    print(f"  {GREEN}{thought}{RESET}")
    return elapsed, result.caki, thought


def run_parallel(
    seed_tokens:   List[int],
    target_tokens: Set[int],
    cwf:           Dict[Tuple[int, int], float],
    nodes:         List[QCANode],
    K:             int,
) -> Tuple[float, float, str]:

    bar("QCA Parallel Actualizer Pipeline  (K clusters)")

    start = time.perf_counter()

    # Stage 1 — Quench
    print(f"  {DIM}[Stage 1]{RESET} QCA Quench — partitioning {len(nodes)} phrases into K={K} clusters ...")
    qca      = QuenchClusterAlgorithm(K=K, seed=42)
    qca_res  = qca.run(nodes)
    for c in qca_res.clusters:
        member_phrases = [PHRASES[n.node_id] for n in c.nodes]
        print(f"    Cluster {c.cluster_id}: {[p[:40]+'...' for p in member_phrases]}")

    # Stage 2 — Parallel Actualizer per cluster
    print(f"\n  {DIM}[Stage 2]{RESET} Parallel Actualizer + FDSA on each cluster ...")
    parallel = PromptParallelActualizer(
        vocab_size=VOCAB_SIZE,
        cwf_penalty_matrix=cwf,
        max_iterations=100,
        no_repeat_window=6,
        caki_threshold=0.30,
    )
    cluster_results = parallel.run(
        clusters=qca_res.clusters,
        initial_chain=list(seed_tokens),
        target_tokens=target_tokens,
        diept_a=[0.9] * 5,
        diept_b=[0.1] * 5,
        n_steps=3,
    )
    for r in cluster_results:
        sub = decode_chain(r.causal_chain, len(seed_tokens))
        cryst = f"{GREEN}crystallized{RESET}" if r.is_crystallized else f"{RED}not crystallized{RESET}"
        print(f"    Cluster {r.cluster_id}: CAKI={r.caki:.4f} ({cryst}) | {sub}")

    # Stage 3 — Global Actualizer on MCE results
    print(f"\n  {DIM}[Stage 3]{RESET} Global Actualizer — synthesizing {sum(1 for r in cluster_results if r.mce) } cluster MCEs ...")
    global_act = GlobalActualizer(
        vocab_size=VOCAB_SIZE,
        cwf_penalty_matrix=cwf,
        max_iterations=100,
        n_steps=3,
        caki_threshold=0.30,
    )
    solution = global_act.run(
        cluster_results=cluster_results,
        initial_chain=list(seed_tokens),
        target_tokens=target_tokens,
        diept_a=[0.9] * 5,
        diept_b=[0.1] * 5,
    )

    elapsed = time.perf_counter() - start
    thought = decode_chain(solution.global_chain, len(seed_tokens))
    print(f"\n{YELLOW}  Time:  {RESET}{elapsed:.4f} s")
    print(f"{YELLOW}  CAKI:  {RESET}{solution.global_caki:.4f}  (anchor: '{solution.anchor_domain}')")
    print(f"{YELLOW}  Response:{RESET}")
    print(f"  {GREEN}{thought}{RESET}")
    return elapsed, solution.global_caki, thought


# ===========================================================================
# 6. Main
# ===========================================================================

PROMPT = "How does QCA use parallel clustering to improve inference?"
K      = 4   # clusters for QCA

if __name__ == "__main__":
    bar(f'Prompt  →  "{PROMPT}"')

    seed_tokens, target_tokens = embed_prompt(PROMPT)

    print(f"\n  Seed phrases (prompt embedding):")
    for t in seed_tokens:
        print(f"    [{t:02d}] {PHRASES[t]}")

    print(f"\n  Target phrases (answer goal):")
    for t in sorted(target_tokens):
        print(f"    [{t:02d}] {PHRASES[t]}")

    nodes = build_nodes()
    cwf   = build_semantic_cwf()

    seq_t, seq_caki, seq_resp = run_sequential(seed_tokens, target_tokens, cwf, nodes)
    par_t, par_caki, par_resp = run_parallel(seed_tokens, target_tokens, cwf, nodes, K)

    # --- Comparison table ---
    bar("Comparison Summary")
    N = VOCAB_SIZE
    theoretical_speedup = K  # Theorem 2: O(N²) → O(N²/K)

    speedup_wall = seq_t / par_t if par_t > 0 else 0
    winner_caki  = "Parallel" if par_caki >= seq_caki else "Sequential"

    print(f"""
  Prompt : "{PROMPT}"

  ┌───────────────────────┬─────────────────┬─────────────────┐
  │ Metric                │ Sequential      │ QCA Parallel    │
  ├───────────────────────┼─────────────────┼─────────────────┤
  │ Wall-clock time       │ {seq_t:.4f} s       │ {par_t:.4f} s       │
  │ Wall-clock speedup    │   1.00x         │   {speedup_wall:.2f}x          │
  │ Theorem-2 speedup     │   1.00x         │   {theoretical_speedup:.0f}x (O(N²/K))   │
  │ CAKI score            │ {seq_caki:.4f}          │ {par_caki:.4f}          │
  │ CAKI winner           │ {"← winner" if winner_caki == "Sequential" else "         "}      │ {"← winner" if winner_caki == "Parallel" else ""}         │
  └───────────────────────┴─────────────────┴─────────────────┘

  Note: at N={N} phrases the QCA overhead (distance matrix + K cluster
  instantiation) dominates wall-clock time. The Theorem-2 factor-{K}
  speedup is the asymptotic gain and becomes dominant at N > ~1 000.
""")
