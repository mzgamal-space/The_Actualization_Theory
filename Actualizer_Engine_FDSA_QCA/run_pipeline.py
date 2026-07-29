"""
run_pipeline.py — Quick-Start Runner for Actualizer_Engine_FDSA_QCA
=====================================================================
Pipeline Name : Actualizer_Engine_FDSA_QCA
Author        : Mohamed Gamal Eldin Abdelaziz Noureldin
Framework     : Consciousness and Prime Base Intelligence (CKT V3_U1)

Run this file directly for an interactive demo of the pipeline:
    python run_pipeline.py
    python run_pipeline.py --mode parallel --K 4
    python run_pipeline.py --mode benchmark
    python run_pipeline.py --mode sequence --steps 10
"""

from __future__ import annotations

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "02_Core_Engine"))
for _p in (_THIS_DIR, _CORE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import (
    ActualizerFDSAQCAPipeline,
    PipelineConfig,
    create_sequential_pipeline,
    create_parallel_pipeline,
)
from benchmark_with_attention import (
    run_attention_benchmark,
    run_full_benchmark,
    verify_canonical_ordering,
    SimulatedAttentionEngine,
)


BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║     Actualizer_Engine_FDSA_QCA  —  Pipeline Runner v1.0.0          ║
║     Framework: Consciousness and Prime Base Intelligence (CKT V3_U1)║
║     Author:    Mohamed Gamal Eldin Abdelaziz Noureldin              ║
║     ORCID:     0009-0006-3991-1153                                  ║
║     DOI:       10.5281/zenodo.21420098                              ║
╠══════════════════════════════════════════════════════════════════════╣
║     Pipeline: FDSA Pruner → QCA Parallel Engine → Actualizer Snap  ║
║     Ordering: FDSA FIRST (Theorem 2.1 — canonical ordering)        ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def run_demo_sequential(vocab_size: int = 500, verbose: bool = True) -> None:
    """Demo: single-step sequential pipeline."""
    print("\n[MODE: Sequential Demo]")
    pipeline = create_sequential_pipeline(vocab_size=vocab_size, verbose=verbose)
    context = [42, 17, 305, 88]
    result = pipeline.run(context_ids=context, step=0)
    print(f"\n  Result: {result.summary()}")
    print(f"  FDSA Stage : {result.fdsa_result}")
    print(f"  ACT Stage  : {result.act_result}")


def run_demo_parallel(vocab_size: int = 500, K: int = 4, verbose: bool = True) -> None:
    """Demo: single-step parallel pipeline with QCA clustering."""
    print(f"\n[MODE: Parallel Demo (K={K})]")
    pipeline = create_parallel_pipeline(vocab_size=vocab_size, K=K, verbose=verbose)
    context = [42, 17, 305, 88]
    result = pipeline.run(context_ids=context, step=0)
    print(f"\n  Result: {result.summary()}")
    if result.parallel_result:
        pr = result.parallel_result
        print(f"  QCA clusters  : {len(pr.cluster_results)}")
        print(f"  QCA time      : {pr.qca_time_ms:.2f}ms")
        print(f"  Parallel time : {pr.parallel_time_ms:.2f}ms")
        print(f"  Backend       : {pr.backend_used}")


def run_demo_sequence(
    vocab_size: int = 500, max_tokens: int = 10, mode: str = "sequential"
) -> None:
    """Demo: autoregressive sequence generation."""
    print(f"\n[MODE: Sequence Generation ({mode}, {max_tokens} tokens)]")
    if mode == "parallel":
        pipeline = create_parallel_pipeline(vocab_size=vocab_size, K=4, verbose=False)
    else:
        pipeline = create_sequential_pipeline(vocab_size=vocab_size, verbose=False)

    prompt = [42, 17, 305, 88, 21]
    print(f"  Prompt context: {prompt}")
    generated, step_results = pipeline.generate_sequence(
        prompt_ids=prompt, max_new_tokens=max_tokens
    )
    print(f"  Generated IDs : {generated}")
    print(f"\n  Step-by-step:")
    for i, r in enumerate(step_results):
        status = "✓" if r.is_actualized else "✗"
        print(
            f"    Step {i:>2}: token={r.final_token:>6} | {status} | "
            f"ν={r.global_valuation:.4f} | "
            f"pruned={r.fdsa_result.pruning_ratio:.1%} | "
            f"{r.total_time_ms:.2f}ms"
        )


def run_demo_benchmark(quick: bool = True) -> None:
    """Demo: full benchmark suite."""
    print("\n[MODE: Benchmark Suite]")
    run_full_benchmark(verbose=True, quick=quick)


def run_demo_ordering() -> None:
    """Demo: verify Theorem 2.1 (canonical ordering)."""
    print("\n[MODE: Theorem 2.1 Ordering Verification]")
    verify_canonical_ordering(vocab_size=300, T_steps=5, verbose=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Actualizer_Engine_FDSA_QCA — Quick Start Runner"
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "parallel", "sequence", "benchmark", "ordering", "all"],
        help=(
            "sequential  : Single-step FDSA → Actualizer demo\n"
            "parallel    : Single-step FDSA → QCA → Actualizer demo\n"
            "sequence    : Autoregressive sequence generation demo\n"
            "benchmark   : Full benchmark suite\n"
            "ordering    : Theorem 2.1 ordering verification\n"
            "all         : Run all demos"
        ),
    )
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--K", type=int, default=4, help="QCA cluster count")
    parser.add_argument("--steps", type=int, default=8, help="Sequence steps / benchmark T")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--quick", action="store_true",
                        help="Quick benchmark (reduced parameters)")

    args = parser.parse_args()
    verbose = not args.quiet
    V = args.vocab_size
    K = args.K
    T = args.steps

    if args.mode == "sequential" or args.mode == "all":
        run_demo_sequential(vocab_size=V, verbose=verbose)

    if args.mode == "parallel" or args.mode == "all":
        run_demo_parallel(vocab_size=V, K=K, verbose=verbose)

    if args.mode == "sequence" or args.mode == "all":
        run_demo_sequence(vocab_size=V, max_tokens=T, mode="sequential")

    if args.mode == "benchmark" or args.mode == "all":
        run_demo_benchmark(quick=args.quick or args.mode == "all")

    if args.mode == "ordering" or args.mode == "all":
        run_demo_ordering()

    print("\n[DONE] Actualizer_Engine_FDSA_QCA pipeline runner complete.")
