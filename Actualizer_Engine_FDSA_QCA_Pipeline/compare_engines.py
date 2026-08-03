import sys, os, time


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_Core_Engine'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pipeline import ActualizerFDSAQCAPipeline, PipelineConfig
    from benchmark_with_attention import SimulatedAttentionEngine
    from qca_parallel_engine import QCAParallelEngine, QCANode
    import random

    print("=" * 80)
    print("  SIDE-BY-SIDE COMPARISON: pipeline.py vs qca_parallel_engine.py")
    print("  Parameters: K=10 clusters, V=1000 vocab")
    print("=" * 80 + "\n")

    V = 1000
    K = 10
    seed = 42
    N_VALUES = [21, 60, 120, 240]

    # ---------------------------------------------------------------------------
    # Test 1: pipeline.py (ActualizerFDSAQCAPipeline with ThreadPool)
    # ---------------------------------------------------------------------------
    cfg = PipelineConfig(vocab_size=V, execution_mode="parallel", K=K, seed=seed, verbose=False)
    attn = SimulatedAttentionEngine(vocab_size=V, seed=seed)
    pipe = ActualizerFDSAQCAPipeline(config=cfg, attention_engine=attn)

    # Warm-up
    pipe.run(context_ids=[42, 100, 250], step=0)

    pipe_times = []
    for s in range(1, 4):
        res_pipe = pipe.run(context_ids=[42, 100, 250], step=s)
        pipe_times.append(res_pipe.parallel_result.total_time_ms if res_pipe.parallel_result else 0)

    mean_pipe_ms = sum(pipe_times) / len(pipe_times)
    pr_p = res_pipe.parallel_result
    pipe_N = sum(len(cr.node_ids) for cr in pr_p.cluster_results) if pr_p else 0

    print(f"[1] pipeline.py Stage 2 (ThreadPool Backend, N={pipe_N} nodes):")
    print(f"    Total Stage 2 Latency  : {mean_pipe_ms:.2f} ms")
    print(f"    - QCA Crystallization  : {pr_p.qca_time_ms:.2f} ms")
    print(f"    - Parallel Execution   : {pr_p.parallel_time_ms:.2f} ms")
    print(f"    - Synthesis Pass       : {pr_p.synthesis_time_ms:.2f} ms")
    print(f"    - Backend Used         : {pr_p.backend_used}")

    # ---------------------------------------------------------------------------
    # Test 2-3: qca_parallel_engine.py with multiple N values
    # ---------------------------------------------------------------------------
    scaling_results = {}  # {N: {"processes": ms, "auto": ms}}

    for N in N_VALUES:
        rng = random.Random(seed)
        nodes = []
        for i in range(N):
            coords = [rng.uniform(0, 10) for _ in range(5)]
            prime_prof = [rng.uniform(0.1, 0.9) for _ in range(5)]
            nodes.append(QCANode(node_id=i, coords=coords, prime_profile=prime_prof))

        scaling_results[N] = {}

        # Backend: processes
        eng_proc = QCAParallelEngine(K=K, vocab_size=V, backend="processes", seed=seed)
        eng_proc.process_parallel(nodes, verbose=False)  # warm-up

        proc_times = []
        for _ in range(3):
            res_proc = eng_proc.process_parallel(nodes, verbose=False)
            proc_times.append(res_proc.total_time_ms)
        mean_proc_ms = sum(proc_times) / len(proc_times)
        scaling_results[N]["processes"] = mean_proc_ms
        scaling_results[N]["proc_backend"] = res_proc.backend_used

        # Backend: auto/jax
        eng_jax = QCAParallelEngine(K=K, vocab_size=V, backend="auto", seed=seed)
        eng_jax.process_parallel(nodes, verbose=False)  # warm-up

        jax_times = []
        for _ in range(3):
            res_jax = eng_jax.process_parallel(nodes, verbose=False)
            jax_times.append(res_jax.total_time_ms)
        mean_jax_ms = sum(jax_times) / len(jax_times)
        scaling_results[N]["auto"] = mean_jax_ms
        scaling_results[N]["auto_backend"] = res_jax.backend_used

    # Print per-backend detail for last N value (240) for detailed breakdown
    last_N = N_VALUES[-1]
    print(f"\n[2] qca_parallel_engine.py (backend='processes', N={last_N}):")
    print(f"    Total Engine Latency   : {scaling_results[last_N]['processes']:.2f} ms")
    print(f"    - Backend Used         : {scaling_results[last_N]['proc_backend']}")

    print(f"\n[3] qca_parallel_engine.py (backend='auto', N={last_N}):")
    print(f"    Total Engine Latency   : {scaling_results[last_N]['auto']:.2f} ms")
    print(f"    - Backend Used         : {scaling_results[last_N]['auto_backend']}")

    # ---------------------------------------------------------------------------
    # Summary: Multi-N Scaling Comparison Table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SCALING COMPARISON TABLE (K=10 clusters, V=1000 vocab)")
    print("=" * 80)
    print(f"  {'N nodes':<10} | {'pipeline.py (threads)':<24} | {'Engine (processes)':<20} | {'Engine (auto/jax)':<20}")
    print("  " + "-" * 78)

    # pipeline.py runs at a fixed N determined by the pipeline config
    for N in N_VALUES:
        proc_ms = scaling_results[N]["processes"]
        jax_ms = scaling_results[N]["auto"]
        if N == pipe_N:
            pipe_str = f"{mean_pipe_ms:>10.2f} ms"
        else:
            pipe_str = f"{'---':>10}"
        print(
            f"  {N:<10} | {pipe_str:>24} | {proc_ms:>16.2f} ms | {jax_ms:>16.2f} ms"
        )

    # Speedup comparison at matching N
    print("\n  " + "=" * 78)
    print(f"  {'SPEEDUP vs processes (N=' + str(N_VALUES[-1]) + ')':<38} | {'Ratio':<20}")
    print("  " + "-" * 58)
    ref_ms = scaling_results[N_VALUES[-1]]["processes"]
    print(f"  {'pipeline.py (threads, N=' + str(pipe_N) + ')':<38} | {ref_ms / mean_pipe_ms:>16.2f}x")
    for N in N_VALUES:
        jax_ms = scaling_results[N]["auto"]
        print(f"  {'Engine auto/jax (N=' + str(N) + ')':<38} | {ref_ms / jax_ms:>16.2f}x")
    print("=" * 80)


if __name__ == '__main__':
    main()
