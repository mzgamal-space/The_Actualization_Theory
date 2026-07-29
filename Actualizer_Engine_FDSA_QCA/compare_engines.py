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
    print("  Parameters: K=10 clusters, V=1000 vocab, N=120 nodes")
    print("=" * 80 + "\n")

    V = 1000
    K = 10
    seed = 42

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

    print("[1] pipeline.py Stage 2 (ThreadPool Backend):")
    print(f"    Total Stage 2 Latency  : {mean_pipe_ms:.2f} ms")
    print(f"    - QCA Crystallization  : {pr_p.qca_time_ms:.2f} ms")
    print(f"    - Parallel Execution   : {pr_p.parallel_time_ms:.2f} ms")
    print(f"    - Synthesis Pass       : {pr_p.synthesis_time_ms:.2f} ms")
    print(f"    - Backend Used         : {pr_p.backend_used}")

    # ---------------------------------------------------------------------------
    # Test 2: qca_parallel_engine.py (backend='processes')
    # ---------------------------------------------------------------------------
    rng = random.Random(seed)
    nodes_21 = []
    for i in range(120):
        coords = [rng.uniform(0, 10) for _ in range(5)]
        prime_prof = [rng.uniform(0.1, 0.9) for _ in range(5)]
        nodes_21.append(QCANode(node_id=i, coords=coords, prime_profile=prime_prof))

    eng_proc = QCAParallelEngine(K=K, vocab_size=V, backend="processes", seed=seed)
    eng_proc.process_parallel(nodes_21, verbose=False) # warm-up

    proc_times = []
    for _ in range(3):
        res_proc = eng_proc.process_parallel(nodes_21, verbose=False)
        proc_times.append(res_proc.total_time_ms)
    mean_proc_ms = sum(proc_times) / len(proc_times)

    print("\n[2] qca_parallel_engine.py (backend='processes'):")
    print(f"    Total Engine Latency   : {mean_proc_ms:.2f} ms")
    print(f"    - QCA Crystallization  : {res_proc.qca_time_ms:.2f} ms")
    print(f"    - Parallel Execution   : {res_proc.parallel_time_ms:.2f} ms")
    print(f"    - Synthesis Pass       : {res_proc.synthesis_time_ms:.2f} ms")
    print(f"    - Backend Used         : {res_proc.backend_used}")

    # ---------------------------------------------------------------------------
    # Test 3: qca_parallel_engine.py (backend='auto/jax')
    # ---------------------------------------------------------------------------
    eng_jax = QCAParallelEngine(K=K, vocab_size=V, backend="auto", seed=seed)
    eng_jax.process_parallel(nodes_21, verbose=False) # warm-up

    jax_times = []
    for _ in range(3):
        res_jax = eng_jax.process_parallel(nodes_21, verbose=False)
        jax_times.append(res_jax.total_time_ms)
    mean_jax_ms = sum(jax_times) / len(jax_times)

    print("\n[3] qca_parallel_engine.py (backend='auto/jax'):")
    print(f"    Total Engine Latency   : {mean_jax_ms:.2f} ms")
    print(f"    - QCA Crystallization  : {res_jax.qca_time_ms:.2f} ms")
    print(f"    - Parallel Execution   : {res_jax.parallel_time_ms:.2f} ms")
    print(f"    - Synthesis Pass       : {res_jax.synthesis_time_ms:.2f} ms")
    print(f"    - Backend Used         : {res_jax.backend_used}")

    print("\n" + "=" * 80)
    print("  SUMMARY COMPARISON TABLE (K=10 clusters, V=1000 vocab, N=21 nodes)")
    print("=" * 80)
    print(f"  {'Engine / Implementation':<38} | {'Latency (ms)':<14} | {'Speedup vs Process':<20}")
    print("  " + "-" * 76)
    print(f"  {'pipeline.py (ThreadPool)':<38} | {mean_pipe_ms:>10.2f} ms   | {mean_proc_ms / mean_pipe_ms:>16.2f}x")
    print(f"  {'qca_parallel_engine.py (processes)':<38} | {mean_proc_ms:>10.2f} ms   | {1.00:>16.2f}x")
    print(f"  {'qca_parallel_engine.py (auto/jax)':<38} | {mean_jax_ms:>10.2f} ms   | {mean_proc_ms / mean_jax_ms:>16.2f}x")
    print("=" * 80)

if __name__ == '__main__':
    main()
