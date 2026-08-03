import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_Core_Engine'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import create_parallel_pipeline, create_sequential_pipeline

print("Testing speedup at increasing vocabulary sizes (K=4)...")
print("    V  |   K |  Sequential |   Parallel |  Speedup")
print("  " + "-" * 52)

for V in [500, 1000, 2000, 4000]:
    K = 10
    context = [42, 17, 305, 88]

    seq = create_sequential_pipeline(vocab_size=V, verbose=False)
    par = create_parallel_pipeline(vocab_size=V, K=K, verbose=False, seed=42)

    # Warm up
    seq.run(context_ids=context, step=0)
    par.run(context_ids=context, step=0)

    # Time 3 runs each
    st = [seq.run(context_ids=context, step=i).total_time_ms for i in range(1, 4)]
    pt = [par.run(context_ids=context, step=i).total_time_ms for i in range(1, 4)]

    s_ms = sum(st) / len(st)
    p_ms = sum(pt) / len(pt)
    spd  = s_ms / p_ms if p_ms > 0 else 1.0
    faster = "FASTER" if spd > 1.0 else "slower"
    print(f"  {V:>5} | {K:>3} | {s_ms:>10.1f}ms | {p_ms:>9.1f}ms | {spd:>6.2f}x  {faster}")
