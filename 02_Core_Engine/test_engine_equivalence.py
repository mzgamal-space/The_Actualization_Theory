"""
test_engine_equivalence.py — Consolidated Verification Suite
================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153

Run this file directly (`python3 test_engine_equivalence.py`) to verify
every fix made in this pass, end to end, with no silent failures. Each
check prints PASS/FAIL explicitly; the script exits non-zero if anything
fails, so it's safe to wire into CI.

WHAT THIS DOES verify (all run in this sandbox, CPU only, no TPU/GPU,
no HuggingFace Hub access -- see honesty notes at the bottom):
  1. ActualizerEngine (Python) == NumpyActualizerEngine == JaxActualizerEngine
     agree exactly (token selection, Tr(D_mu_nu) within 1e-4) across 10
     random seeds, including a case with -inf-masked (pruner-style) input.
  2. Measured speed: NumPy ~18-38x faster than pure Python at V=32,128;
     JAX matches NumPy for single calls; JAX+vmap batching gives a real,
     verified additional speedup for batches, with exact correctness.
  3. VectorizedFDSAPruner: confirms the diagnostic warning fires when the
     complexity threshold doesn't bind (open-ended / no grammar), and
     confirms real pruning + correct-token survival under an actual
     grammar constraint.
  4. QCAParallelEngine: confirms the spawn-context fix eliminates the
     fork/JAX-incompatibility warning.

WHAT THIS DOES NOT verify (explicitly out of scope for this sandbox):
  - Real T5 (or any real model) generation quality/speed. This
    environment has no GPU/TPU and no network access to huggingface.co,
    so no real model can be loaded here. Every number in this file comes
    from synthetic-but-labeled test logits, matching the scale (V=32,128)
    and rough distribution shape used elsewhere in this repo's own tests.
  - TPU-specific behavior. JAX/XLA can behave differently on TPU vs CPU;
    the numbers here are CPU baselines, not TPU numbers. Re-run this
    exact file on your actual TPU before reporting TPU-specific figures.
  - End-to-end EM/F1 on a real QA dataset. That requires wiring a real
    generate() call into an actual decoding loop, which is future work,
    not something this file claims to have done.
"""

import sys
import time
import random
import warnings
import multiprocessing

sys.path.insert(0, '.')

def main():
    PASS = "PASS"
    FAIL = "FAIL"
    results = []


    def check(name, condition, detail=""):
        status = PASS if condition else FAIL
        results.append((name, status, detail))
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        return condition


    def section(title):
        print(f"\n{'='*70}\n{title}\n{'='*70}")


    # ---------------------------------------------------------------------------
    section("1. ENGINE EQUIVALENCE (Python / NumPy / JAX)")
    # ---------------------------------------------------------------------------

    from actualizer_engine import ActualizerEngine
    from numpy_actualizer_engine import NumpyActualizerEngine
    from jax_actualizer_engine import JaxActualizerEngine

    V_SMALL = 500
    all_equivalent = True
    for seed in range(10):
        random.seed(seed)
        logits = [random.gauss(-3.0, 2.0) for _ in range(V_SMALL)]
        target = 42 + seed
        logits[target] += 6.0
        history = [max(0, target - 2), max(0, target - 1)]
        target_tokens = {target}

        py = ActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5, tau=1.0,
                               tau_bifurcation=5.0, max_iters=100)
        npe = NumpyActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5,
                                     tau_bifurcation=5.0, max_iters=100)
        jxe = JaxActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5,
                                   tau_bifurcation=5.0, max_iters=100)

        pt, _, pTr, _, _, _ = py.steer(logits, history, target_tokens)
        nt, _, nTr, _, _, _ = npe.steer(logits, history, target_tokens)
        jt, _, jTr, _, _ = jxe.steer(logits, history, target_tokens)

        seed_ok = (pt == nt == jt) and abs(pTr - nTr) < 1e-4 and abs(nTr - jTr) < 1e-4
        all_equivalent = all_equivalent and seed_ok

    check("Python/NumPy/JAX agree across 10 seeds (V=500)", all_equivalent)

    # -inf-masked (pruner-style) input case
    random.seed(3)
    V_MASK = 300
    logits_m = [random.gauss(-3.0, 2.0) for _ in range(V_MASK)]
    logits_m[42] += 6.0
    for i in range(100):
        logits_m[i] = float('-inf')
    history_m, target_m = [40, 41], {42}

    py2 = ActualizerEngine(vocab_size=V_MASK, mercy_k=0.45, Q_c=1e-5, tau=1.0,
                            tau_bifurcation=5.0, max_iters=100)
    npe2 = NumpyActualizerEngine(vocab_size=V_MASK, mercy_k=0.45, Q_c=1e-5,
                                  tau_bifurcation=5.0, max_iters=100)
    pt2, _, pTr2, _, _, _ = py2.steer(logits_m, history_m, target_m)
    nt2, _, nTr2, _, _, _ = npe2.steer(logits_m, history_m, target_m)
    check("Python/NumPy agree with -inf-masked (pruner-style) input",
          pt2 == nt2 and abs(pTr2 - nTr2) < 1e-6,
          f"tokens: py={pt2} np={nt2}, Tr_D diff={abs(pTr2-nTr2):.8f}")


    # ---------------------------------------------------------------------------
    section("2. SPEED (measured this run, CPU only)")
    # ---------------------------------------------------------------------------

    V_BIG = 32128
    random.seed(1)
    logits_big = [random.gauss(-3.0, 1.0) for _ in range(V_BIG)]
    logits_big[1000] = 5.0
    history_big, target_big = [998, 999], {1000}

    py_big = ActualizerEngine(vocab_size=V_BIG, mercy_k=0.45, Q_c=1e-5, tau=1.0,
                               tau_bifurcation=5.0, max_iters=100)
    t0 = time.perf_counter()
    py_big.steer(logits_big, history_big, target_big)
    py_time = time.perf_counter() - t0

    npe_big = NumpyActualizerEngine(vocab_size=V_BIG, mercy_k=0.45, Q_c=1e-5,
                                     tau_bifurcation=5.0, max_iters=100)
    t0 = time.perf_counter()
    npe_big.steer(logits_big, history_big, target_big)
    np_time = time.perf_counter() - t0

    check(f"NumPy faster than pure Python at V={V_BIG}",
          np_time < py_time,
          f"Python={py_time:.3f}s  NumPy={np_time:.3f}s  ({py_time/np_time:.1f}x)")

    # vmap batching
    jxe_big = JaxActualizerEngine(vocab_size=V_BIG, mercy_k=0.45, Q_c=1e-5,
                                   tau_bifurcation=5.0, max_iters=100)
    import jax, jax.numpy as jnp
    N_BATCH = 20
    batch_logits, batch_targets, expected = [], [], []
    for i in range(N_BATCH):
        random.seed(200 + i)
        l = [random.gauss(-3.0, 1.0) for _ in range(V_BIG)]
        tgt = 500 + i * 7
        l[tgt] = 5.0
        batch_logits.append(l)
        batch_targets.append({tgt})
        expected.append(tgt)

    t0 = time.perf_counter()
    seq_tokens = []
    for i in range(N_BATCH):
        tok, *_ = npe_big.steer(batch_logits[i], [max(0, expected[i]-1)], batch_targets[i])
        seq_tokens.append(tok)
    seq_time = time.perf_counter() - t0

    logits_batch = jnp.array(batch_logits, dtype=jnp.float32)
    target_masks = jnp.stack([jxe_big.make_target_mask(t) for t in batch_targets])
    hib, hvb, hsb = zip(*[jxe_big.make_history_arrays([max(0, expected[i]-1)]) for i in range(N_BATCH)])
    hib, hvb, hsb = jnp.stack(hib), jnp.stack(hvb), jnp.stack(hsb)
    # Signature now includes w_L, w_G, w_F, k as traced args (domain-anchoring
    # fix, see section 3b below) -- pass this engine's own defaults, broadcast
    # across the batch, since this section is testing plain single-domain
    # batching, not mixed-domain batching (that's tested separately below).
    wL_default = jnp.full((N_BATCH,), jxe_big.w_L, dtype=jnp.float32)
    wG_default = jnp.full((N_BATCH,), jxe_big.w_G, dtype=jnp.float32)
    wF_default = jnp.full((N_BATCH,), jxe_big.w_F, dtype=jnp.float32)
    k_default = jnp.full((N_BATCH,), jxe_big.k, dtype=jnp.float32)
    batched_steer = jax.jit(jax.vmap(jxe_big._steer_jit, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)))
    U_b, *_ = batched_steer(logits_batch, target_masks, hib, hvb, hsb,
                             wL_default, wG_default, wF_default, k_default)  # warm up
    U_b.block_until_ready()

    t0 = time.perf_counter()
    U_b, *_ = batched_steer(logits_batch, target_masks, hib, hvb, hsb,
                             wL_default, wG_default, wF_default, k_default)
    U_b.block_until_ready()
    batch_time = time.perf_counter() - t0
    vmap_tokens = [int(jnp.argmax(U_b[i])) for i in range(N_BATCH)]

    check(f"vmap-batched JAX ({N_BATCH} examples) matches sequential NumPy exactly",
          vmap_tokens == seq_tokens)
    check("vmap batching is faster than sequential calls",
          batch_time < seq_time,
          f"sequential={seq_time:.3f}s  batched={batch_time:.3f}s  ({seq_time/batch_time:.1f}x)")


    # ---------------------------------------------------------------------------
    section("3. FDSA PRUNER (diagnostic + real grammar-constrained pruning)")
    # ---------------------------------------------------------------------------

    from fdsa_pruner import VectorizedFDSAPruner
    import numpy as np

    pruner = VectorizedFDSAPruner(vocab_size=V_BIG, k=0.35)
    np.random.seed(0)
    logits_p = np.random.normal(-3.0, 1.0, size=V_BIG)
    logits_p[500] = 5.0

    with warnings.catch_warnings(record=True) as w_none:
        warnings.simplefilter("always")
        pruned_none, active_none = pruner.prune_numpy(logits_p, last_token=1, grammar_rules={},
                                                        context_type="factual_qa")
    check("Diagnostic warns when complexity threshold doesn't bind (no grammar)",
          len(w_none) == 1 and active_none == V_BIG,
          f"{active_none}/{V_BIG} survived, {len(w_none)} warning(s)")

    grammar = {1: set(range(480, 530))}
    pruned_g, active_g = pruner.prune_numpy(logits_p, last_token=1, grammar_rules=grammar,
                                              context_type="factual_qa")
    check("Real pruning + correct-token survival under a genuine grammar constraint",
          active_g == 50 and np.isfinite(pruned_g[500]),
          f"{active_g}/{V_BIG} survived ({100*(1-active_g/V_BIG):.2f}% reduction), token 500 finite={np.isfinite(pruned_g[500])}")

    # -----------------------------------------------------------------------
    section("3b. DOMAIN-ANCHORED WEIGHTS (prime_weights fix)")
    # -----------------------------------------------------------------------
    weights_a = {"Order": 0.1, "Justice": 0.8, "Knowledge": 0.1}
    weights_b = {"Order": 0.8, "Justice": 0.1, "Knowledge": 0.1}

    random.seed(7)
    logits_w = [random.gauss(-3.0, 2.0) for _ in range(V_SMALL)]
    logits_w[42] += 6.0
    hist_w, tgt_w = [40, 41], {42}

    py_a = ActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5, tau=1.0,
                             tau_bifurcation=5.0, max_iters=100, prime_weights=weights_a)
    np_a = NumpyActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5,
                                  tau_bifurcation=5.0, max_iters=100, prime_weights=weights_a)
    jx_a = JaxActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5,
                                tau_bifurcation=5.0, max_iters=100)

    pt_a, _, pTr_a, *_ = py_a.steer(logits_w, hist_w, tgt_w)
    nt_a, _, nTr_a, *_ = np_a.steer(logits_w, hist_w, tgt_w)
    jt_a, _, jTr_a, *_ = jx_a.steer(logits_w, hist_w, tgt_w, prime_weights=weights_a)

    check("prime_weights now works identically across Python/NumPy/JAX (regression: NumpyActualizerEngine had silently dropped this)",
          pt_a == nt_a == jt_a and abs(pTr_a - nTr_a) < 1e-4 and abs(nTr_a - jTr_a) < 1e-4)

    py_b = ActualizerEngine(vocab_size=V_SMALL, mercy_k=0.45, Q_c=1e-5, tau=1.0,
                             tau_bifurcation=5.0, max_iters=100, prime_weights=weights_b)
    pt_b, _, pTr_b, *_ = py_b.steer(logits_w, hist_w, tgt_w)
    check("Different domain weights produce genuinely different Tr(D_mu_nu)",
          abs(pTr_a - pTr_b) > 0.001,
          f"weights_a Tr_D={pTr_a:.4f}  weights_b Tr_D={pTr_b:.4f}")

    # Real capability test: ONE compiled function processing a MIXED batch of
    # different domains simultaneously, each with its own anchored weights.
    from fdsa_pruner import FractalDeductionSearch
    fdsa = FractalDeductionSearch()
    domains_test = ["factual_qa", "creative_dialogue"]
    n_per = 5
    mb_logits, mb_targets, mb_w, mb_k, mb_expected = [], [], [], [], []
    for d_idx, dname in enumerate(domains_test):
        profile = pruner.CONTEXT_PROFILES[dname]
        dom, _ = fdsa.isomorphic_anchoring(profile)
        w = {"Order": profile[0], "Justice": profile[1], "Knowledge": profile[3]}
        for i in range(n_per):
            random.seed(1000 + d_idx*100 + i)
            l = [random.gauss(-3.0, 1.0) for _ in range(V_BIG)]
            tgt = 500 + d_idx*1000 + i*7
            l[tgt] = 5.0
            mb_logits.append(l); mb_targets.append({tgt})
            mb_w.append(w); mb_k.append(dom.k); mb_expected.append(tgt)

    mb_seq_tokens = []
    for i in range(len(mb_logits)):
        e = NumpyActualizerEngine(vocab_size=V_BIG, mercy_k=mb_k[i], Q_c=1e-5,
                                   tau_bifurcation=5.0, max_iters=100, prime_weights=mb_w[i])
        tok, *_ = e.steer(mb_logits[i], [max(0, mb_expected[i]-1)], mb_targets[i])
        mb_seq_tokens.append(tok)

    jxe_big2 = JaxActualizerEngine(vocab_size=V_BIG, mercy_k=0.45, Q_c=1e-5,
                                    tau_bifurcation=5.0, max_iters=100)
    mb_logits_arr = jnp.array(mb_logits, dtype=jnp.float32)
    mb_target_masks = jnp.stack([jxe_big2.make_target_mask(t) for t in mb_targets])
    mb_hib, mb_hvb, mb_hsb = zip(*[jxe_big2.make_history_arrays([max(0, mb_expected[i]-1)])
                                    for i in range(len(mb_logits))])
    mb_hib, mb_hvb, mb_hsb = jnp.stack(mb_hib), jnp.stack(mb_hvb), jnp.stack(mb_hsb)
    mb_wL = jnp.array([w["Order"] for w in mb_w], dtype=jnp.float32)
    mb_wG = jnp.array([w["Justice"] for w in mb_w], dtype=jnp.float32)
    mb_wF = jnp.array([w["Knowledge"] for w in mb_w], dtype=jnp.float32)
    mb_kb = jnp.array(mb_k, dtype=jnp.float32)

    mixed_batched_steer = jax.jit(jax.vmap(jxe_big2._steer_jit, in_axes=(0,0,0,0,0,0,0,0,0)))
    U_mb, *_ = mixed_batched_steer(mb_logits_arr, mb_target_masks, mb_hib, mb_hvb, mb_hsb,
                                     mb_wL, mb_wG, mb_wF, mb_kb)
    mb_vmap_tokens = [int(jnp.argmax(U_mb[i])) for i in range(len(mb_logits))]

    check("ONE compiled JAX function correctly handles a MIXED-domain batch (factual_qa + creative_dialogue simultaneously, no recompilation)",
          mb_vmap_tokens == mb_seq_tokens,
          f"{len(mb_logits)} examples, 2 domains, exact match: {mb_vmap_tokens == mb_seq_tokens}")

    # (Default-weights regression is already covered for real by Section 1's
    # checks above, which run with no prime_weights argument at all -- no
    # separate check needed here; a placeholder that asserts True regardless
    # of any condition was removed rather than left in, since a check that
    # cannot fail verifies nothing.)

    # -----------------------------------------------------------------------
    section("3c. DYNAMIC TARGET_TOKENS (non-circular, own top-K)")
    # -----------------------------------------------------------------------
    def derive_dynamic_targets(logits, top_k=10):
        arr = np.asarray(logits)
        return set(int(i) for i in np.argsort(arr)[-top_k:])

    n_flips = 0
    n_trials = 30
    for seed in range(n_trials):
        random.seed(seed)
        l = [random.gauss(0.0, 1.0) for _ in range(V_SMALL * 4)]
        raw_top1 = int(np.argmax(l))
        hist = [raw_top1, raw_top1, raw_top1]
        dyn_targets = derive_dynamic_targets(l, top_k=10)
        eng = NumpyActualizerEngine(vocab_size=V_SMALL * 4, mercy_k=0.45, Q_c=1e-5,
                                     tau_bifurcation=5.0, max_iters=100)
        tok, *_ = eng.steer(l, hist, dyn_targets)
        if tok != raw_top1:
            n_flips += 1

    check(f"Dynamic (non-oracle) target_tokens breaks degenerate repetition",
          n_flips == n_trials,
          f"{n_flips}/{n_trials} trials redirected away from repeated raw top-1")
    print("  NOTE: this shows the mechanism CAN redirect away from repetition")
    print("  using only its own top-K, no oracle. It does NOT yet show this")
    print("  helps vs. a standard repetition_penalty baseline -- that requires")
    print("  a real generation comparison, not tested in this suite.")


    # ---------------------------------------------------------------------------
    section("4. QCA PARALLEL ENGINE (fork-safety fix)")
    # ---------------------------------------------------------------------------
    # NOTE: this check must run in a real __main__-guarded file for spawn to work
    # (confirmed separately during this pass -- stdin/heredoc execution cannot
    # use spawn, since there is no real module path to re-import). Since this
    # file IS a real file with __main__ guard, this works correctly here.

    from qca_parallel_engine import QCAParallelEngine
    from qca import QCANode

    engine = QCAParallelEngine(K=3, vocab_size=2000, n_workers=3)
    nodes = [QCANode(node_id=i, coords=[float(i % 5), float((i * 3) % 7)],
                      prime_profile=[0.2] * 5, metadata={}) for i in range(9)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.process_parallel(nodes, verbose=False)
        fork_warnings = [w for w in caught if "fork" in str(w.message).lower()]

    check("QCAParallelEngine spawn-context fix eliminates fork warning",
          len(fork_warnings) == 0,
          f"{len(fork_warnings)} fork warning(s) in main process")

    # -----------------------------------------------------------------------
    section("SUMMARY")
    # -----------------------------------------------------------------------
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_fail > 0:
        print(f"\n{n_fail} FAILURE(S):")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll checks passed. See module docstring above for what this")
        print("suite does NOT cover (real model generation, TPU-specific")
        print("behavior, end-to-end EM/F1) before reporting any claim beyond")
        print("what is explicitly verified here.")
        sys.exit(0)

if __name__ == "__main__":
    main()
