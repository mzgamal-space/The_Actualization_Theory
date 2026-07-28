# CHANGELOG — Engine Correctness Pass

This documents every change made in this pass, why, and how each was verified.
Run `python3 02_Core_Engine/test_engine_equivalence.py` to reproduce all checks
(12/12 pass, confirmed stable across 3 consecutive runs, CPU only).

## Fixed

### 1. `actualizer_engine.py` — D_future formula mismatch (real bug)
The docstring stated the D_future proxy should be `log(p_v) + 1` (the correct
unconstrained partial derivative of Shannon entropy). The code actually computed
`-log(p_v) * 0.08` — a different, undocumented formula, inconsistent with its
own docstring AND with `numpy_actualizer_engine.py`, which already implemented
`log(p_v) + 1` correctly. This caused `Tr(D_mu_nu)` to diverge by orders of
magnitude between the "equivalent" engines on identical inputs (measured:
`0.000027` vs `-0.522246` before the fix).
**Fix:** code now matches the docstring's own derivation, and matches the
NumPy engine exactly.
**Verified:** `Tr_D` now matches to 6 decimal places across 10 random seeds.

### 2. `actualizer_engine.py` — latent `p_v == 0.0` skip (real bug, previously inactive)
The Python engine skipped the D_global penalty entirely for any token with
exactly-zero probability (e.g. one already masked to `-inf` by an upstream
pruner). The NumPy engine applies D_global unconditionally — no such skip.
This didn't surface in dense-logit tests (nothing was exactly zero), but
activates the moment `steer()` receives pruner-masked logits, which is exactly
how these two components are meant to be used together.
**Fix:** removed the skip; both engines now behave identically on masked input.
**Verified:** tested directly with 100/300 tokens set to `-inf`; both engines
match exactly (`Tr_D` diff = 0.00000000).

### 3. `jax_actualizer_engine.py` — new file, JIT-compiled port
Ported from the now-corrected `numpy_actualizer_engine.py`. Verified
numerically equivalent (10 seeds, token + `Tr_D` match). Measured on CPU
(no TPU/GPU available in this environment):
  - Single-call: JAX matches NumPy speed exactly (JIT doesn't beat NumPy's
    BLAS on this workload at CPU scale) — reported honestly, not inflated.
  - **Batched (`vmap`) execution of 20 independent `steer()` calls is ~5-7x
    faster than sequential calls, with exact token-for-token agreement.**
    This is the mechanism actually worth using on real hardware — batch
    across examples, not single calls.
  - **Not yet verified on TPU.** CPU and accelerator backends can produce
    small floating-point differences under XLA; re-run
    `test_engine_equivalence.py` on real TPU hardware before reporting any
    TPU-specific number.
  - Known incomplete edge case, documented in the file: when the reference
    engines converge via the Q_c threshold but `Tr_D > tau_bifurcation`
    ("not actualized"), they return a specific fallback token (last target
    token seen in history, or 0). This JAX port always returns
    `argmax(final_U)` in that branch. Matches reference behavior for the
    "actualized" and "max-iters-exhausted" cases; does not yet replicate
    that one fallback branch. Flagged in the file, not silently mismatched.

### 4. `fdsa_pruner.py` — overclaiming docstring + silent no-op risk
The class docstring claimed "up to 99.99% reduction / 12.4x speedup at
V=100,000" without qualification. Measured directly: with **no** `grammar_rules`
supplied (open-ended generation — e.g. free-text QA), the fractal-dimension
complexity threshold alone prunes **~0%** of the vocabulary at natural-language
logit scale (V=32,128, `factual_qa` context: threshold computes to ≈-14.8,
which no realistic logit falls below). The large-reduction claim is only true
when a genuine `grammar_rules` constraint is supplied — a fundamentally
different use case (constrained/structured decoding) than open-ended
generation.
**Fix:** docstring corrected to state this precisely. Added a `RuntimeWarning`
in `prune_numpy` that fires whenever the complexity threshold alone prunes
nothing, naming the exact condition, so this can't cause a silent false-null
benchmark result again.
**Verified both regimes directly:**
  - No grammar: 32128/32128 survive, warning fires as designed.
  - With a real grammar constraint (50 valid tokens): 50/32128 survive
    (99.84% reduction), correct token confirmed still present.

### 5. `qca_parallel_engine.py` — fork/JAX incompatibility (real risk)
`ProcessPoolExecutor` defaulted to `fork` on Linux. If a JAX/TPU runtime is
already initialized in the parent process (true of any real generation
pipeline), forking risks corrupting or deadlocking worker state — this is a
documented JAX limitation, not specific to this code, but this code was
exposed to it.
**Fix:** forced `multiprocessing.get_context("spawn")`.
**Verified:** confirmed the fork-incompatibility warning no longer fires,
across 3 runs, in a real (non-stdin) file execution.

### 6. `qca_parallel_engine.py` — scope clarification (prevents a specific real mistake)
Traced a previously reported "4.13x speedup, real TriviaQA benchmark" result
back to this file. Found: `_worker_process_cluster` never calls a real model —
each node's logits are drawn from a synthetic Gaussian seeded by
embedding-derived coordinates. The real dataset is used only to produce those
embeddings; no decode step exists anywhere in this file, so its output tokens
cannot be meaningfully compared to ground-truth answers. Any EM/F1 reported
"from" this engine's output was computed elsewhere and reused, not computed
from this engine's actual return values.
**Fix:** added an explicit, prominent class-level docstring warning stating
this scope limitation directly, so this specific mistake — reporting this
module's synthetic diagnostic output as a real-dataset generation result —
cannot recur silently.

### 7. `numpy_actualizer_engine.py` — silently dropped `prime_weights` capability (real regression)
`ActualizerEngine` (the pure-Python reference) already supports
domain-conditional drift weights via a `prime_weights` dict, read fresh inside
`compute_drift_tensor` on every call (`w_L = self.prime_weights.get("Order",
0.35)`, etc.). `NumpyActualizerEngine` — meant to be a faithful, faster port —
instead hardcoded `self.w_L = 0.35`, `self.w_G = 0.35`, `self.w_F = 0.20` as
fixed constructor-time constants, with no way to override them. This matters
directly for domain-anchored steering (mapping a prompt's classified domain
profile onto the drift weights): the capability existed in the slow reference
engine and was silently missing from the two faster engines anyone would
actually use.
**Fix:** `NumpyActualizerEngine` now accepts an optional `prime_weights` dict,
same keys/defaults as the reference.
**Verified:** identical `Tr(D_mu_nu)` (within 1e-4) across Python/NumPy/JAX
with an explicit non-default `prime_weights` config, and confirmed two
different weight configs produce genuinely different `Tr_D` values (not just
structurally accepted and ignored).

### 8. `jax_actualizer_engine.py` — weights/k moved from closure constants to traced arguments
Originally ported the same hardcoded-weights limitation from
`numpy_actualizer_engine.py` (see #7), and additionally baked `mercy_k` into
the compiled function at construction time — meaning a different domain's `k`
would have required building and re-JIT-compiling an entirely new engine
instance.
**Fix:** `w_L, w_G, w_F, k` are now traced runtime arguments to the jitted
`steer_fn`, not Python floats captured by closure. One compiled function now
handles any domain's weights and contraction factor without recompiling.
**Verified, and this is the genuinely new capability worth highlighting:** a
single `vmap`-batched call correctly processes a **mixed batch of different
domains simultaneously** (5 `factual_qa` + 5 `creative_dialogue` examples in
one call, each anchored via the existing `isomorphic_anchoring` mechanism to
its own weights and `k`) — exact token-for-token match against rebuilding a
fresh sequential engine per domain. This is the concrete mechanism for
"dynamic, domain-anchored" steering requested directly by the repo author.

### 9. Dynamic (non-circular) `target_tokens` — verified mechanism, not yet a verified improvement
`ActualizerEngine.steer()` requires `target_tokens` — for open-ended
generation (e.g. closed-book QA) there is no non-circular source for this
without already knowing the answer. Tested instead: deriving `target_tokens`
dynamically as the model's own top-K candidates by raw logit at each step (no
oracle, no ground truth — computable from any model's own output alone).
**Verified:** with a raw top-1 token artificially repeated 3x in history
(simulating degenerate repetition) and `target_tokens` = the model's own
top-10, the steering mechanism redirected away from the repeated token in
**30/30 trials**, using no information beyond what's available at real
generation time.
**Explicitly not yet shown:** that this redirection *improves* real output
quality, or that it does anything a standard `repetition_penalty` /
`no_repeat_ngram_size` decoding parameter doesn't already do more simply. A
fair test requires comparing against that baseline directly, on real
generation — not attempted in this pass; the mechanism-level result above is
real, but it is not yet an efficacy result.

## Not fixed (explicitly out of scope this pass, listed so nothing is implied silently)

- No real model (T5 or otherwise) generation has been run anywhere in this
  pass. This sandbox has no GPU/TPU and no network access to huggingface.co.
  Every number above comes from synthetic-but-labeled test logits at the
  correct scale (V=32,128) and a distribution shape consistent with this
  repo's own existing test conventions.
- `qca_parallel_engine.py` has not been wired to call a real generation
  function per node — that requires a genuine dependency-injection design
  compatible with `ProcessPoolExecutor`'s pickling constraints (closures/
  lambdas don't pickle; a real hook needs a top-level importable callable).
  This is real, separate engineering work, not a quick fix — flagged as a
  concrete next step, not attempted here to avoid delivering a
  half-working hook that creates a new false impression of completeness.
- The one known JAX-port edge case in item 3 above (fallback-token branch)
  remains unreplicated, documented in the file.

## How to verify this yourself

```
cd 02_Core_Engine
python3 test_engine_equivalence.py
```

Expect `12/12 checks passed`. The script exits non-zero on any failure, so it's
safe to wire into CI. Read the module docstring at the top of that file for
exactly what is and is not covered before citing any number from it externally.
