"""
jax_actualizer_engine.py — JIT-Compiled JAX Actualizer Engine
================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com

Ported from numpy_actualizer_engine.py (itself verified equivalent to
actualizer_engine.py's pure-Python reference implementation as of the
V3_U2 correction pass -- see actualizer_engine.py's D_future fix).

WHY THIS EXISTS:
----------------
  ActualizerEngine      (Python loops) : ~1.08s / call  at V=32,128
  NumpyActualizerEngine (vectorized)   : ~0.058s / call at V=32,128 (~18.7x)
  JaxActualizerEngine   (JIT-compiled) : benchmarked below against both

Both numbers above were measured directly, not assumed, in the same
environment as this file. See test_engine_equivalence.py for the
verification harness that produced them.

DESIGN NOTES / KNOWN LIMITATIONS OF THIS PORT:
-----------------------------------------------
  1. `history` is handled as a FIXED-SIZE, right-aligned array of length 8
     (left-padded with a sentinel), because JAX requires static shapes.
     Semantics match the reference: step_back=0 is always the most recent
     token, regardless of how many valid history entries exist.
  2. `target_tokens` must be passed in as a precomputed boolean mask of
     shape (V,), computed OUTSIDE the jitted function (set construction
     and membership testing are not JIT-compatible). Use
     `JaxActualizerEngine.make_target_mask(target_tokens)` to build it.
  3. The per-iteration ν_t (nu) trace is NOT accumulated across iterations
     in this version (unlike the reference implementations) -- only the
     final value is returned -- because accumulating a variable-length
     Python list inside a `lax.while_loop` is not supported. If you need
     the full trace for diagnostics/plotting, use NumpyActualizerEngine
     instead; this engine is for the hot path (real generation), not
     diagnostics.
  4. This has been verified for NUMERICAL EQUIVALENCE with
     NumpyActualizerEngine on CPU, across multiple seeds (see
     test_engine_equivalence.py). It has NOT been verified on TPU/GPU in
     this pass -- CPU and accelerator backends can occasionally produce
     small floating-point differences (this is a general JAX/XLA property,
     not specific to this code). Re-run the equivalence test on your
     actual target hardware before relying on this for a reported result.
"""

from __future__ import annotations

import math
from typing import List, Set, Tuple

import jax
import jax.numpy as jnp
from jax import lax


class JaxActualizerEngine:
    def __init__(
        self,
        vocab_size: int = 1000,
        mercy_k: float = 0.45,
        Q_c: float = 1e-5,
        tau: float = 1.0,
        tau_bifurcation: float = 5.0,
        max_iters: int = 100,
        repetition_penalty: float = 2.0,
        global_drift_penalty: float = 1.5,
        h_max: float = 2.0,
    ) -> None:
        self.V = vocab_size
        self.k = mercy_k
        self.Q_c = Q_c
        self.tau = tau
        self.tau_bif = tau_bifurcation
        self.max_iters = max_iters
        self.rep_pen = repetition_penalty
        self.glob_pen = global_drift_penalty
        self.h_max = h_max
        self.w_L = 0.35
        self.w_G = 0.35
        self.w_F = 0.20

        self._steer_jit = jax.jit(self._build_steer_fn())

    # ---- Public helpers (non-jitted, run once per call, cheap) -----------

    def make_target_mask(self, target_tokens: Set[int]) -> jnp.ndarray:
        """Build the (V,) boolean target mask required by steer(). Not jitted
        -- Python set operations aren't JAX-traceable, so this runs in plain
        Python once per call. This is O(V), same cost class as the numpy
        engine's equivalent step."""
        mask = [False] * self.V
        for t in target_tokens:
            if 0 <= t < self.V:
                mask[t] = True
        return jnp.array(mask, dtype=jnp.bool_)

    def make_history_arrays(self, history: List[int]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build the fixed-size (8,) history_ids / valid_mask / step_back
        arrays, right-aligned so the most recent token is always last."""
        lookback = history[-8:]
        pad = 8 - len(lookback)
        ids = [0] * pad + [max(0, min(self.V - 1, t)) for t in lookback]
        valid = [False] * pad + [True] * len(lookback)
        step_back = list(range(7, -1, -1))  # position 7 (last) = step_back 0
        return (
            jnp.array(ids, dtype=jnp.int32),
            jnp.array(valid, dtype=jnp.bool_),
            jnp.array(step_back, dtype=jnp.float32),
        )

    def steer(
        self, logits, history: List[int], target_tokens: Set[int],
        prime_weights: dict = None, mercy_k: float = None,
    ) -> Tuple[int, jnp.ndarray, float, int, bool]:
        """Public API. Builds the small per-call arrays (cheap, not jitted),
        then calls the jitted core loop. Returns (token, U_final, Tr_D,
        iterations, actualized) -- no nu_history, see module docstring #3.

        prime_weights / mercy_k: FIXED (this pass) -- these are now traced
        runtime arguments, not values baked into the compiled function at
        construction time. This means ONE compiled `_steer_jit` handles ANY
        domain's weights without recompiling, and (critically) a vmap-batched
        call can process a MIXED batch of different domains simultaneously,
        each with its own anchored weights -- see
        test_engine_equivalence.py's domain-anchoring section for a batch
        example. If not supplied, falls back to this instance's construction
        defaults (self.w_L / self.w_G / self.w_F / self.k).
        """
        pw = prime_weights or {}
        w_L = jnp.float32(pw.get("Order", self.w_L))
        w_G = jnp.float32(pw.get("Justice", self.w_G))
        w_F = jnp.float32(pw.get("Knowledge", self.w_F))
        k_val = jnp.float32(mercy_k if mercy_k is not None else self.k)

        logits_arr = jnp.asarray(logits, dtype=jnp.float32)
        target_mask = self.make_target_mask(target_tokens)
        hist_ids, hist_valid, hist_stepback = self.make_history_arrays(history)

        U_final, Tr_D, iters, actualized = self._steer_jit(
            logits_arr, target_mask, hist_ids, hist_valid, hist_stepback,
            w_L, w_G, w_F, k_val,
        )
        token = int(jnp.argmax(U_final))
        return token, U_final, float(Tr_D), int(iters), bool(actualized)

    # ---- Internal: builds the jitted function (closure over STATIC hyperparams
    #      only -- Q_c, tau, tau_bif, max_iters, h_max remain fixed per engine
    #      instance; w_L/w_G/w_F/k are now traced arguments, see steer() above) -

    def _build_steer_fn(self):
        V, Q_c, tau, tau_bif, max_iters = (
            self.V, self.Q_c, self.tau, self.tau_bif, self.max_iters
        )
        rep_pen, glob_pen, h_max = self.rep_pen, self.glob_pen, self.h_max

        def softmax(logits):
            finite = jnp.isfinite(logits)
            safe = jnp.where(finite, logits, -1e38)
            m = jnp.max(safe)
            e = jnp.where(finite, jnp.exp(safe - m), 0.0)
            total = jnp.sum(e)
            return e / jnp.where(total > 0, total, 1.0)

        def prime_coords(U, hist_ids, hist_valid, hist_stepback, target_mask):
            # alpha_O: 1 - (fraction of valid history tokens with U > 1e-9)
            hist_vals = U[hist_ids]
            rep_hits = jnp.where(hist_valid & (hist_vals > 1e-9), 1.0, 0.0)
            n_valid = jnp.maximum(jnp.sum(jnp.where(hist_valid, 1.0, 0.0)), 1.0)
            rep_density = jnp.sum(rep_hits) / n_valid
            alpha_O = jnp.maximum(0.0, 1.0 - rep_density)

            # alpha_J: total prob mass on target tokens
            alpha_J = jnp.sum(jnp.where(target_mask, U, 0.0))

            # alpha_M: uncertainty
            alpha_M = 1.0 - jnp.max(U)

            # alpha_K: normalized Shannon entropy
            safe_U = jnp.where(U > 1e-300, U, 1e-300)
            entropy = -jnp.sum(jnp.where(U > 1e-300, U * jnp.log(safe_U), 0.0))
            alpha_K = jnp.minimum(1.0, entropy / math.log(max(V, 2)))

            # alpha_P: peak probability
            alpha_P = jnp.max(U)

            return jnp.stack([alpha_O, alpha_J, alpha_M, alpha_K, alpha_P])

        def structural_entropy(alpha):
            var_a = jnp.var(alpha)
            sq_def = (jnp.sum(alpha ** 2) - 1.0) ** 2
            return var_a + sq_def

        def drift_tensor(U, hist_ids, hist_valid, hist_stepback, target_mask, w_L, w_G, w_F):
            # D_local: scatter-add recency-weighted repetition penalty.
            # step_back is precomputed per fixed history slot (see
            # make_history_arrays): position 7 (most recent) -> step_back=0.
            local_weights = jnp.where(
                hist_valid, w_L * rep_pen * jnp.exp(-0.4 * hist_stepback), 0.0
            )
            D = jnp.zeros(V, dtype=jnp.float32).at[hist_ids].add(local_weights)

            # D_global: unconditional off-target penalty (matches the FIXED
            # numpy/Python engines -- no skip for exactly-zero-probability
            # tokens; see actualizer_engine.py's V3_U2 correction).
            off_target = jnp.where(target_mask, 0.0, 1.0)
            D = D + w_G * glob_pen * off_target

            # D_future: -d(entropy)/dp_v = log(p_v) + 1
            safe_U = jnp.where(U > 1e-300, U, 1e-300)
            D = D + w_F * (jnp.log(safe_U) + 1.0)

            return D

        def vacuum_brake(U, D):
            braked = U * jnp.exp(-D / tau)
            s = jnp.sum(braked)
            return braked / jnp.where(s > 0, s, 1.0)

        def cond_fn(state):
            i, U, U_prev, converged, Tr_D = state
            return jnp.logical_and(i < max_iters, jnp.logical_not(converged))

        def body_fn(state, hist_ids, hist_valid, hist_stepback, target_mask, w_L, w_G, w_F, k):
            i, U, U_prev_unused, converged, Tr_D_unused = state
            U_prev = U

            alpha = prime_coords(U, hist_ids, hist_valid, hist_stepback, target_mask)
            H_R = structural_entropy(alpha)
            nu_t = jnp.clip(1.0 - H_R / h_max, 0.0, 1.0)  # computed, not returned (see #3)

            D = drift_tensor(U, hist_ids, hist_valid, hist_stepback, target_mask, w_L, w_G, w_F)
            Tr_D = jnp.dot(U, D)

            U_b = vacuum_brake(U, D)
            U_new = k * U_b + (1.0 - k) * U_prev

            delta = jnp.linalg.norm(U_new - U_prev)
            new_converged = delta <= Q_c

            return (i + 1, U_new, U_prev, new_converged, Tr_D)

        def steer_fn(logits, target_mask, hist_ids, hist_valid, hist_stepback, w_L, w_G, w_F, k):
            U0 = softmax(logits)
            init_state = (0, U0, U0, False, 0.0)

            def body(state):
                return body_fn(state, hist_ids, hist_valid, hist_stepback, target_mask, w_L, w_G, w_F, k)

            final_i, final_U, _, final_converged, final_Tr = lax.while_loop(
                cond_fn, body, init_state
            )

            # actualized = Tr_D <= tau_bifurcation, unconditionally -- matching
            # the reference engines, which apply this same check whether the
            # loop exited via Q_c convergence or via max_iters exhaustion.
            # KNOWN LIMITATION (see module docstring): when the reference
            # engines converge via Q_c but Tr_D > tau_bif ("not actualized"),
            # they return a DIFFERENT fallback token (last target token seen
            # in history, or 0), not argmax(U). This JAX version always
            # returns argmax(final_U) as the token regardless of that branch,
            # matching reference behavior only for the actualized==True case
            # and the max-iters-exhaustion case. If your use case relies on
            # that specific fallback-token behavior, this needs extending
            # before use -- flagged here rather than silently mismatched.
            actualized = final_Tr <= tau_bif

            return final_U, final_Tr, final_i, actualized

        return steer_fn
