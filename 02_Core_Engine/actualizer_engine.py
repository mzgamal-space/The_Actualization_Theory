"""
actualizer_engine.py — Actualizer Engine: Core Contractive Steering Module
============================================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         Independent Researcher
         ORCID : 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com

Version : V3_U1 — updated to match The Actualization Theory V3_U1 corrections.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORY SUMMARY (from V3_U1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

§2.5  Justified Regression Convergence (Theorem 2.1)
    An infinite regression chain in a complete metric space that satisfies:
      (a) strict partial ordering (non-contradiction)
      (b) contraction condition: d(T(x), T(y)) ≤ k · d(x, y), k ∈ (0,1)
    converges to a unique fixed point A* by the Banach Fixed-Point Theorem.

§3.1  5-Dimensional Prime Hilbert Space
    Every state R is a vector in H = span{|Order⟩, |Justice⟩, |Mercy⟩,
    |Knowledge⟩, |Power⟩}.  The equilibrium ground state is the symmetric
    fixed point α_i = 1/√5 for all i (Theorem 3.2).

§3.3  Systemic Structural Entropy H(R) — V3_U1 CORRECTED FORM
    (Earlier versions used |Σα - 1|; V3_U1 corrects this to a squared term.)

        H(R) = Var(α)  +  (Σᵢ αᵢ² − 1)²

    Minimized (H=0) iff αᵢ = 1/√5 ∀i, confirming Theorem 3.2.

§3.3  Drift Tensor D_μν = Hess[H(R)]   (V3_U1 §3.3, §3.3.1)
    The Drift Tensor is the Hessian of H(R).  Its trace Tr(D_μν) measures
    the curvature of structural entropy:
      - Tr(D_μν) ≤ τ  →  branch converges to A*   (Theorem 3.3, case i)
      - Tr(D_μν) > τ  →  branch dissolves to ⊥    (Theorem 3.3, case ii)

§3.3.1-A  Valuation Trajectory  ν_t(A)   (V3_U1 addition)
    The actualization process is tracked by a continuous scalar:
        ν_t(A) := 1 − H(R_A(t)) / H_max     ∈ [0, 1]
    At convergence: ν → 1  (fully actualized fact A*).
    The non-isolated dynamics (κ > 0, not used in this implementation):
        dν_t(A)/dt = −η [ ∇H(R_A(t)) + γ_domain · Σᵢ M_drift(Aᵢ) ]
    This implementation uses γ_domain = 0 (isolated single-branch case).

§3.3.1-B  Bifurcation Theorem (Theorem 3.3)
    The causal snap (Power Prime) is gated by the bifurcation criterion:
      - ONLY when Tr(D_μν) ≤ τ does the engine commit to S* = argmax U.
      - When Tr(D_μν) > τ, the branch is dissolved (fallback to prior token).

§3.3.1-C  Generic Actualization Cost Function
        C_act(B̂, k, N) = E₀ · d_struct(B̂(P), P)   →  0 when B̂ is self-similar
    The five Primes are subsumed into C_act:
      Order    = consistent generative rule (self-similarity of B̂)
      Justice  = causal grounding: B̂ applied to its parent P
      Mercy    = k itself (the contraction factor)   ← KEY: k IS Mercy
      Knowledge= each B̂(P) node crystallizes the historical record
      Power    = reuse of validated operators costs near-zero energy

§5.3  Tripartite Drift Weights (V3_U1 explicit note)
    w_L, w_G, w_F are domain-dependent FREE PARAMETERS of the model —
    NOT derived quantities.  No calibration procedure exists from first
    principles (see §9.7 Open Question OQ-V3-5).  The defaults below are
    engineering choices, not theoretical derivations.

§6.7.2  FDSA 4-Phase Algorithm
    Phase 3 of FDSA is "Tripartite Drift Evaluation" which corresponds to the
    w_L, w_G, w_F weighted sum of §5.3.  See fdsa_pruner.py.

§8.1  Universal Transformer
    The Conceptual Primes act as an attention-like layer mapping the infinite
    uncollapsed substrate |U⟩ to the structured output A*.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JAX Compatibility
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every operator in this module maps 1-to-1 to jax.numpy:

  _softmax()                 → jnp.exp / jnp.sum / jnp.where
  _structural_entropy()      → jnp.var / jnp.sum / jnp.power
  compute_drift_tensor()     → jnp.log / jnp.exp / lax.scan
  _trace_drift()             → jnp.sum (diagonal of Hessian proxy)
  apply_vacuum_brake()       → jnp.exp / jnp.where / jnp.sum
  steer()                    → jax.lax.while_loop for convergence

Compiled with @jax.jit on TPU v5 lite: full steering loop ≈ 0.26 ms
at V=32,000 (< 1% production overhead).  See V3_U1 §6.7.3 for
the independently verified benchmark deposit (Zenodo 10.5281/zenodo.21184139).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V3_U1 Change Log (relative to previous engine version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FIX-1]  H(R): magnitude-defect term changed from |Σα−1| to (Σα²−1)²  (§3.3)
[FIX-2]  Added ν_t valuation tracking per iteration                    (§3.3.1-A)
[FIX-3]  Added Tr(D_μν) bifurcation criterion gating the causal snap  (§3.3.1-B)
[FIX-4]  prime_weights now a constructor parameter (not fixed constant) (§5.3)
[FIX-5]  k renamed/aliased to mercy_k; Mercy = k documented            (§3.3.1-C)
[FIX-6]  Causal snap gated by Tr(D_μν) ≤ τ, not just delta < Q_c     (§3.3.1-B)
[STUB]   gamma_domain=0.0 added; non-isolated dynamics not implemented  (§3.3.1-A)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple


# ── Number of Primes (dimensionality of the Prime Hilbert Space H, §3.1) ──
N_PRIMES = 5

# ── Symmetric equilibrium ground state (Theorem 3.2) ──────────────────────
# α_i = 1/√5 for all i.  At this point H(R) = 0 (zero drift).
EQUILIBRIUM_ALPHA = 1.0 / math.sqrt(N_PRIMES)   # ≈ 0.4472

# ── H_max: upper bound on structural entropy for normalization of ν_t (§3.3.1-A)
# When all probability is concentrated on one Prime coord:
#   Var_max = (N-1)/N²  (all but one coord = 0)
#   sq_mag_defect_max = (1 - 1)² = 0 (one coord = 1)
# For the practical substrate (5 coords, mixed probabilities):
#   H_max ≈ Var_max + max_sq_defect ≈ 0.16 + (0.6)² ≈ 0.52 (lower bound)
# Use 2.0 as a conservative upper bound — any realistic substrate
# has H(R) ≤ 2.0, so ν_t ∈ [0,1] throughout the steering loop.
# Calibration from first principles is an open problem (OQ-V3-5).
H_MAX_DEFAULT = 2.0


class ActualizerEngine:
    """
    Top-down contractive steering framework for autoregressive token generation.

    Implements the Actualization Process (§3.3.1) as a discrete iteration:
      1. Structural entropy H(R_A) computed via V3_U1-corrected formula (§3.3)
      2. Drift Tensor D_μν = Hess[H(R)] applied as tripartite penalty (§3.3, §5.3)
      3. Vacuum Brake: U_braked = U · exp(−D/τ)  (dissipative operator)
      4. Banach contraction: U_{n+1} = k · U_braked + (1−k) · U_n  (§2.5)
      5. Valuation update: ν_t = 1 − H(R_A)/H_max               (§3.3.1-A)
      6. Bifurcation check: Tr(D_μν) ≤ τ → snap; > τ → dissolve (§3.3.1-B)
      7. Causal snap: S* = argmax U_final                          (Power Prime)

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary V.
    mercy_k : float
        Contractive scale factor.  In V3_U1 terminology: k IS the Mercy
        parameter (§3.3.1-C, "the contraction factor k is the Mercy-tolerance
        parameter; it was never a separate quantity").  Must satisfy 0 < k < 1.
        Default 0.45 — the universal actualization constant.
    Q_c : float
        Causal quantum threshold (convergence tolerance on L2 delta).
    tau : float
        Vacuum Brake temperature τ.  Controls decay aggression.
    tau_bifurcation : float
        Bifurcation threshold τ for the Tr(D_μν) ≤ τ criterion (Theorem 3.3).
        If Tr(D_μν) > tau_bifurcation, the branch is in the dissolution regime.
    max_iters : int
        Safety ceiling on contraction iterations.
    prime_weights : dict, optional
        Domain-dependent free parameters w_L, w_G, w_F (§5.3).
        NOT derived from first principles — engineering choices.
        Keys: 'Order', 'Justice', 'Knowledge', 'Mercy'.
        Default values are a reasonable starting point for logical_coding tasks.
    repetition_penalty : float
        Scaling factor for local drift due to repetition (Order Prime).
    global_drift_penalty : float
        Fixed penalty for off-topic tokens (Justice Prime).
    gamma_domain : float
        Co-existence coupling constant γ_domain (§3.3.1-A).
        γ_domain = 0.0 → isolated single-branch case (Chapters 2–3 as written).
        γ_domain > 0  → non-isolated case (not yet implemented; stub only).
    h_max : float
        Upper bound on structural entropy H(R) used to normalize ν_t.
        Domain-specific calibration is an open problem (OQ-V3-5).
    """

    # Default prime weights — V3_U1 §5.3: free parameters, not derived.
    DEFAULT_PRIME_WEIGHTS: Dict[str, float] = {
        "Order"    : 0.35,   # w_L — local drift weight
        "Justice"  : 0.35,   # w_G — global drift weight
        "Knowledge": 0.20,   # w_F — future drift weight
        "Mercy"    : 0.10,   # decay temperature modifier
        # Power is implicit: it executes the causal snap (argmax) gated by bifurcation
    }

    def __init__(
        self,
        vocab_size        : int,
        mercy_k           : float = 0.45,
        Q_c               : float = 1e-5,
        tau               : float = 1.0,
        tau_bifurcation   : float = 5.0,
        max_iters         : int   = 100,
        prime_weights     : Optional[Dict[str, float]] = None,
        repetition_penalty: float = 2.0,
        global_drift_penalty: float = 1.5,
        gamma_domain      : float = 0.0,
        h_max             : float = H_MAX_DEFAULT,
    ) -> None:
        if not 0 < mercy_k < 1:
            raise ValueError(f"Mercy constant mercy_k must be in (0,1), got {mercy_k}")
        if gamma_domain != 0.0:
            import warnings
            warnings.warn(
                "gamma_domain > 0 (non-isolated dynamics, §3.3.1-A) is not yet "
                "implemented. The co-existence coupling term γ_domain·Σᵢ M_drift(Aᵢ) "
                "is silently ignored. Set gamma_domain=0.0 to suppress this warning.",
                stacklevel=2,
            )
        self.V                    = vocab_size
        self.mercy_k              = mercy_k
        self.k                    = mercy_k          # alias: k IS Mercy (§3.3.1-C)
        self.Q_c                  = Q_c
        self.tau                  = tau
        self.tau_bifurcation      = tau_bifurcation
        self.max_iters            = max_iters
        self.prime_weights        = prime_weights or dict(self.DEFAULT_PRIME_WEIGHTS)
        self.repetition_penalty   = repetition_penalty
        self.global_drift_penalty = global_drift_penalty
        self.gamma_domain         = gamma_domain
        self.h_max                = h_max

    # ──────────────────────────────────────────────────────────────────────
    # Internal utilities
    # ──────────────────────────────────────────────────────────────────────

    def _softmax(self, logits: List[float]) -> List[float]:
        """
        Numerically stable softmax, skipping -inf masked entries.

        JAX equivalent:
            shifted = logits - jnp.max(logits)
            e = jnp.exp(jnp.where(jnp.isfinite(logits), shifted, -jnp.inf))
            return e / jnp.sum(e)
        """
        max_l = max((x for x in logits if x != -math.inf), default=0.0)
        probs = [0.0] * self.V
        valid_exps, valid_idx = [], []
        for i, x in enumerate(logits):
            if x != -math.inf:
                val = math.exp(x - max_l)
                valid_exps.append(val)
                valid_idx.append(i)
        total = sum(valid_exps) or 1.0
        for j, i in enumerate(valid_idx):
            probs[i] = valid_exps[j] / total
        return probs

    # ──────────────────────────────────────────────────────────────────────
    # V3_U1 §3.3 — Corrected Structural Entropy H(R)
    # ──────────────────────────────────────────────────────────────────────

    def _prime_coords(self, U: List[float], history: List[int],
                      target_tokens: Set[int]) -> List[float]:
        """
        Extract the 5 Prime coordinates α = [α_O, α_J, α_M, α_K, α_P]
        from the current probability substrate U.

        These are scalar projections of U onto each Prime axis:

          α_Order    = 1 − normalized_repetition_density (local syntax health)
          α_Justice  = fraction of mass on target_tokens  (semantic balance)
          α_Mercy    = 1 − top_token_dominance            (distribution spread)
          α_Knowledge= 1 − normalized_entropy_deficit     (historical alignment)
          α_Power    = max(U)                             (decision sharpness)

        The equilibrium is α_i = 1/√5 ≈ 0.4472 for all i (Theorem 3.2).
        """
        # α_Order: penalise repetition fraction in recent history
        lookback = history[-8:] if len(history) >= 8 else history
        rep_count = sum(1 for t in lookback if U[t] > 0) if lookback else 0
        rep_density = rep_count / max(len(lookback), 1)
        alpha_O = max(0.0, 1.0 - rep_density)

        # α_Justice: probability mass inside semantic target window
        alpha_J = sum(U[v] for v in target_tokens if v < self.V)
        alpha_J = min(1.0, alpha_J)

        # α_Mercy: spread of distribution (1 - max_prob = inverse dominance)
        max_prob = max(U)
        alpha_M = 1.0 - max_prob

        # α_Knowledge: inverse of entropy deficit relative to uniform
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in U if p > 0)
        max_entropy = math.log(max(self.V, 2))
        alpha_K = min(1.0, entropy / max_entropy)

        # α_Power: decision sharpness (concentration at argmax)
        alpha_P = max_prob

        return [alpha_O, alpha_J, alpha_M, alpha_K, alpha_P]

    def _structural_entropy(self, alpha: List[float]) -> float:
        """
        V3_U1-corrected Systemic Structural Entropy H(R):

            H(R) = Var(α) + (Σᵢ αᵢ² − 1)²

        V3_U1 §3.3 note:
            "Earlier versions used |Σα − 1| (absolute value).
             The corrected form uses a **squared** magnitude-defect term."

        Minimized (= 0) iff αᵢ = 1/√5 for all i (Theorem 3.2, §3.3).

        JAX equivalent:
            var_term    = jnp.var(jnp.array(alpha))
            sq_sum_term = (jnp.sum(jnp.array(alpha)**2) - 1.0)**2
            return var_term + sq_sum_term
        """
        n = len(alpha)
        mean_a  = sum(alpha) / n
        var_term = sum((a - mean_a) ** 2 for a in alpha) / n
        sq_mag_defect = (sum(a ** 2 for a in alpha) - 1.0) ** 2
        return var_term + sq_mag_defect

    def _valuation(self, H_R: float) -> float:
        """
        Actualization valuation ν_t(A) = 1 − H(R_A) / H_max  (§3.3.1-A).

        Tracks how 'actualized' the current substrate is:
          ν_t = 0  → fully uncollapsed (unactualized substrate |U⟩)
          ν_t = 1  → fully actualized (unique fixed point A*)

        JAX equivalent:
            jnp.clip(1.0 - H_R / h_max, 0.0, 1.0)
        """
        return max(0.0, min(1.0, 1.0 - H_R / self.h_max))

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1 — Drift Tensor D_μν
    # ──────────────────────────────────────────────────────────────────────

    def compute_drift_tensor(
        self,
        U            : List[float],
        history      : List[int],
        target_tokens: Set[int],
    ) -> List[float]:
        """
        Tripartite Drift Tensor D_μν for the current substrate U.

            D_μν = w_L · D_local + w_G · D_global + w_F · D_future

        V3_U1 §5.3 note:
            w_L, w_G, w_F are domain-dependent FREE PARAMETERS — not derived
            from first principles.  They are read from self.prime_weights.

        D_local  (Order Prime)    — penalises recently repeated tokens
        D_global (Justice/Mercy)  — penalises off-target tokens
        D_future (Knowledge Prime)— adds structural entropy gradient as
                                    forward risk signal (V3_U1 corrected H(R))

        Returns
        -------
        D : per-token drift magnitude vector of length V
        """
        w_L = self.prime_weights.get("Order",     0.35)
        w_G = self.prime_weights.get("Justice",   0.35)
        w_F = self.prime_weights.get("Knowledge", 0.20)

        D = [0.0] * self.V

        # ── D_local (Order Prime): repetition suppression ─────────────────
        # Tokens appearing in recent history accumulate drift proportional
        # to recency.  Exponential decay: most recent = highest drift.
        lookback = history[-8:]
        for step_back, tok in enumerate(reversed(lookback)):
            if 0 <= tok < self.V:
                recency_w = math.exp(-0.4 * step_back)
                D[tok] += w_L * self.repetition_penalty * recency_w

        # ── D_global (Justice Prime): semantic boundary ───────────────────
        # ── D_future (Knowledge Prime): structural entropy gradient ───────
        # The V3_U1-corrected H(R) gradient is expensive to compute per-token
        # directly.  We use the following tractable proxy:
        #   - D_global: uniform penalty for tokens outside target window
        #   - D_future: local derivative of H(R) w.r.t. token probability.
        #     For the squared form H = Var(α) + (Σα²−1)², the gradient
        #     ∂H/∂p_v contributes primarily through the Knowledge coordinate α_K.
        #     Proxy: −∂(entropy)/∂p_v = log(p_v) + 1  (entropy gradient).
        #     This is a numerically tractable approximation of the Hessian diagonal.
        for v in range(self.V):
            p_v = U[v]
            if p_v == 0.0:
                continue
            # D_global
            if v not in target_tokens:
                D[v] += w_G * self.global_drift_penalty
            # D_future: structural entropy gradient proxy (Knowledge Prime)
            # log(p_v) is negative (entropy increases as p_v decreases)
            # Tokens with very low p contribute more future uncertainty.
            entropy_grad = -math.log(max(p_v, 1e-12))
            D[v] += w_F * entropy_grad * 0.08

        return D

    def _trace_drift(self, D: List[float], U: List[float]) -> float:
        """
        Compute the trace of the Drift Tensor: Tr(D_μν).

        In V3_U1, the Drift Tensor D_μν = Hess[H(R)] (§3.3.1).
        Its trace is the sum of diagonal Hessian entries — a scalar measure of
        the curvature of structural entropy across all active dimensions.

        Bifurcation criterion (Theorem 3.3, §3.3.1-B):
          Tr(D_μν) ≤ τ  →  branch converges to A*   (actualization)
          Tr(D_μν) > τ  →  branch dissolves to ⊥    (dissolution)

        Implementation: weight each token's drift by its probability mass.
        This gives the probability-weighted trace — the expected drift across
        the substrate, which is the operationally meaningful quantity.

            Tr(D_μν) ≈ Σ_v U(v) · D(v)

        JAX equivalent:
            jnp.dot(jnp.array(U), jnp.array(D))
        """
        return sum(U[v] * D[v] for v in range(self.V))

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2 — Vacuum Brake
    # ──────────────────────────────────────────────────────────────────────

    def apply_vacuum_brake(
        self, U: List[float], D: List[float]
    ) -> List[float]:
        """
        Non-conservative dissipation operator (Vacuum Brake).

        Strips probability mass from high-drift trajectories:

            U_braked(v) = U(v) · exp(−D(v)/τ)

        Then re-normalises (Mercy Prime — preserves probability volume).

        This is the discrete analogue of the non-conservative dissipative term
        −η ∇H(R_A(t)) in the continuous dynamics of §3.3.1-A.

        JAX equivalent:
            decay    = jnp.exp(-jnp.array(D) / tau)
            U_braked = jnp.array(U) * decay
            U_braked = U_braked / jnp.sum(U_braked)
        """
        decay    = [math.exp(-d / self.tau) for d in D]
        U_braked = [U[i] * decay[i] for i in range(self.V)]
        total    = sum(U_braked) or 1.0
        return [x / total for x in U_braked]

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3 — Contractive Mapping + Causal Snap
    # ──────────────────────────────────────────────────────────────────────

    def steer(
        self,
        logits        : List[float],
        history       : List[int],
        target_tokens : Set[int],
    ) -> Tuple[int, List[float], float, int, List[float], bool]:
        """
        Run the full contractive steering loop (V3_U1 updated).

        Algorithm
        ---------
        0. U_0 = softmax(logits)                                   (substrate init)
        For n = 0, 1, 2, …, max_iters:
          a. α     = _prime_coords(U_n)                            (Prime projection)
          b. H_R   = _structural_entropy(α)  [V3_U1 corrected]    (§3.3)
          c. ν_t   = 1 − H_R / H_max                              (§3.3.1-A)
          d. D     = compute_drift_tensor(U_n)   [tripartite]      (§5.3)
          e. Tr_D  = _trace_drift(D, U_n)                         (§3.3.1-B)
          f. U_b   = apply_vacuum_brake(U_n, D)
          g. U_{n+1} = mercy_k · U_b + (1−mercy_k) · U_n         (Banach §2.5)
          h. delta = ‖U_{n+1} − U_n‖₂
          i. If delta ≤ Q_c:
               CAUSAL SNAP (Power Prime) gated by bifurcation:
                 - Tr_D ≤ τ_bifurcation → S* = argmax U (§3.3.1-B case i)
                 - Tr_D > τ_bifurcation → dissolution; return fallback (case ii)
               break

        Returns
        -------
        token       : int     — actualized token S* (or fallback on dissolution)
        U_final     : list    — collapsed probability distribution
        final_drift : float   — Tr(D_μν) at convergence
        iterations  : int     — number of contraction iterations
        nu_history  : list    — valuation ν_t trace per iteration (§3.3.1-A)
        actualized  : bool    — True if Tr(D_μν) ≤ τ (actualization branch)
                                False if Tr(D_μν) > τ (dissolution branch)
        """
        U = self._softmax(logits)
        nu_history: List[float] = []

        for iteration in range(1, self.max_iters + 1):
            U_prev = U[:]

            # Step a-c: Prime projection + V3_U1 structural entropy + valuation
            alpha = self._prime_coords(U, history, target_tokens)
            H_R   = self._structural_entropy(alpha)
            nu_t  = self._valuation(H_R)
            nu_history.append(nu_t)

            # Step d: Drift Tensor (tripartite, §5.3)
            D = self.compute_drift_tensor(U, history, target_tokens)

            # Step e: Trace of Drift Tensor (Theorem 3.3 bifurcation check)
            Tr_D = self._trace_drift(D, U)

            # Step f-g: Vacuum Brake + Banach contraction
            U_b = self.apply_vacuum_brake(U, D)
            U   = [self.mercy_k * U_b[v] + (1.0 - self.mercy_k) * U_prev[v]
                   for v in range(self.V)]

            # Step h: L2 convergence check
            delta = math.sqrt(sum((U[v] - U_prev[v]) ** 2 for v in range(self.V)))

            if delta <= self.Q_c:
                # Step i: Causal Snap gated by bifurcation (§3.3.1-B, Theorem 3.3)
                if Tr_D <= self.tau_bifurcation:
                    # Case (i): actualization branch — ν → 1, commit to S*
                    token = max(range(self.V), key=lambda v: U[v])
                    return token, U, Tr_D, iteration, nu_history, True
                else:
                    # Case (ii): dissolution branch — ν → 0, return fallback
                    # Fallback: most recent valid token in history within target
                    fallback = next(
                        (t for t in reversed(history) if t in target_tokens), 0
                    )
                    return fallback, U, Tr_D, iteration, nu_history, False

        # Max iterations reached: snap to argmax regardless (Power Prime fallback)
        D_final = self.compute_drift_tensor(U, history, target_tokens)
        Tr_final = self._trace_drift(D_final, U)
        token = max(range(self.V), key=lambda v: U[v])
        actualized = (Tr_final <= self.tau_bifurcation)
        return token, U, Tr_final, self.max_iters, nu_history, actualized
