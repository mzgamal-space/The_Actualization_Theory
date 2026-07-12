# 07_CKT_Core (Computational Knowledge Theory Engine Core)

This folder contains the complete, production-grade implementation of the Upgraded Actualizer Engine integrated with the **DIEPT (Dynamic Information and Epistemic Phase Transformation)** framework, fully aligned with the mathematical parameters of the Consciousness and Prime-Base Intelligence (PBI) life cycle.

## 📂 Folder Structure

All core modules are designed with flat imports to allow plug-and-play execution in academic publication or presentation environments:

```
07_CKT_Core/
│
├── exceptions.py       # Custom exception classes for robust error handling.
├── diept.py            # DIEPTState, CausalDensityFunction, InformationCurvatureLoss, & OperatorParser.
├── mce.py              # ReferenceDomain and crystallized MCE classes.
├── thought.py          # CandidateThought class holding subspaces and Prime scores.
├── fdsa.py             # FractalDeductionSearch & VectorizedFDSAPruner (contraction search).
├── filters.py          # EpistemicVerificationSuite (Pipelines A & C, and C(R) Cost evaluation).
├── engine.py           # UpgradedActualizerEngine wrapping process_query().
│
├── tests/
│   └── test_ckt_core.py # Full unittest suite covering 36 validation cases.
│
└── examples/
    ├── run_diept_demo.py     # Simple DIEPT subspace phase angle & modality demo.
    └── run_lifecycle_demo.py # Complete end-to-end PBI Cognitive Life Cycle demo.
```

---

## 🔬 Mathematical Integration Details

This implementation addresses the core constraints of the Noureldin (2026) whitepapers:

### 1. Stage A0: Question Operator Triple
The parser maps the leading question operator (e.g., `"Why"`, `"What if"`, `"Should"`) to a specific context lens, Prime compliance weights, and target phase angle $\theta_{\text{target}}$:
- **`"Why"`**: Causal Law dominant lens, strict causal bounds ($\theta_{\text{target}} = 0.15$).
- **`"What if"`**: Speculation lens, relaxed boundary for creative generation ($\theta_{\text{target}} = 1.20$).

### 2. Theorem 7 CAKI Boundary (Unsimulability of Reality)
Instead of forcing CAKI to exactly $1.0$ (which violates the limit of finite simulations), we model the maximum throughput as:
$$I_{\text{max}} = I_{\text{in}} + \delta_{\text{finite}}$$
Where $\delta_{\text{finite}} > 0$ represents the unsimulable boundary of potential reality. This guarantees that:
$$\text{CAKI} = \frac{K_{\text{acc}}}{I_{\text{max}}} \in [0, 1)$$
Even a perfect thought in a finite system approaches but never reaches $1.0$.

### 3. Pipeline A & C Verification
- **Pipeline A (Justice)**: Utilizes the `CausalDensityFunction` depth-weighted scaling factor $C(x)$ to verify transitions. Causal violations collapse propensity to $0.0$.
- **Pipeline C (Mercy)**: Isolates candidate thoughts whose phase angle $\theta = \arctan(\|B\|/\|A\|)$ exceeds the query's $\theta_{\text{target}}$.
- **Epistemic Cost $C(R)$**: Evaluates cost with a *Justice Dominance Constraint* ($\lambda_L > \lambda_R$ and $\lambda_L > \lambda_D$), utilizing `InformationCurvatureLoss` to penalize cycles or logical contradictions (re-entrant loops).

---

## 🚀 Execution Instructions

Ensure you are inside the `07_CKT_Core` directory:

### Run the Unit Tests
Execute the 36-case unit test suite:
```bash
python -m unittest tests/test_ckt_core.py
```

### Run the Examples
1. **DIEPT Subspace Demo**:
   ```bash
   python examples/run_diept_demo.py
   ```
2. **Cognitive Lifecycle Demo**:
   ```bash
   python examples/run_lifecycle_demo.py
   ```
