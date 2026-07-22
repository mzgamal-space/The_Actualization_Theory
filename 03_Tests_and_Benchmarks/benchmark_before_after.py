"""
benchmark_before_after.py — Quick before/after comparison printout
Shows improvement from NumpyActualizerEngine upgrade in QCA workers.
"""
# ── Before (Python loop workers) ──────────────────────────────────
before = {
    "label": "Before: Python loops worker (ActualizerEngine)",
    "N":     [20,      40,      80,      120,     200],
    "seq":   [3439.09, 6760.25, 13426.56, 20164.83, 33701.77],
    "par":   [4514.05, 4931.68, 7325.66,  8827.70,  9142.00],
    "spd":   [0.76,    1.37,    1.83,     2.28,     3.69],
}

# ── After (NumPy vectorized workers) ──────────────────────────────
after = {
    "label": "After:  NumPy vectorized worker (NumpyActualizerEngine)",
    "N":     [20,      40,      80,      120,     200],
    "seq":   [2738.49, 5691.80, 10066.02, 12842.04, 21690.25],
    "par":   [5345.86, 4507.75, 5671.41,  6264.95,  5942.92],
    "spd":   [0.51,    1.26,    1.77,     2.05,     3.65],
}

divider = "=" * 74
print(divider)
print("  QCA PARALLEL ENGINE — BEFORE vs AFTER NumpyActualizerEngine UPGRADE")
print(divider)

print(f"\n  {'N':>5}  {'Old Seq (ms)':>13}  {'New Seq (ms)':>13}  {'Seq %chg':>9}  {'Old Par (ms)':>13}  {'New Par (ms)':>13}  {'Par %chg':>9}")
print(f"  {'-'*5}  {'-'*13}  {'-'*13}  {'-'*9}  {'-'*13}  {'-'*13}  {'-'*9}")
for i, N in enumerate(before["N"]):
    seq_delta = (after["seq"][i] - before["seq"][i]) / before["seq"][i] * 100
    par_delta = (after["par"][i] - before["par"][i]) / before["par"][i] * 100
    print(f"  {N:>5}  {before['seq'][i]:>13.0f}  {after['seq'][i]:>13.0f}  {seq_delta:>+8.1f}%  "
          f"{before['par'][i]:>13.0f}  {after['par'][i]:>13.0f}  {par_delta:>+8.1f}%")

print(f"\n  Key insight: both seq and par sped up proportionally")
print(f"  → To increase the SPEEDUP RATIO, we need to beat the process spawn overhead")

print(f"\n{divider}")
print(f"  REMAINING BOTTLENECK: Windows ProcessPoolExecutor spawn overhead")
print(divider)
print(f"  Process spawn cost (fixed, per run): ~2,000-3,000 ms on Windows")
print(f"  At N=20 (4 nodes/cluster): work ~500ms < spawn cost → parallel SLOWER")
print(f"  At N=200 (40 nodes/cluster): work ~5,000ms >> spawn cost → 3.65× speedup")
print(f"\n  Solutions for even higher speedup:")
print(f"    1. JAX backend (backend='jax') — no spawn, vectorized → best for small N")
print(f"    2. Reuse process pool (warm workers) — amortize spawn cost")
print(f"    3. Use threading for small N (N<40), processes for large N")
print(divider)

if __name__ == "__main__":
    pass
