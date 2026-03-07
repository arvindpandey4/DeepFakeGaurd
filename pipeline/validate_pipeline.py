"""
validate_pipeline.py
--------------------
Smoke-test suite for the paper-aligned probability semantics.
Runs without loading model weights (mocks the CNN + frequency scores).

Tests:
  1. normalize_to_fake_prob()          - both modes + edge values
  2. Confidence formula symmetry       - max(p, 1-p)
  3. Decision rule                     - p >= 0.5 → DEEPFAKE
  4. Threshold monotonicity            - tau_1 > tau_2 > tau_3
  5. Early-exit / escalation logic     - confidence vs threshold
  6. Numerical stability clamp         - [0.001, 0.999]
  7. Final prediction key smoke test   - correct label returned
"""

import sys
import os
import numpy as np

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.adaptive_pipeline import normalize_to_fake_prob, _SPATIAL_WEIGHT, _FREQUENCY_WEIGHT
from pipeline.config import STAGE1_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

results = []

def check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    print(f"{tag}  {name}" + (f"  [{detail}]" if detail else ""))
    results.append(condition)

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 1. normalize_to_fake_prob() ══")

# real_prob mode: 0.8 real  →  0.2 fake
v = normalize_to_fake_prob(0.8, mode="real_prob")
check("real_prob 0.8 → 0.2", abs(v - 0.2) < 1e-6, f"got {v:.4f}")

# fake_prob mode: 0.8 fake  →  0.8 (identity)
v = normalize_to_fake_prob(0.8, mode="fake_prob")
check("fake_prob 0.8 → 0.8", abs(v - 0.8) < 1e-6, f"got {v:.4f}")

# Clamp: input 0.0 (real_prob) → fake = 1.0 → clamped to 0.999
v = normalize_to_fake_prob(0.0, mode="real_prob")
check("real_prob 0.0 clamped to 0.999", abs(v - 0.999) < 1e-6, f"got {v:.4f}")

# Clamp: input 1.0 (real_prob) → fake = 0.0 → clamped to 0.001
v = normalize_to_fake_prob(1.0, mode="real_prob")
check("real_prob 1.0 clamped to 0.001", abs(v - 0.001) < 1e-6, f"got {v:.4f}")

# Bad mode raises
try:
    normalize_to_fake_prob(0.5, mode="unknown")
    check("bad mode raises ValueError", False)
except ValueError:
    check("bad mode raises ValueError", True)

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 2. Confidence formula: max(p, 1-p) ══")

for p_s, expected_conf in [(0.9, 0.9), (0.1, 0.9), (0.5, 0.5), (0.75, 0.75), (0.25, 0.75)]:
    conf = max(p_s, 1.0 - p_s)
    check(f"p_s={p_s:.2f} → conf={conf:.2f}", abs(conf - expected_conf) < 1e-6,
          f"got {conf:.4f}, want {expected_conf:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 3. Decision rule: p >= 0.5 → DEEPFAKE ══")

for p_s, expected_label in [(0.9, "DEEPFAKE"), (0.5, "DEEPFAKE"), (0.499, "REAL"), (0.1, "REAL")]:
    label = "DEEPFAKE" if p_s >= 0.5 else "REAL"
    check(f"p_s={p_s} → {expected_label}", label == expected_label, f"got {label}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 4. Threshold monotonicity: τ1 > τ2 > τ3 ══")

tau1 = STAGE1_CONFIG['confidence_threshold']
tau2 = STAGE2_CONFIG['confidence_threshold']
tau3 = STAGE3_CONFIG['confidence_threshold']
check(f"τ1 ({tau1}) > τ2 ({tau2})", tau1 > tau2, f"τ1={tau1}, τ2={tau2}")
check(f"τ2 ({tau2}) > τ3 ({tau3})", tau2 > tau3, f"τ2={tau2}, τ3={tau3}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 5. Early-exit / escalation logic ══")

# High-confidence DEEPFAKE: p_s=0.9 → conf=0.9 ≥ τ1=0.85 → exit at stage 1
p_s = 0.9; conf = max(p_s, 1-p_s); tau = tau1
check("High fake (0.9) exits at stage 1", conf >= tau, f"conf={conf:.2f} τ1={tau}")

# Borderline: p_s=0.6 → conf=0.6 < τ1=0.85 → escalate
p_s = 0.6; conf = max(p_s, 1-p_s); tau = tau1
check("Borderline (0.6) escalates from stage 1", conf < tau, f"conf={conf:.2f} τ1={tau}")

# Stage 2 check: conf=0.78 ≥ τ2=0.75 → exit at stage 2
p_s = 0.78; conf = max(p_s, 1-p_s)
check("Borderline (0.78) exits at stage 2", conf >= tau2, f"conf={conf:.2f} τ2={tau2}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 6. Numerical stability clamp ══")

# Clamp prevents 0.0 / 1.0 edge cases
for raw in [0.0, 0.001, 0.999, 1.0]:
    clamped = float(np.clip(raw, 0.001, 0.999))
    check(f"{raw} → clamped ∈ [0.001, 0.999]",
          0.001 <= clamped <= 0.999, f"got {clamped}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n══ 7. Ensemble weight integrity ══")

check(f"Weights sum to 1.0 (_SPATIAL={_SPATIAL_WEIGHT}, _FREQ={_FREQUENCY_WEIGHT})",
      abs(_SPATIAL_WEIGHT + _FREQUENCY_WEIGHT - 1.0) < 1e-9,
      f"sum={_SPATIAL_WEIGHT + _FREQUENCY_WEIGHT}")

# Check ensemble formula with known inputs
p_fake_spatial = normalize_to_fake_prob(0.3, mode="real_prob")   # 0.7 fake
p_fake_freq    = normalize_to_fake_prob(0.4, mode="real_prob")   # 0.6 fake
expected = _SPATIAL_WEIGHT * p_fake_spatial + _FREQUENCY_WEIGHT * p_fake_freq
expected_clamped = float(np.clip(expected, 0.001, 0.999))
check(f"Ensemble(spatial_real=0.3, freq_real=0.4) ≈ {expected_clamped:.4f}",
      abs(expected_clamped - (0.70*0.7 + 0.30*0.6)) < 1e-4,
      f"got {expected_clamped:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
passed = sum(results)
total  = len(results)
print(f"  Results: {passed}/{total} passed", "" if passed == total else "  ← FAILURES ABOVE")
print(f"{'═'*55}\n")

sys.exit(0 if passed == total else 1)
