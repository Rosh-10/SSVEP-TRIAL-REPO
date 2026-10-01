"""
Test accuracy with fine-resolution FFT spectrum features.
Expected: 80%+ for single-subject leave-one-block-out (vs 5% for 9-band power).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
# Load S1 spectrum features
spectrum_path = Path("feature_extraction") / "S1_spectrum.npz"

try:
    data = np.load(spectrum_path)["spectrum"]
except FileNotFoundError:
    print("[ERROR] S1_spectrum.npz not found. Run step9d_spectrum_features.py first.")
    exit(1)



print(f"[INFO] Loaded spectrum features: shape {data.shape}")

# Flatten to (240, 17920) where 17920 = 64 channels × 280 freq bins
X = data.reshape(240, -1)
y = np.repeat(np.arange(40), 6)

print(f"[INFO] Flattened to X shape: {X.shape}")

# Split: blocks 1-5 train (200 trials), block 6 test (40 trials)
train_idx = [i for i in range(240) if i % 6 < 5]
test_idx = [i for i in range(240) if i % 6 == 5]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"[INFO] Training set: {X_train.shape} (blocks 1-5)")
print(f"[INFO] Test set: {X_test.shape} (block 6)")

# Train
print("\n[TRAINING] Fitting logistic regression...")
model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_train, y_train)

# Test
acc = model.score(X_test, y_test)

print(f"\n" + "=" * 60)
print(f"S1 Single-subject accuracy (spectrum): {acc:.1%}")
print(f"Chance (40 targets): {1/40:.1%}")
print(f"Improvement over 9-band power: 5% → {acc:.1%} ({acc*20:.0f}x better)")
print("=" * 60)

if acc > 0.70:
    print("\n[SUCCESS] Spectrum features are highly discriminative!")
    print("Proceed to Step 10 (Graph Construction) with spectrum features.")
elif acc > 0.50:
    print("\n[OK] Spectrum features are good. Proceed to Step 10.")
else:
    print("\n[WARNING] Accuracy still modest. May need subject-specific preprocessing.")
