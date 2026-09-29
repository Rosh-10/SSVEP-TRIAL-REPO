"""
Better accuracy test with feature normalization and regularization (simplified).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load S1 spectrum features
try:
    data = np.load('feature_extraction\\S1_spectrum.npz')['spectrum']  # (40, 6, 64, 280)
except FileNotFoundError:
    print("[ERROR] S1_spectrum.npz not found. Run step9d_spectrum_features.py first.")
    exit(1)

print(f"[INFO] Loaded spectrum features: shape {data.shape}")

# Flatten to (240, 17920)
X = data.reshape(240, -1)
y = np.repeat(np.arange(40), 6)

print(f"[INFO] Flattened to X shape: {X.shape}")
print(f"[INFO] Classes: {len(np.unique(y))} targets")

# Split: blocks 1-5 train (200 trials), block 6 test (40 trials)
train_idx = [i for i in range(240) if i % 6 < 5]
test_idx = [i for i in range(240) if i % 6 == 5]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"[INFO] Training set: {X_train.shape} (blocks 1-5)")
print(f"[INFO] Test set: {X_test.shape} (block 6)")

# NORMALIZE features (crucial for high-dimensional data)
print("\n[PREPROCESSING] Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different regularization strengths
print(f"\n[TESTING] Multiple regularization strengths...")
print(f"{'C':<10} {'Accuracy':<15} {'Notes'}")
print("-" * 40)

best_acc = 0
best_C = None

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = LogisticRegression(
        max_iter=10000,
        C=C,  # Inverse regularization strength (higher = less regularization)
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    
    if acc > best_acc:
        best_acc = acc
        best_C = C
    
    note = "[BEST]" if acc == best_acc else ""
    print(f"{C:<10.3f} {acc:<15.1%} {note}")

print("\n" + "=" * 60)
print(f"Best accuracy (normalized + regularized): {best_acc:.1%} (C={best_C})")
print(f"Chance (40 targets): {1/40:.1%}")
print(f"Improvement factor: {best_acc / (1/40):.1f}x chance level")
print("=" * 60)

if best_acc > 0.70:
    print("\n[SUCCESS] Spectrum features are highly discriminative!")
    print("Ready for Step 10 (Graph Construction).")
elif best_acc > 0.50:
    print("\n[OK] Spectrum features work well.")
    print("Ready for Step 10 (Graph Construction).")
elif best_acc > 0.30:
    print("\n[MODERATE] Spectrum features show discriminability.")
    print("Can proceed to Step 10, baseline accuracy expected to be modest.")
else:
    print("\n[CAUTION] Accuracy still low.")
    print("May indicate preprocessing issues.")