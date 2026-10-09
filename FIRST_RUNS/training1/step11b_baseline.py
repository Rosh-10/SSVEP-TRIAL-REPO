"""STEP 11b: honest logistic-regression baseline. Put in training/. Run: python training/step11b_baseline.py [subjects...]
Protocol: leave-one-block-out (6 folds); C chosen on a validation block, never on the test block."""
import sys, warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from data_loader import load_subject, split
warnings.filterwarnings('ignore', category=RuntimeWarning)          # spurious macOS matmul warnings; finiteness is asserted below
for s in map(int, sys.argv[1:] or [1, 15, 30]):
    X, y, b = load_subject(s); assert np.isfinite(X).all()
    accs = []
    for t in range(6):
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, b, test_block=t, val_block=(t+1) % 6)
        Xtr, Xva, Xte = (a.reshape(len(a), -1) for a in (Xtr, Xva, Xte))
        ms = {C: LogisticRegression(C=C, max_iter=300).fit(Xtr, ytr) for C in [1e-3, 1e-2, 1e-1, 1.0]}
        C = max(ms, key=lambda c: ms[c].score(Xva, yva))
        accs.append(ms[C].score(Xte, yte))
    print(f'S{s}: per-block {np.round(accs, 3)} | mean {np.mean(accs):.3f} +- {np.std(accs):.3f}')
