"""Stage-2 LOSO runner. Saves per-fold JSON after every fold.
Usage (from ZERO_SHOT/):
  python3 zs_runner.py --sl 0.5 --methods fbcca
  python3 zs_runner.py --sl 0.5 1.0 --methods fbcca fbttcca_yf
"""
import json, argparse, time
import numpy as np
from pathlib import Path
import zs_config as C, zs_data as D, zs_baselines as B


def compute_itr(acc, T, n=40):
    t = T / C.FS + 0.5
    return B.itr(acc, n=n, t=t)


def run_fold(sid, sl, methods):
    fold = D.make_fold(sid)
    T = int(sl * C.FS)
    freqs, _, _ = D.load_freq_phase()
    X_test, y_test, _ = D.get_trials(sid, sl)
    results = {}

    if "fbcca" in methods:
        t0 = time.time()
        preds, scores = B.run_fbcca(X_test, freqs, fs=C.FS)
        acc = float((preds == y_test).mean())
        elapsed = round(time.time() - t0, 2)
        results["fbcca"] = dict(
            subject=sid, method="fbcca", sl=sl,
            acc=acc, itr=compute_itr(acc, T), elapsed=elapsed,
            preds=preds.tolist(), labels=y_test.tolist())
        print(f"  S{sid:02d} sl={sl} FBCCA       acc={acc:.4f}  ITR={results['fbcca']['itr']:.2f}  ({elapsed:.1f}s)")

    if "fbttcca_yf" in methods:
        # all 34 source subjects (train + val); no early stopping for baselines [REPO]
        src_ids = fold["train"] + fold["val"]
        print(f"  S{sid:02d} sl={sl} FBTtCCA_Yf  loading {len(src_ids)} source subjects...", flush=True)
        parts = [D.get_trials(s, sl) for s in src_ids]
        X_src = np.concatenate([p[0] for p in parts])
        y_src = np.concatenate([p[1] for p in parts])  # labels 0..39
        t0 = time.time()
        # sl in seconds — generate_cca_references uses t=linspace(0,sl,int(sl*fs))
        est, Yf = B.make_fbttcca_yf(freqs, sl, fs=C.FS)
        B.fit_fbttcca(est, X_src, y_src, Yf)
        preds, scores = B.pred_fbttcca(est, X_test)
        acc = float((preds == y_test).mean())
        elapsed = round(time.time() - t0, 2)
        results["fbttcca_yf"] = dict(
            subject=sid, method="fbttcca_yf", sl=sl,
            acc=acc, itr=compute_itr(acc, T), elapsed=elapsed,
            preds=preds.tolist(), labels=y_test.tolist())
        print(f"  S{sid:02d} sl={sl} FBTtCCA_Yf  acc={acc:.4f}  ITR={results['fbttcca_yf']['itr']:.2f}  ({elapsed:.1f}s)")

    return results


def save_result(r, results_dir):
    results_dir.mkdir(exist_ok=True, parents=True)
    sid, method, sl = r["subject"], r["method"], r["sl"]
    tag = str(sl).replace(".", "p")
    p = results_dir / f"fold_s{sid:02d}_{method}_{tag}.json"
    p.write_text(json.dumps(r, indent=1))


def summarise(all_results, sl, method):
    rs = [r for r in all_results if r["sl"] == sl and r["method"] == method]
    if not rs:
        return
    accs = np.array([r["acc"] for r in rs])
    itrs = np.array([r["itr"] for r in rs])
    n = len(accs)
    print(f"\n{method.upper()}  sl={sl}  n={n} subjects")
    print(f"  acc   mean={accs.mean():.4f}  std={accs.std():.4f}  se={accs.std()/np.sqrt(n):.4f}")
    print(f"  ITR   mean={itrs.mean():.2f}    std={itrs.std():.2f}    se={itrs.std()/np.sqrt(n):.2f}")
    print(f"  per-subject acc: {[round(a, 3) for a in accs.tolist()]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", nargs="+", type=float, default=[0.5])
    ap.add_argument("--methods", nargs="+", default=["fbcca"])
    ap.add_argument("--subjects", nargs="+", type=int, default=C.SUBJECTS)
    args = ap.parse_args()

    all_results = []
    for sl in args.sl:
        print(f"\n=== sl={sl} ===")
        for sid in args.subjects:
            res = run_fold(sid, sl, args.methods)
            for r in res.values():
                save_result(r, C.RESULTS_DIR)
                all_results.append(r)

    print("\n=== SUMMARY ===")
    for sl in args.sl:
        for method in args.methods:
            summarise(all_results, sl, method)