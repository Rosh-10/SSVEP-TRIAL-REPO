"""Stage-1 tests T1-T4 (+T6-style equivalence vs official files). Usage:
  python test_stage1.py --real            # your data (SSVEP_RAW or ../data/raw/benchmark)
  python test_stage1.py --synthetic       # crash/shape/index tests ONLY, not validation
Env: SSVEP_OFFICIAL=<dir with official function.py, FBCCA.py> (default ./official)"""
import os, re, sys, types, tempfile, importlib.util
from pathlib import Path
import numpy as np
from scipy.io import savemat
from scipy.signal import freqz, sosfreqz, sos2zpk, cheby1
import zs_config as C, zs_data as D

def rep(name, a):
    a = np.asarray(a); f = np.issubdtype(a.dtype, np.floating)
    print(f"  {name:<28}{str(a.shape):<26}{str(a.dtype):<9}nan={int(np.isnan(a).sum()) if f else 0} inf={int(np.isinf(a).sum()) if f else 0}"
          + (f" min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g} std={a.std():.4g}" if f else ""))

def make_synth(d, sids):
    """Known-answer encoding: value = 1e4*ch + t + 1e-3*target + 1e-4*block  (float32 file; exact for the integer part)."""
    d.mkdir(parents=True, exist_ok=True)
    ch, t, k, b = np.meshgrid(np.arange(64), np.arange(1500), np.arange(40), np.arange(6), indexing="ij")
    base = (1e4 * ch + t).astype(np.float64)
    rng = np.random.RandomState(0)
    for s in sids:
        x = (base + 0.001 * k + 0.0001 * b).astype(np.float32); savemat(d / f"S{s}.mat", {"data": x})
    fr = np.random.RandomState(1).permutation(np.round(np.arange(8, 15.9, 0.2), 1))
    savemat(d / "Freq_Phase.mat", {"freqs": fr[None], "phases": C.PHASE[None]})

def t1_t2(sids, sl_list, synth):
    print("\n=== T1 shapes/dtypes/NaN  +  T2 window indices ===")
    for sl in sl_list:
        s, e, T = D.window_idx(sl)
        print(f"[sl={sl}] LATENCY={C.LATENCY}  slice=[{s}:{e}]  T={T}  int((0.14+0.5)*250)={int((0.14+0.5)*250)}")
        assert s == 160 and e - s == T == int(sl * 250)
        sid = sids[0]
        d64 = D.load_raw64(sid); rep("raw64 .mat", d64); assert d64.shape == (64, 1500, 40, 6)
        d9 = D.load_raw9(sid); rep("9ch cache", d9); assert d9.shape == (9, 1500, 40, 6)
        X, y, blk = D.get_trials(sid, sl); rep("trials X", X); rep("y", y); rep("blk", blk)
        assert X.shape == (240, 9, T) and X.shape[2] == e - s
        # independent re-index straight from the 64-ch array, different code path
        for k, b in [(0, 0), (7, 3), (39, 5), (12, 1)]:
            assert np.array_equal(X[k * 6 + b], d64[C.CHANNELS][:, s:e, k, b].astype(np.float64)) and y[k * 6 + b] == k and blk[k * 6 + b] == b
        if synth:   # ramp: value = 1e4*ch64 + t  -> exact start/end/channel check
            v = np.floor(X[5 * 6 + 2] + 1e-6)
            assert np.array_equal(v[:, 0], 1e4 * np.array(C.CHANNELS) + 160) and np.array_equal(v[:, -1], 1e4 * np.array(C.CHANNELS) + 160 + T - 1)
            print("  synthetic ramp: first sample idx 160, last 160+T-1, channels = CHANNELS  OK")
        else:
            print("  first/last time idx used:", s, e - 1, "(independent re-index matches OK)")
        A = D.apply_bank_A(X); rep("bank A out", A); assert A.shape == (240, 3, 9, T)
        Cc = D.apply_bank_C(X); rep("bank C out", Cc); assert Cc.shape == (240, 5, 9, T)
        Bb = D.apply_bank_B(X); rep("bank B out", Bb); assert Bb.shape == (240, 5, 9, T)
        Z = D.fb_stand(A); rep("A after FB_stand", Z)
        assert np.allclose(Z.mean(-1), 0, atol=1e-8) and np.allclose(Z.std(-1), 1, atol=1e-6) or synth
        for n, a in (("A", A), ("B", Bb), ("C", Cc), ("Z", Z)): assert np.isfinite(a).all(), n
    fr, ph, ph_mat = D.load_freq_phase(); rep("freqs", fr); rep("phases(repo)", ph)
    print("  freqs[:8] (keyboard order, not sorted):", fr[:8], "| sorted?", bool(np.all(np.diff(fr) > 0)))
    if ph_mat is not None: print("  repo phase list == Freq_Phase.mat 'phases' (mod 2pi)?", bool(np.allclose(np.mod(ph, 2*np.pi), np.mod(ph_mat, 2*np.pi), atol=1e-6)))
    else: print("  Freq_Phase.mat has no 'phases' key")
    loc = C.DATA_DIR / "64-channels.loc"
    if loc.exists():
        lab = [l.split()[-1] for l in loc.read_text().splitlines() if l.strip()]
        print("  loc labels at CHANNELS (0-based):", [(i, lab[i]) for i in C.CHANNELS])
    else: print("  64-channels.loc not found in DATA_DIR (needed for G1 later)")

def poles_ba(B, A): return np.abs(np.roots(A)).max()
def poles_sos(sos): return np.abs(sos2zpk(sos)[1]).max()

def t3():
    print("\n=== T3 filter responses, max|pole|, SOS-vs-analytic on sines ===")
    fs = C.FS; t = np.arange(0, 12, 1 / fs)
    def gain_db(h): return 20 * np.log10(np.maximum(abs(h), 1e-12))
    for name, coefs, sb, hi in (("A", D.bank_A_coefs(), C.A_STOP, C.A_HI_STOP), ("C", D.bank_C_coefs(), C.C_STOP, C.C_HI_STOP)):
        print(f"bank {name} (b,a), single-pass dB at stopband-low / passband-low / mid / passband-hi / stopband-hi")
        for i, (B, A, N) in enumerate(coefs):
            pb = (C.A_PASS if name == "A" else C.C_PASS)[i]
            f = [sb[i], pb, (pb + 90) / 2, 90, hi]
            h = freqz(B, A, worN=f, fs=fs)[1]
            print(f"  band{i} N={N} len(b)={len(B)} padlen={3*(max(len(B),len(A))-1)} max|pole|={poles_ba(B,A):.6f}  dB:", np.round(gain_db(h), 1))
            # measured filtfilt gain on a long sine == |H|^2 (zero-phase doubles dB)
            fm = (pb + 90) / 2 if (pb + 90) / 2 < 100 else 50
            x = np.sin(2 * np.pi * 20 * t) if pb < 20 else np.sin(2 * np.pi * 45 * t)
            fx = 20 if pb < 20 else 45
            from scipy.signal import filtfilt
            y = filtfilt(B, A, x, padtype="odd", padlen=3 * (max(len(B), len(A)) - 1))[500:-500]
            meas = np.sqrt(np.mean(y**2) / np.mean(x[500:-500]**2)); pred = abs(freqz(B, A, worN=[fx], fs=fs)[1][0])**2
            print(f"      sine {fx} Hz: filtfilt gain measured={meas:.5f} predicted |H|^2={pred:.5f}")
    print("bank B (order 15 SOS) vs the same filter in (b,a) form")
    for i, (wp, sos) in enumerate(zip(C.B_WP, D.bank_B_sos())):
        b, a = cheby1(C.B_ORDER, C.B_RP, wp, btype="bandpass", output="ba", fs=fs)
        f = [wp[0] - 2, wp[0], (wp[0] + wp[1]) / 2, wp[1]]
        h = sosfreqz(sos, worN=f, fs=fs)[1]
        fx = 20 if wp[0] < 20 else 45
        from scipy.signal import sosfilt, lfilter
        x = np.sin(2 * np.pi * fx * t); y = sosfilt(sos, x)[1000:]
        meas = np.sqrt(np.mean(y**2) / np.mean(x[1000:]**2)); pred = abs(sosfreqz(sos, worN=[fx], fs=fs)[1][0])
        yb = lfilter(b, a, x)[1000:]; measb = np.sqrt(np.mean(yb**2) / np.mean(x[1000:]**2))
        print(f"  band{i} wp={wp} SOS max|pole|={poles_sos(sos):.6f}  (b,a) max|pole|={poles_ba(b,a):.6f} | dB@[wp0-2,wp0,mid,wp1]:", np.round(gain_db(h), 1),
              f"| sine{fx}Hz SOS meas={meas:.5f} pred={pred:.5f} (b,a) meas={measb:.5g}")
        assert poles_sos(sos) < 1 and abs(meas - pred) < 1e-3

def load_official():
    od = Path(os.environ.get("SSVEP_OFFICIAL", C.ROOT / "official"))
    if not (od / "function.py").exists(): print("\n[skip] official files not found at", od); return None, None
    st = types.SimpleNamespace(Dataset=1, C=9, rfs=250, num_class=40, Nh=4, Nm=3, T=125, sample_length=0.5, nBlock=6,
                               list_freqs=np.linspace(8, 15.8, 40), list_phase=C.PHASE)
    m = types.ModuleType("config"); m.config = st; sys.modules["config"] = m   # official config.py would loadmat('/home/wc/...') on import
    mods = []
    for n in ("function", "FBCCA"):
        sp = importlib.util.spec_from_file_location("off_" + n, od / f"{n}.py"); mo = importlib.util.module_from_spec(sp); sp.loader.exec_module(mo); mods.append(mo)
    return mods, od

def t6_equiv(sl_list):
    print("\n=== T6-style: my banks vs the OFFICIAL functions on random input (same filters? NOT a data validation) ===")
    mods, od = load_official()
    if mods is None: return
    F, FB = mods; rng = np.random.RandomState(0)
    for sl in sl_list:
        T = int(sl * 250); X = rng.randn(6, 9, T)
        # official filter_bank expects (label,block,C,T) -> (label,block,3,C,T); feed one trial at a time
        ref = np.stack([F.filter_bank(X[None, i:i+1])[0, 0] for i in range(6)])
        print(f" T={T} bank A  max|mine - official filter_bank| =", np.abs(D.apply_bank_A(X) - ref).max())
        refC = np.stack([np.stack([FB.filterbank(X[i], 250, k) for k in range(5)]) for i in range(6)])
        print(f" T={T} bank C  max|mine - official FBCCA.filterbank| =", np.abs(D.apply_bank_C(X) - refC).max())
    sos_ref = F.generate_filterbank(C.B_WP, C.B_WS, srate=250, order=15, rp=0.5)
    print(" bank B  max|mine - official generate_filterbank| (SOS coefs) =", max(np.abs(a - b).max() for a, b in zip(D.bank_B_sos(), sos_ref)))
    a, c = D.bank_A_coefs(), D.bank_C_coefs()
    print(" bank A coefs == first 3 bands of bank C coefs?", all(np.array_equal(a[i][0], c[i][0]) and np.array_equal(a[i][1], c[i][1]) for i in range(3)))
    # FB_stand vs official (sklearn) and phase lists vs official files
    Z = rng.randn(5, 3, 9, 125) * 7 + 3; Z[0, 0, 0] = 5.0
    print(" FB_stand max|mine - official(sklearn)| =", np.abs(D.fb_stand(Z) - F.FB_stand(Z.transpose(0,1,2,3))).max())
    for fn, pat in (("main.py", r"is_dataset == 1:(.*?)elif is_dataset == 2"), ("config.py", r"self\.Dataset==1:(.*?)if self\.Dataset==2")):
        txt = (od / fn).read_text(); blk = re.search(pat, txt, re.S).group(1)
        lst = np.array([float(v) for v in re.search(r"list_phase\s*=\s*np\.array\(\[(.*?)\]\)", blk, re.S).group(1).split(",")]) * np.pi
        print(f" phase list in official {fn} Dataset=1 branch == zs_config.PHASE?", bool(np.allclose(lst, C.PHASE)))

def t4(synth_sids=None):
    print("\n=== T4 leakage checks ===")
    allf = [D.make_fold(s) for s in C.SUBJECTS]
    for f in allf:
        tr, va, te = set(f["train"]), set(f["val"]), {f["test"]}
        assert not (tr & va) and not (tr & te) and not (va & te) and len(tr) + len(va) + 1 == 35 and len(va) == C.N_VAL
    assert sorted(f["test"] for f in allf) == C.SUBJECTS
    assert allf == [D.make_fold(s) for s in C.SUBJECTS]
    print(f" 35 folds: each subject tested once; train/val/test disjoint; |train|={len(allf[0]['train'])} |val|={len(allf[0]['val'])}; deterministic re-creation OK")
    print(" example fold test=S1 val =", allf[0]["val"], "| seed", C.SEED)
    D.save_fold_manifest(); print(" manifest saved ->", C.RESULTS_DIR / "stage1_folds.json")
    # normalization is per-trial: output for one trial must not change when other trials change
    rng = np.random.RandomState(0); x = rng.randn(10, 3, 9, 125); x2 = x.copy(); x2[1:] = rng.randn(9, 3, 9, 125) * 50 + 9
    assert np.array_equal(D.fb_stand(x)[0], D.fb_stand(x2)[0]); print(" FB_stand is per-trial: trial 0 unchanged when all other trials are replaced OK (no cross-trial/cross-subject stats)")
    sids = synth_sids or C.SUBJECTS
    f = D.make_fold(sids[0], n_val=1 if synth_sids else C.N_VAL, subjects=sids) if synth_sids else allf[0]
    fd = D.build_fold(f, 0.5)
    for k, v in fd.items(): print(f"  split {k:<5} subjects={sorted(set(v['sid'].tolist()))[:6]}{'...' if len(set(v['sid'].tolist()))>6 else ''} n={len(v['y'])}  A{v['A'].shape}"); rep("   A", v["A"])
    assert set(fd["test"]["sid"]) == {f["test"]} and not (set(fd["train"]["sid"]) | set(fd["val"]["sid"])) & {f["test"]}
    assert not set(fd["train"]["sid"]) & set(fd["val"]["sid"])
    print(" per-trial subject-id audit: test subject absent from train/val OK")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--real"
    if mode == "--synthetic":
        tmp = Path(tempfile.mkdtemp()); C.DATA_DIR = tmp / "raw"; C.CACHE_DIR = tmp / "cache"; C.RESULTS_DIR = tmp / "results"
        sids = [1, 2, 3, 4, 5]; make_synth(C.DATA_DIR, sids); print("SYNTHETIC DATA: crash/shape/index tests only")
        t1_t2(sids, (0.5, 1.0), True); t3(); t6_equiv((0.5, 1.0)); t4(sids)
    else:
        sids = C.SUBJECTS; t1_t2(sids[:1], (0.5, 1.0), False); t3(); t6_equiv((0.5, 1.0)); t4()
