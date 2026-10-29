"""zs_data64.py -- 64-channel version of zs_data.prep (bank A + fb_stand), cached per subject/window.
prep64(sid, sl, idx=None) -> dict(A=(240,3,len(idx) or 64,T) float32, y=(240,))"""
import numpy as np
import zs_config as C, zs_data as D

def prep64(sid, sl, idx=None):
    f = C.CACHE_DIR / f"S{sid}_A64_{str(sl).replace('.', 'p')}.npy"
    if not f.exists():
        s, e, T = D.window_idx(sl)
        w = D.load_raw64(sid)[:, s:e].astype(np.float64)                  # (64,T,40,6)
        X = w.transpose(2, 3, 0, 1).reshape(C.N_CLS * C.N_BLK, 64, T)     # (240,64,T), same trial order as D.prep
        C.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        tmp = f.with_suffix(".tmp.npy"); np.save(tmp, D.fb_stand(D.apply_bank_A(X)).astype(np.float32)); tmp.replace(f)
    A = np.load(f, mmap_mode="r")
    return dict(A=np.array(A if idx is None else A[:, :, idx]), y=np.repeat(np.arange(C.N_CLS), C.N_BLK))

if __name__ == "__main__":
    for sl in (0.5, 1.0):
        for sid in C.SUBJECTS:
            prep64(sid, sl); print(f"S{sid:02d} sl={sl} cached", flush=True)
