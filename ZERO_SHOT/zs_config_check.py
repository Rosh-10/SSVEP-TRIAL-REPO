"""zs_config_check.py -- guard for zs_config.py edits. Run from ZERO_SHOT/ with the SAME environment both times.
  python3 zs_config_check.py --save    # before editing zs_config.py: writes zs_config_snapshot.json
  python3 zs_config_check.py --check   # after editing: exits 1 if any constant/path differs from the snapshot
  SSVEP_DATASET=beta python3 zs_config_check.py --show   # print constants for another dataset"""
import json, os, sys
from pathlib import Path
import numpy as np
import zs_config as C
SNAP = Path(__file__).resolve().parent / "zs_config_snapshot.json"
def enc(v):
    if isinstance(v, Path): return "PATH:" + os.path.relpath(v, C.ROOT)
    if isinstance(v, np.ndarray): return {"nd": v.tolist()}
    if isinstance(v, (np.integer, np.floating)): return v.item()
    if isinstance(v, (list, tuple)): return [enc(x) for x in v]
    return v
def state():
    return {k: enc(getattr(C, k)) for k in sorted(dir(C)) if k.isupper() and k != "ROOT"}
if __name__ == "__main__":
    s, mode = state(), (sys.argv[1] if len(sys.argv) > 1 else "--show")
    if mode == "--save": SNAP.write_text(json.dumps(s, indent=1)); print(f"saved {len(s)} constants -> {SNAP.name}")
    elif mode == "--check":
        old = json.loads(SNAP.read_text()); bad = [k for k in sorted(old) if old[k] != s.get(k, "<missing>")]
        for k in sorted(set(s) - set(old)): print(f"NEW  {k} = {str(s[k])[:60]}   (added, not a failure)")
        for k in bad: print(f"DIFF {k}: before={str(old.get(k, '<missing>'))[:80]}  after={str(s.get(k, '<missing>'))[:80]}")
        print("IDENTICAL" if not bad else f"{len(bad)} DIFFERENCES"); sys.exit(1 if bad else 0)
    else:
        for k, v in s.items(): print(f"{k} = {str(v)[:100]}")
