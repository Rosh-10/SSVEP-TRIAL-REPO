"""zs_params.py -- trainable parameters per model and per fusion row (read-only on results/)."""
import glob, json
import pandas as pd
import zs_config as C

P = {}
for f in glob.glob(str(C.RESULTS_DIR / "*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if isinstance(d, dict) and {"subject", "method", "sl", "acc"} <= d.keys() and d.get("n_params") is not None:
        P.setdefault(d["method"], set()).add(int(d["n_params"]))

rows = [{"row": m, "kind": "network", "params_min": min(v), "params_max": max(v)} for m, v in sorted(P.items())]

# fusion rows -> member networks (fusion.csv name -> result method names); CCA features add 0 trainable params
FUSE = {"cnnref": ["cnn_ref"], "gnnv2c_g2": ["gnnv2c_g2"], "dense19": ["gnnv3g_dense19"], "b5d19": ["gnnv3l_b5d19"],
        "b5d9 (9 ch)": ["gnnv3l_b5d9"], "lr (9 ch)": ["gnnv3e_lr"],
        "dense19 x3 seeds": ["gnnv3g_dense19", "gnnv3k_s42d19", "gnnv3k_s43d19"],
        "b5d19 x3 seeds": ["gnnv3l_b5d19", "gnnv3m_s42b5d19", "gnnv3m_s43b5d19"]}
try: fz = pd.read_csv(C.RESULTS_DIR / "consolidated" / "fusion.csv"); fz = fz[fz.window == "1p0"].set_index("net")
except Exception: fz = None
for name, ms in FUSE.items():
    if not all(m in P for m in ms): rows.append({"row": f"{name} + CCA", "kind": "fusion", "params_min": None, "params_max": None}); continue
    r = {"row": f"{name} + CCA", "kind": "fusion", "params_min": sum(min(P[m]) for m in ms), "params_max": sum(max(P[m]) for m in ms)}
    if fz is not None and name in fz.index: r["net_alone_1p0"] = fz.loc[name, "network_alone"]; r["fuse4_1p0"] = fz.loc[name, "fuse4"]
    rows.append(r)

D = pd.DataFrame(rows); D.to_csv(C.RESULTS_DIR / "consolidated" / "params.csv", index=False)
print(D.to_string(index=False))
