"""Stage-1 constants. Tags: [REPO]=copied from official main.py/config.py/function.py/FBCCA.py, [MY CHOICE]=mine."""
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = os.environ.get("SSVEP_DATASET", "benchmark").lower()   # benchmark (default, = paper config) | beta | ucsd
assert DATASET in ("benchmark", "beta", "ucsd"), f"unknown SSVEP_DATASET={DATASET}"
_SUF = "" if DATASET == "benchmark" else "_" + DATASET            # results_beta/, cache_beta/ keep datasets apart
DATA_DIR = Path(os.environ.get("SSVEP_RAW", ROOT.parent / "data" / "raw" / DATASET))  # [MY CHOICE] override with env SSVEP_RAW
CACHE_DIR = ROOT / f"cache{_SUF}"      # [MY CHOICE] 9-channel float64 copies, so each .mat is read once
RESULTS_DIR = ROOT / f"results{_SUF}"  # [MY CHOICE]

FS = 250                                    # [REPO] main.py rfs
CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]  # [REPO] 0-based
LATENCY = int((0.14 + 0.5) * FS)            # [REPO] =160 ; 0.5 s pre-stim + 0.14 s visual delay
N_CLS, N_BLK = 40, 6                        # [REPO]
SUBJECTS = list(range(1, 36))               # [REPO] S1..S35
NH, NM = 4, 3                               # [REPO] config.Nh, config.Nm
# [REPO] main.py Dataset==1 branch (== config.py Dataset==1 branch; checked in test_stage1.py)
PHASE = np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
                  1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) * np.pi

# Bank A (3 bands, DNN path) [REPO] function.py filter_bank (the active, second definition)
A_PASS, A_STOP, A_HI_PASS, A_HI_STOP, A_GPASS, A_GSTOP, A_RP = [6, 14, 22], [4, 10, 16], 90, 100, 3, 40, 0.5
# Bank B (5 bands, order 15, SOS) [REPO] main.py FBITCCA block -> function.generate_filterbank(order=15)
B_WP = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
B_WS = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]   # NOTE: ignored by generate_filterbank when order is given
B_ORDER, B_RP = 15, 0.5
# Bank C (5 bands, FBCCA baseline) [REPO] FBCCA.py filterbank()/fbcca()
C_PASS, C_STOP, C_HI_PASS, C_HI_STOP = [6, 14, 22, 30, 38], [4, 10, 16, 24, 32], 90, 100

N_VAL = 4        # [MY CHOICE] source subjects held out for early stopping
SEED = 41        # [REPO] main.py setup_seed(41) ; fold val split uses SEED*1000+test_sid [MY CHOICE]

# ---- other datasets: only values certain from the dataset papers / toolbox docs; None = fill in after md_explore.py ----
if DATASET != "benchmark":
    FS, N_CLS, N_BLK, SUBJECTS = {"beta": (250, 40, 4, list(range(1, 71))), "ucsd": (256, 12, 15, list(range(1, 11)))}[DATASET]
    CHANNELS = LATENCY = PHASE = None   # TODO: set from md_explore.py output (None makes dependent code fail loudly, not silently use Benchmark values)