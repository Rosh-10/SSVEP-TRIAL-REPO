"""zs_elec.py -- electrode labels + 2D positions from 64-channels.loc (EEGLAB polar: index, theta deg, radius, label)."""
import re
import numpy as np
import zs_config as C

def load_loc():
    rows = [l.split() for l in open(C.DATA_DIR / "64-channels.loc") if l.strip()]
    lab = [r[3] for r in rows]
    th, rad = np.radians([float(r[1]) for r in rows]), np.array([float(r[2]) for r in rows])
    return lab, np.stack([rad * np.sin(th), rad * np.cos(th)], 1)   # x = right, y = front (theta 0 = nose)

def region(pattern=r"^(P\d|PZ|PO|O\d|OZ)"):
    """0-based channel indices whose label matches (default: parieto-occipital block, 19 channels)."""
    return [i for i, l in enumerate(load_loc()[0]) if re.match(pattern, l, re.I)]
