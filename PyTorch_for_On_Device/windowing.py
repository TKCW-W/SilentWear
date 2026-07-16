# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
SilentWear windowing — faithful copy of the on-device windowing used in Onnx4Deeploy
(`onnx4deeploy/data/silent_wear_datasource.py`, onset-anchored, commit 77e269d).

Reproduces the paper's `WINDOWING_SPEC.md` and matches the shipped `balanced_acc_no_ft`
exactly (S01 sess3: batch1 80.56%, batch2 81.67%):

  - read h5 key "emg" from data_raw_and_filt/<subj>/<cond>/sess_<S>_batch_<B>.h5
  - segment by contiguous Label_int runs
  - ONE window per run, anchored at the run ONSET, length 700 (= int(1.4*500))
  - skip only if the 700-window would overrun the end of the recording
  - 14 filtered channels in the model channel order (below); NO normalization
  - optional rest-downsampling to the min class count (20/class -> 180 windows)
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

WIN = 700  # int(1.4 s * 500 Hz)

# 14 filtered channels in the model channel_order [0,1,2,5,3,4,7,6,8,15,9,14,10,13]
CHANNEL_ORDER = [0, 1, 2, 5, 3, 4, 7, 6, 8, 15, 9, 14, 10, 13]
EMG_FILT_COLS = [f"Ch_{c}_filt" for c in CHANNEL_ORDER]


def load_windows(
    data_path: str,
    subject: str,
    session: int,
    batch: int,
    condition: str = "vocalized",
    downsample_rest: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X = (N,1,14,700) float32, y = (N,) int64."""
    h5 = f"{data_path}/{subject}/{condition}/sess_{session}_batch_{batch}.h5"
    df = pd.read_hdf(h5, key="emg")
    emg = df[EMG_FILT_COLS].values.astype(np.float32)      # (N_samples, 14), channel_order
    lab = df["Label_int"].values

    chg = np.where(np.diff(lab) != 0)[0] + 1
    starts = np.concatenate([[0], chg])                    # onset of each label run

    X: List[np.ndarray] = []
    y: List[int] = []
    for s in starts:
        if s + WIN > len(emg):                             # window overruns recording end -> drop
            continue
        w = emg[s:s + WIN]                                 # (700,14) onset-anchored
        X.append(w.T[np.newaxis, np.newaxis, :, :])        # (1,1,14,700)
        y.append(int(lab[s]))
    X = np.concatenate(X, axis=0)
    y = np.array(y, dtype=np.int64)

    if downsample_rest:
        from collections import Counter
        c = Counter(y.tolist())
        mn = min(v for k, v in c.items() if k != 0)        # min command-class count
        rest = np.where(y == 0)[0]
        if len(rest) > mn:
            keep_rest = np.random.RandomState(seed).choice(rest, size=mn, replace=False)
            keep = np.array(sorted(list(np.where(y != 0)[0]) + list(keep_rest)))
            X, y = X[keep], y[keep]
    return X, y


def stratified_draw(X: np.ndarray, y: np.ndarray, per_class: int, seed: int = 42
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Draw `per_class` windows per class (stratified) — the FT training subset."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in sorted(set(y.tolist())):
        pool = np.where(y == c)[0]
        idx += list(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    idx = np.array(sorted(idx))
    return X[idx], y[idx]
