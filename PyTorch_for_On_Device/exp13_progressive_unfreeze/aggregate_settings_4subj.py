# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Aggregate existing settings (no-FT, head-only, streaming AdaBN) per-batch across 4 subjects x 3 folds
(seed 42), from the exp5/exp7 per-subject CSVs — for the exp13 settings comparison."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PAR = os.path.dirname(HERE)
SUBJ = ["S01", "S02", "S03", "S04"]
ho = {s: pd.read_csv(f"{PAR}/exp7_headonly_4subj/results/headonly_{s}.csv").set_index("batch") for s in SUBJ}
ad = {s: pd.read_csv(f"{PAR}/exp5_streaming_adabn_incremental/results/streaming_incr_{s}.csv").set_index("batch") for s in SUBJ}

print("=== per-batch, averaged across 4 subjects x 3 folds (seed 42) ===")
print("%5s %8s %10s %16s" % ("batch", "no_ft", "head_only", "streaming_adabn"))
for b in range(2, 6):
    nf = np.mean([ho[s].loc[b, "no_ft_mean"] for s in SUBJ])
    h = np.mean([ho[s].loc[b, "headonly_mean"] for s in SUBJ])
    a = np.mean([ad[s].loc[b, "streaming_adabn_mean"] for s in SUBJ])
    print("%5d %8.2f %10.2f %16.2f" % (b, nf, h, a))
nf = np.mean([ho[s].loc[b, "no_ft_mean"] for b in range(2, 6) for s in SUBJ])
h = np.mean([ho[s].loc[b, "headonly_mean"] for b in range(2, 6) for s in SUBJ])
a = np.mean([ad[s].loc[b, "streaming_adabn_mean"] for b in range(2, 6) for s in SUBJ])
print("%5s %8.2f %10.2f %16.2f" % ("mean", nf, h, a))

print("\n=== exp13 K=1 multi-seed (10 draws/subj), mean_b2-5 per subject ===")
ms = {"S01": (85.91, 86.11), "S02": (65.58, 70.08), "S03": (74.79, 75.15), "S04": (84.01, 84.26)}
for s, (a0, a1) in ms.items():
    print("  %s: K0 %.2f  K1 %.2f  d %+.2f" % (s, a0, a1, a1 - a0))
k0 = np.mean([v[0] for v in ms.values()]); k1 = np.mean([v[1] for v in ms.values()])
print("  4-subj mean: K0 %.2f  K1 %.2f  d %+.2f" % (k0, k1, k1 - k0))
