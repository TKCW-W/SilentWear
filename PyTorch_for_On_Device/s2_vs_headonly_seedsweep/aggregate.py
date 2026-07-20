import pandas as pd, numpy as np, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "results/seed_*.csv")))],
              ignore_index=True)
head = d[d.recipe == "head"].sort_values("seed")
s2 = d[d.recipe == "s2"].sort_values("seed")
seeds = sorted(d.seed.unique())

print("per-seed mean(b2-5):  seed  head    s2   (s2-head)")
diffs = []
for s in seeds:
    h = head[head.seed == s].mean_b2_5.values[0]; v = s2[s2.seed == s].mean_b2_5.values[0]
    diffs.append(v - h)
    print(f"                       {s:>3}  {h:6.2f} {v:6.2f}   {v-h:+6.2f}")
diffs = np.array(diffs)
hm = head.mean_b2_5.values; sm = s2.mean_b2_5.values

print("\n=== aggregate over %d seeds (mean(b2-5)) ===" % len(seeds))
print(f"head-only : {hm.mean():.2f} ± {hm.std(ddof=1):.2f}   range [{hm.min():.1f}, {hm.max():.1f}]")
print(f"S2 (full) : {sm.mean():.2f} ± {sm.std(ddof=1):.2f}   range [{sm.min():.1f}, {sm.max():.1f}]")
print(f"paired diff (S2 - head): {diffs.mean():+.2f} ± {diffs.std(ddof=1):.2f}   "
      f"S2 wins {int((diffs>0).sum())}/{len(diffs)}")
# paired t-test (manual)
t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
print(f"paired t = {t:.2f} (|t|<2.26 => not significant at p=0.05, df=9)")

# per-batch mean over seeds (avg of the 3-fold means)
print("\nper-batch mean over seeds (3-fold-mean averaged over seeds):")
print(f"{'batch':>5} {'head':>8} {'s2':>8}")
for b in ["b2", "b3", "b4", "b5"]:
    print(f"{b:>5} {head[b].mean():>8.2f} {s2[b].mean():>8.2f}")

summ = pd.DataFrame([
    dict(recipe="head_only", mean=round(hm.mean(), 2), std=round(hm.std(ddof=1), 2),
         min=round(hm.min(), 2), max=round(hm.max(), 2)),
    dict(recipe="s2_frozenstat_full", mean=round(sm.mean(), 2), std=round(sm.std(ddof=1), 2),
         min=round(sm.min(), 2), max=round(sm.max(), 2)),
    dict(recipe="paired_diff_s2_minus_head", mean=round(diffs.mean(), 2), std=round(diffs.std(ddof=1), 2),
         min=round(diffs.min(), 2), max=round(diffs.max(), 2)),
])
summ["s2_wins"] = [None, None, f"{int((diffs>0).sum())}/{len(diffs)}"]
summ.to_csv(os.path.join(HERE, "results/seedsweep_summary.csv"), index=False)
print("\nsaved -> results/seedsweep_summary.csv")
