import pandas as pd, numpy as np, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results/frac_*.csv"))],
              ignore_index=True)
rows = []
for subj in ["S01", "S02"]:
    for K in sorted(d[d.subject == subj].K.unique()):
        sub = d[(d.subject == subj) & (d.K == K)]
        # per-draw (seed) mean over folds x batches -> draw-to-draw variance of the config
        per_seed = sub.groupby("seed")["acc"].mean()
        rows.append(dict(subject=subj, K=int(K),
                         mean=round(per_seed.mean(), 2),
                         draw_std=round(per_seed.std(ddof=1), 2),
                         worst_draw=round(per_seed.min(), 2),
                         best_draw=round(per_seed.max(), 2)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "results/frac_summary.csv"), index=False)
for subj in ["S01", "S02"]:
    print(f"\n=== {subj} (recollect-only, mean b2-5 over 8 draws) ===")
    print(f"{'K':>4} {'mean':>7} {'draw_std':>9} {'worst':>7} {'best':>7}")
    for _, r in out[out.subject == subj].iterrows():
        print(f"{r.K:>4} {r['mean']:>7.2f} {r.draw_std:>9.2f} {r.worst_draw:>7.2f} {r.best_draw:>7.2f}")
print("\nsaved -> results/frac_summary.csv")
