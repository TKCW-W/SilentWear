import pandas as pd, numpy as np, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECTS = ["S01", "S02", "S03", "S04"]


def paper_ft(subj):
    p = f"{ART}/inter_session_ft/{subj}/vocalized/speechnet/w1400ms/ft_config_0/ft_summary.csv"
    d = pd.read_csv(p)
    return {b: d[d.zero_shot_test_batch == b].zero_shot_balanced_acc.mean() * 100 for b in range(2, 6)}


rows = []
per_batch = {}
for subj in SUBJECTS:
    f = os.path.join(HERE, f"results/streaming_incr_{subj}.csv")
    if not os.path.exists(f):
        print(f"[missing] {subj}"); continue
    d = pd.read_csv(f).set_index("batch")
    pf = paper_ft(subj)
    ours = {b: d.loc[b, "streaming_adabn_mean"] for b in range(2, 6)}
    noft = {b: d.loc[b, "no_ft_mean"] for b in range(2, 6)}
    per_batch[subj] = (noft, ours, pf)
    o25 = np.mean([ours[b] for b in range(2, 6)])
    p25 = np.mean([pf[b] for b in range(2, 6)])
    n25 = np.mean([noft[b] for b in range(2, 6)])
    rows.append(dict(subject=subj, no_ft=round(n25, 2), streaming_adabn=round(o25, 2),
                     paper=round(p25, 2), ours_minus_paper=round(o25 - p25, 2),
                     ours_gain_over_noft=round(o25 - n25, 2)))

out = pd.DataFrame(rows)
# 4-subject average row
avg = dict(subject="MEAN(4subj)", no_ft=round(out.no_ft.mean(), 2),
           streaming_adabn=round(out.streaming_adabn.mean(), 2), paper=round(out.paper.mean(), 2),
           ours_minus_paper=round((out.streaming_adabn - out.paper).mean(), 2),
           ours_gain_over_noft=round((out.streaming_adabn - out.no_ft).mean(), 2))
out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
out.to_csv(os.path.join(HERE, "results/streaming_4subj_vs_paper.csv"), index=False)

print("=== Streaming AdaBN vs paper — vocalized, mean over batches 2-5 (3 folds each) ===")
print(out.to_string(index=False))
print("\n=== per-batch, streaming AdaBN (S / paper P) ===")
print(f"{'batch':>5}  " + "  ".join(f"{s:>13}" for s in SUBJECTS))
for b in range(2, 6):
    cells = []
    for s in SUBJECTS:
        if s in per_batch:
            _, ours, pf = per_batch[s]; cells.append(f"{ours[b]:5.1f}/{pf[b]:5.1f}")
        else:
            cells.append("   -/-   ")
    print(f"{b:>5}  " + "  ".join(f"{c:>13}" for c in cells))
print("saved -> results/streaming_4subj_vs_paper.csv")
