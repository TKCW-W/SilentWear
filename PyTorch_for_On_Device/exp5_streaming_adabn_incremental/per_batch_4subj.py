import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECTS = ["S01", "S02", "S03", "S04"]


def paper_per_batch(subj):
    d = pd.read_csv(f"{ART}/inter_session_ft/{subj}/vocalized/speechnet/w1400ms/ft_config_0/ft_summary.csv")
    return {b: d[d.zero_shot_test_batch == b].zero_shot_balanced_acc.mean() * 100 for b in range(1, 6)}


noft = {b: [] for b in range(1, 6)}
strm = {b: [] for b in range(1, 6)}
papr = {b: [] for b in range(1, 6)}
for subj in SUBJECTS:
    d = pd.read_csv(os.path.join(HERE, f"results/streaming_incr_{subj}.csv")).set_index("batch")
    pf = paper_per_batch(subj)
    for b in range(1, 6):
        noft[b].append(d.loc[b, "no_ft_mean"])
        strm[b].append(d.loc[b, "streaming_adabn_mean"])
        papr[b].append(pf[b])

rows = []
for b in range(1, 6):
    rows.append(dict(batch=b,
                     no_ft=round(np.mean(noft[b]), 2),
                     streaming_adabn=round(np.mean(strm[b]), 2),
                     streaming_std_subj=round(np.std(strm[b], ddof=1), 2),
                     paper=round(np.mean(papr[b]), 2),
                     ours_minus_paper=round(np.mean(strm[b]) - np.mean(papr[b]), 2),
                     gain_over_noft=round(np.mean(strm[b]) - np.mean(noft[b]), 2)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "results/streaming_per_batch_4subj.csv"), index=False)
print("=== Streaming AdaBN, per batch averaged across 4 subjects (vocalized) ===")
print(out.to_string(index=False))
print(f"\nmean over batches 2-5:  no_ft={np.mean([out[out.batch==b].no_ft.values[0] for b in range(2,6)]):.2f}  "
      f"streaming={np.mean([out[out.batch==b].streaming_adabn.values[0] for b in range(2,6)]):.2f}  "
      f"paper={np.mean([out[out.batch==b].paper.values[0] for b in range(2,6)]):.2f}")
