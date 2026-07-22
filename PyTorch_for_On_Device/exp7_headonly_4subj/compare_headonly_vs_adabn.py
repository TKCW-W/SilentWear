import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
PAR = os.path.dirname(HERE)
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJ = ["S01", "S02", "S03", "S04"]


def paper_ft(subj):
    d = pd.read_csv(f"{ART}/inter_session_ft/{subj}/vocalized/speechnet/w1400ms/ft_config_0/ft_summary.csv")
    return np.mean([d[d.zero_shot_test_batch == b].zero_shot_balanced_acc.mean() * 100 for b in range(2, 6)])


rows = []
pb_head = {b: [] for b in range(2, 6)}; pb_ada = {b: [] for b in range(2, 6)}; pb_nf = {b: [] for b in range(2, 6)}
for s in SUBJ:
    h = pd.read_csv(os.path.join(HERE, f"results/headonly_{s}.csv")).set_index("batch")
    a = pd.read_csv(os.path.join(PAR, f"exp5_streaming_adabn_incremental/results/streaming_incr_{s}.csv")).set_index("batch")
    hh = np.mean([h.loc[b, "headonly_mean"] for b in range(2, 6)])
    aa = np.mean([a.loc[b, "streaming_adabn_mean"] for b in range(2, 6)])
    nf = np.mean([h.loc[b, "no_ft_mean"] for b in range(2, 6)])
    pf = paper_ft(s)
    rows.append(dict(subject=s, no_ft=round(nf, 2), head_only=round(hh, 2),
                     streaming_adabn=round(aa, 2), paper=round(pf, 2),
                     adabn_minus_head=round(aa - hh, 2)))
    for b in range(2, 6):
        pb_head[b].append(h.loc[b, "headonly_mean"]); pb_ada[b].append(a.loc[b, "streaming_adabn_mean"])
        pb_nf[b].append(h.loc[b, "no_ft_mean"])

out = pd.DataFrame(rows)
avg = dict(subject="MEAN", no_ft=round(out.no_ft.mean(), 2), head_only=round(out.head_only.mean(), 2),
           streaming_adabn=round(out.streaming_adabn.mean(), 2), paper=round(out.paper.mean(), 2),
           adabn_minus_head=round((out.streaming_adabn - out.head_only).mean(), 2))
out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
out.to_csv(os.path.join(HERE, "results/headonly_vs_adabn_4subj.csv"), index=False)
print("=== Head-only vs streaming AdaBN vs paper — vocalized, mean b2-5 (3 folds) ===")
print(out.to_string(index=False))
print("\n=== per-batch, 4-subject avg ===")
print(f"{'batch':>5} {'no_ft':>7} {'head_only':>10} {'streaming_adabn':>16} ")
for b in range(2, 6):
    print(f"{b:>5} {np.mean(pb_nf[b]):>7.2f} {np.mean(pb_head[b]):>10.2f} {np.mean(pb_ada[b]):>16.2f}")
print("saved -> results/headonly_vs_adabn_4subj.csv")
