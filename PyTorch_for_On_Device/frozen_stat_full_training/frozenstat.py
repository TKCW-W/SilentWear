# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Full-model FT with PRETRAINED BN stats — two settings, on-device SGD recipe. See PLAN.md.

S1 (live) : train-forward uses batch-1 window stats (BN train mode), running stats NOT updated
            (momentum=0) -> inference uses frozen PRETRAINED stats. (faithful on-device naive)
S2 (frozen): BN uses frozen PRETRAINED stats for BOTH train forward and inference (BN eval mode).

Both: full model (conv + BN gamma,beta + fc), SGD no-momentum, batch-1 + n_accum SUM accumulation,
fixed lr, 40 ep, 30% data. Sweep n_accum x lr (b1->b2, 3 folds); full b1->b5 incremental at best.

CLI:
  python3 frozenstat.py sweep {s1|s2} 8:0.0003 4:0.001 ... --out sweep_s1.csv
  python3 frozenstat.py incremental {s1|s2} <n_accum> <lr>
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)   # reuse parent-dir modules

from idea2_alt_norm import SpeechNetNorm      # noqa: E402
from windowing import load_windows, stratified_draw  # noqa: E402
from ondevice_ft import balanced_accuracy     # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]
EPOCHS = 40


def make_model(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=0.0); m.load_state_dict(sd); m.eval()
    return m


def full_train(model, X, y, setting, lr, n_accum, epochs=EPOCHS, seed=42):
    """Full-model SGD, batch-1 + n_accum SUM. setting 's1'=live-train/frozen-infer, 's2'=frozen both."""
    if setting == "s1":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.0                 # running stats stay at pretrained (never updated)
    for p in model.parameters():
        p.requires_grad = True                   # full model (conv + BN affine + fc)
    opt = torch.optim.SGD(model.parameters(), lr=lr)   # no momentum / no wd
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        if setting == "s1":
            model.train()          # BN forward uses batch-1 window stats; running frozen (momentum 0)
        else:
            model.eval()           # S2: BN forward uses frozen pretrained stats
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1   # SUM accumulation
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    model.eval()                   # inference: frozen pretrained running stats (both settings)
    return model


# ---- sweep (b1->b2) ----
def run_config(setting, n_accum, lr):
    accs = []
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xtr, ytr = stratified_draw(Xb1, yb1, 6, seed=42)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        m = make_model(ckpt)
        full_train(m, Xtr, ytr, setting, lr, n_accum)
        accs.append(balanced_accuracy(m, Xb2, yb2))
    return np.mean(accs), np.std(accs, ddof=1)


# ---- full incremental (b1->b5) ----
def run_incremental(setting, n_accum, lr):
    byb = {b: {"nf": [], "zs": []} for b in range(1, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        base = make_model(ckpt)
        m = make_model(ckpt)
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
            byb[b]["nf"].append(balanced_accuracy(base, Xe, ye))
            byb[b]["zs"].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                Xtr, ytr = stratified_draw(Xe, ye, 6, seed=42)
                full_train(m, Xtr, ytr, setting, lr, n_accum)
        print(f"  [{setting}] fold {fold} done", flush=True)
    import pandas as pd
    rows = []
    for b in range(1, 6):
        nf = np.array(byb[b]["nf"]); zs = np.array(byb[b]["zs"])
        rows.append(dict(batch=b, no_ft_mean=round(nf.mean(), 2), no_ft_std=round(nf.std(ddof=1), 2),
                         ft_mean=round(zs.mean(), 2), ft_std=round(zs.std(ddof=1), 2)))
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", f"ft_summary_{setting}_S01_vocalized_SUBJECT_MEAN.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"mean(b2-5) ft = {np.mean([r['ft_mean'] for r in rows if r['batch']>=2]):.2f}")
    print(f"saved -> {out}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sweep"); sp.add_argument("setting"); sp.add_argument("combos", nargs="+"); sp.add_argument("--out", required=True)
    ip = sub.add_parser("incremental"); ip.add_argument("setting"); ip.add_argument("n_accum", type=int); ip.add_argument("lr", type=float)
    a = ap.parse_args()
    import pandas as pd
    if a.cmd == "sweep":
        csv = os.path.join(HERE, "results", a.out)
        for combo in a.combos:
            na, lr = combo.split(":"); na = int(na); lr = float(lr)
            mean, std = run_config(a.setting, na, lr)
            row = dict(setting=a.setting, n_accum=na, lr=lr, eff_lr=round(na * lr, 5),
                       mean=round(mean, 2), std=round(std, 2))
            if os.path.exists(csv):
                df = pd.read_csv(csv); df = df[~((df.n_accum == na) & (df.lr == lr))]
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_csv(csv, index=False)
            print(f"[{a.setting}] n_accum={na:2d} lr={lr:<7g} eff={na*lr:<7g} | {mean:5.1f}±{std:4.1f}", flush=True)
    else:
        run_incremental(a.setting, a.n_accum, a.lr)


if __name__ == "__main__":
    main()
