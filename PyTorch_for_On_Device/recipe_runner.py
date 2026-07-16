# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Parameterized incremental inter-session FT runner for the factor-attribution study.
Each experiment is one config; flips one paper setting to its on-device value (see ABLATION_PLAN.md).

Usage:  python3 recipe_runner.py P0 E1 E3 ...        # runs the named configs
Appends one row per config to results/ablation_results.csv.
"""
import argparse, copy, os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from idea2_alt_norm import SpeechNetNorm
from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
ART = "/app/SilentWear/SilentWear/artifacts/models"
SUBJECT, COND, FOLDS = "S01", "vocalized", [1, 2, 3]

P0 = dict(scope="full", opt="adam", lr=1e-3, wd=1e-4, momentum=0.0, sched="plateau",
          bs=32, bn="live", dropout=0.5, data="paper", epochs=50, patience=10)
CONFIGS = {
    "P0": P0,
    "E1": {**P0, "opt": "sgd", "lr": 0.01, "wd": 0.0, "momentum": 0.0},
    "E2": {**P0, "sched": None},
    "E3": {**P0, "bs": 1},
    "E4": {**P0, "bn": "frozen"},
    "E5": {**P0, "scope": "head"},
    "E6": {**P0, "dropout": 0.0},
    "E7": {**P0, "data": "ondevice", "epochs": 40, "sched": None},
    "ODE": dict(scope="head", opt="sgd", lr=0.01, wd=0.0, momentum=0.0, sched=None,
                bs=1, bn="frozen", dropout=0.0, data="ondevice", epochs=40, patience=10),
}


def make_model(ckpt, dropout):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
    m = SpeechNetNorm(norm="bn", p_dropout=dropout); m.load_state_dict(sd); m.eval()
    return m


def strat_split(X, y, frac, seed):
    rng = np.random.default_rng(seed); tr, va = [], []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = max(1, int(round(len(idx) * frac))); tr += list(idx[:n]); va += list(idx[n:])
    tr, va = np.array(tr), np.array(va if len(va) else tr)
    return X[tr], y[tr], X[va], y[va]


def ft(model, X, y, cfg, seed):
    if cfg["data"] == "paper":
        Xtr, ytr, Xva, yva = strat_split(X, y, 0.7, seed); has_val = True
        Xva_t = torch.from_numpy(Xva.astype(np.float32)); yva_t = torch.from_numpy(yva.astype(np.int64))
    else:
        Xtr, ytr = stratified_draw(X, y, 6, seed); has_val = False
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)); ytr_t = torch.from_numpy(ytr.astype(np.int64))

    if cfg["scope"] == "head":
        for p in model.parameters(): p.requires_grad = False
        params = [model.fc.weight, model.fc.bias]
        for p in params: p.requires_grad = True
    else:
        for p in model.parameters(): p.requires_grad = True
        params = list(model.parameters())

    if cfg["opt"] == "adam":
        opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["wd"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
             if cfg["sched"] == "plateau" and has_val else None)
    crit = nn.CrossEntropyLoss()

    def set_modes():
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and (cfg["bn"] == "frozen" or cfg["scope"] == "head"):
                m.eval()

    N = len(ytr); bs = cfg["bs"]; best, best_sd, bad = 1e9, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        set_modes(); perm = torch.randperm(N, generator=g)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            opt.zero_grad(); crit(model(Xtr_t[b]), ytr_t[b]).backward(); opt.step()
        if has_val:
            model.eval()
            with torch.no_grad(): vl = float(crit(model(Xva_t), yva_t))
            if sched: sched.step(vl)
            if vl < best - 1e-4: best, best_sd, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= cfg["patience"]: break
    if best_sd: model.load_state_dict(best_sd)
    model.eval(); return model


def run_config(name):
    cfg = CONFIGS[name]
    byb = {b: {"nf": [], "zs": []} for b in range(1, 6)}
    for fold in FOLDS:
        ckpt = (f"{ART}/inter_session_ft/{SUBJECT}/{COND}/speechnet/w1400ms/model_1/"
                f"leave_one_session_out_fold_{fold}.pt")
        base = make_model(ckpt, cfg["dropout"])
        m = make_model(ckpt, cfg["dropout"])
        for b in range(1, 6):
            Xe, ye = load_windows(DATA, SUBJECT, fold, b, COND, downsample_rest=True)
            byb[b]["nf"].append(balanced_accuracy(base, Xe, ye))
            byb[b]["zs"].append(balanced_accuracy(m, Xe, ye))
            if b != 5:
                m = ft(m, Xe, ye, cfg, seed=1000 + fold * 10 + b)
        print(f"  [{name}] fold {fold} done", flush=True)
    per_batch = {b: (np.mean(byb[b]["zs"]), np.std(byb[b]["zs"], ddof=1)) for b in range(1, 6)}
    mean_ft = np.mean([per_batch[b][0] for b in range(2, 6)])
    return per_batch, mean_ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="ablation_results.csv")
    args = ap.parse_args()
    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    csv = os.path.join(outdir, args.out)
    import pandas as pd
    for name in args.configs:
        pb, mft = run_config(name)
        row = dict(config=name,
                   **{f"b{b}_mean": round(pb[b][0], 2) for b in range(2, 6)},
                   **{f"b{b}_std": round(pb[b][1], 2) for b in range(2, 6)},
                   mean_ft_2_5=round(mft, 2))
        if os.path.exists(csv):
            df = pd.read_csv(csv); df = df[df["config"] != name]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(csv, index=False)
        print(f"[{name}] mean_ft(2-5) = {mft:.2f}  (b2-5: "
              + " ".join(f"{pb[b][0]:.1f}" for b in range(2, 6)) + ")", flush=True)


if __name__ == "__main__":
    main()
