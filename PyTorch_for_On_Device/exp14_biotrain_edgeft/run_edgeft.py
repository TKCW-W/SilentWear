# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Exp 14 — reproduce BioTrain's "Edge-FT" deployment recipe and test whether it genuinely beats our
head-only recipe on SpeechNet / SilentWear.

BioTrain Edge-FT (biotrain_summary.md §7): replace every BatchNorm with GroupNorm, train the FULL
network (GN affine included), SGD momentum 0.9 + weight decay 1e-3 + cosine-annealing LR (init 5e-3),
30 epochs, effective batch 8 via single-sample gradient accumulation. My idea2_alt_norm screen showed
GN-full (80.0) < BN-head-only (87.78) but used PLAIN SGD — here we apply the true Edge-FT optimizer and
ablate its ingredients.

Protocol matches idea2_alt_norm (for direct comparability): S01 vocalized, pretrain each norm variant
from scratch on the 2 non-held-out sessions (LOSO), FT on 30% of batch 1 (6/class, seed 42), eval on
batch 2; mean +- std over 3 folds.
CLI:  python3 run_edgeft.py
"""
import os, sys, copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from idea2_alt_norm import (SpeechNetNorm, train_from_scratch, load_train_data,   # noqa: E402
                            finetune_ondevice, DATA, SUBJECT, COND, FOLDS)
from windowing import load_windows, stratified_draw                               # noqa: E402
from ondevice_ft import balanced_accuracy                                         # noqa: E402


def finetune_edgeft(model, X, y, lr=5e-3, n_accum=8, epochs=30, momentum=0.9, wd=1e-3,
                    cosine=True, meangrad=True):
    """BioTrain Edge-FT: full-network, GN (batch-independent so eval==train), SGD+momentum+wd+cosine,
    effective batch = n_accum via gradient accumulation. meangrad=True divides by n_accum (true batch-N)."""
    model.eval()                                              # GN: eval == train (batch-independent)
    for p in model.parameters():
        p.requires_grad = True                               # full network
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y); rng = np.random.default_rng(42)
    for _ in range(epochs):
        perm = rng.permutation(N); opt.zero_grad(); c = 0
        for j in perm:
            loss = crit(model(Xt[j:j + 1]), yt[j:j + 1])
            (loss / n_accum if meangrad else loss).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
        if sched:
            sched.step()
    model.eval(); return model


# FT configs: (label, norm, callable(base_model, Xft, yft) -> ft_model)
def cfgs():
    edge = lambda **kw: (lambda m, X, y: finetune_edgeft(m, X, y, **kw))
    return [
        ("BN head-only (ours)", "bn", lambda m, X, y: finetune_ondevice(m, X, y, "head", lr=0.01, n_accum=4, epochs=40)),
        ("GN full plain-SGD (idea2)", "gn", lambda m, X, y: finetune_ondevice(m, X, y, "full", lr=0.01, n_accum=4, epochs=40)),
        ("GN Edge-FT (mom0.9 wd1e-3 cos lr5e-3 b8)", "gn", edge()),
        ("GN Edge-FT lr1e-2", "gn", edge(lr=1e-2)),
        ("GN Edge-FT lr1e-3", "gn", edge(lr=1e-3)),
        ("GN Edge-FT no-momentum", "gn", edge(momentum=0.0)),
        ("GN Edge-FT no-wd", "gn", edge(wd=0.0)),
        ("GN Edge-FT no-cosine", "gn", edge(cosine=False)),
        ("GN Edge-FT sum-grad", "gn", edge(meangrad=False)),
    ]


def main():
    import pandas as pd
    CFGS = cfgs()
    acc = {lab: [] for lab, _, _ in CFGS}
    for fold in FOLDS:
        Xtr, ytr = load_train_data(fold)
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xft, yft = stratified_draw(Xb1, yb1, per_class=6, seed=42)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        base = {"bn": train_from_scratch("bn", Xtr, ytr), "gn": train_from_scratch("gn", Xtr, ytr)}
        for lab, norm, fn in CFGS:
            m = copy.deepcopy(base[norm])
            fn(m, Xft, yft)
            acc[lab].append(balanced_accuracy(m, Xb2, yb2))
        print(f"fold {fold} done", flush=True)
    rows = []
    print(f"\n=== exp14 BioTrain Edge-FT — S01 vocalized, b1->b2, mean+-std over 3 folds ===")
    print(f"{'config':<44} {'b2 acc':>14}")
    for lab, _, _ in CFGS:
        a = np.array(acc[lab]); rows.append(dict(config=lab, mean=round(a.mean(), 2), std=round(a.std(ddof=1), 2)))
        print(f"{lab:<44} {a.mean():6.2f} +- {a.std(ddof=1):4.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results_edgeft_S01.csv"), index=False)
    print("\nreference: idea2 BN-head 87.78, GN-plain 80.00 ; official BN head-only ~87.8", flush=True)


if __name__ == "__main__":
    main()
