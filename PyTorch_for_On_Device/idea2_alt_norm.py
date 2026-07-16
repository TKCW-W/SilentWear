# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
IDEA 2 — replace BatchNorm with a batch-independent normalization for the on-device setting.

BatchNorm needs batch statistics -> degenerate at the on-device effective-batch-1, which forces
head-only FT. GroupNorm / LayerNorm / InstanceNorm normalize **per sample** (batch-independent),
so they behave identically at batch 1 and any batch, and allow **full-model** on-device FT.

The checkpoints are BN-trained, so each norm variant is trained FROM SCRATCH here with the SAME
recipe (only the norm layer differs) -> fair relative comparison. We then compare, per fold:
  - base zero-shot balanced acc on batch 2 (does the norm reach comparable accuracy?)
  - on-device FT on 30% of batch 1 -> batch 2:
        BN     -> head-only (its on-device constraint)
        GN/LN/IN -> FULL model (enabled by batch-independent norm)

S01 vocalized, 3 folds. Train sessions = the two non-held-out sessions of each fold.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # batch-1 FT loops: single-thread avoids thread-thrash overhead

from windowing import load_windows, stratified_draw
from ondevice_ft import balanced_accuracy

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
SUBJECT, COND = "S01", "vocalized"
FOLDS = [1, 2, 3]
ALL_SESSIONS = [1, 2, 3]
BLOCKS = [(8, (1, 4), (1, 8)), (16, (1, 16), (1, 4)), (16, (1, 8), (1, 4)),
          (32, (7, 1), (1, 1)), (32, (7, 1), (1, 1))]


def make_norm(kind, C, groups=4):
    if kind == "bn":
        return nn.BatchNorm2d(C)
    if kind == "gn":
        return nn.GroupNorm(groups, C)
    if kind == "ln":
        return nn.GroupNorm(1, C)          # LayerNorm over (C,H,W) per sample
    if kind == "in":
        return nn.GroupNorm(C, C)          # InstanceNorm (per-channel per-sample)
    raise ValueError(kind)


class SpeechNetNorm(nn.Module):
    def __init__(self, norm="bn", num_classes=9, p_dropout=0.5, groups=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = 1
        for out_ch, (kc, kt), (pc, pt) in BLOCKS:
            pool = nn.Identity() if (pc == 1 and pt == 1) else \
                nn.MaxPool2d((pc, pt), (pc, pt))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (kc, kt), (1, 1), (0, kt // 2), bias=True),
                make_norm(norm, out_ch, groups), nn.ReLU(inplace=False), pool))
            in_ch = out_ch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc_in = in_ch
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.global_pool(x).reshape(x.shape[0], self._fc_in)
        return self.fc(self.dropout(x))


def load_train_data(fold, seed=42):
    """All rest-downsampled windows from the two non-held-out sessions (all 5 batches)."""
    Xs, ys = [], []
    for s in ALL_SESSIONS:
        if s == fold:
            continue
        for b in range(1, 6):
            X, y = load_windows(DATA, SUBJECT, s, b, COND, downsample_rest=True, seed=seed)
            Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def train_from_scratch(norm, X, y, epochs=60, lr=1e-3, wd=1e-4, bs=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(y); idx = np.random.RandomState(seed).permutation(n)
    nval = max(bs, int(0.2 * n)); va, tr = idx[:nval], idx[nval:]
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    model = SpeechNetNorm(norm=norm)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5)
    crit = nn.CrossEntropyLoss()
    best_acc, best_sd = -1.0, None
    for _ in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad(); crit(model(Xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xt[va]).argmax(1).numpy()
        vacc = float(np.mean([(vp[y[va] == c] == c).mean()
                              for c in sorted(set(y[va].tolist()))]))
        sched.step(vacc)
        if vacc > best_acc:
            best_acc = vacc; best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    return model


def finetune_ondevice(model, X, y, scope, lr=0.01, n_accum=4, epochs=40):
    """Our on-device recipe: SGD sum-accum, eff-batch 1. scope 'head' or 'full'.
    Norm layers stay in eval() (for GN/LN/IN that is identical to train() — batch-independent)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if scope == "head":
        params = [model.fc.weight, model.fc.bias]
    else:
        params = [p for p in model.parameters()]
        for p in params:
            p.requires_grad = True
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr); crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    N = len(y)
    for _ in range(epochs):
        opt.zero_grad(); c = 0
        for j in range(N):
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward(); c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()
        if c % n_accum:
            opt.step(); opt.zero_grad()
    return model


def main():
    norms = ["bn", "gn", "ln", "in"]
    zs = {n: [] for n in norms}
    ft = {n: [] for n in norms}
    for fold in FOLDS:
        Xtr_all, ytr_all = load_train_data(fold)
        Xb1, yb1 = load_windows(DATA, SUBJECT, fold, 1, COND, downsample_rest=True)
        Xft, yft = stratified_draw(Xb1, yb1, per_class=6, seed=42)
        Xb2, yb2 = load_windows(DATA, SUBJECT, fold, 2, COND, downsample_rest=True)
        for norm in norms:
            base = train_from_scratch(norm, Xtr_all, ytr_all)
            zs[norm].append(balanced_accuracy(base, Xb2, yb2))
            scope = "head" if norm == "bn" else "full"     # BN: head-only; others: full
            import copy
            m = copy.deepcopy(base)
            finetune_ondevice(m, Xft, yft, scope)
            ft[norm].append(balanced_accuracy(m, Xb2, yb2))
            print(f"fold {fold} {norm}: zs={zs[norm][-1]:.1f}  ft({scope})={ft[norm][-1]:.1f}")

    print("\n=== IDEA 2: normalization layers — S01 vocalized, b1->b2, mean+-std over 3 folds ===")
    print(f"{'norm':<6} {'ft_scope':<8} {'base_zs':>16} {'after_ondevice_ft':>20}")
    for n in norms:
        a, b = np.array(zs[n]), np.array(ft[n])
        scope = "head" if n == "bn" else "full"
        print(f"{n:<6} {scope:<8} {a.mean():7.2f}+-{a.std(ddof=1):4.2f}   "
              f"{b.mean():9.2f}+-{b.std(ddof=1):4.2f}")
    print("\nnote: all norms trained from scratch with the SAME recipe (only norm differs);"
          " BN base_zs here is the in-experiment BN (not the official checkpoint 81.67%).")


if __name__ == "__main__":
    main()
