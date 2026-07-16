# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
On-device fine-tuning recipe (PyTorch host equivalent) + evaluation.

This is the hardware-constrained recipe run on the Siracusa/Deeploy target, reproduced on
the host. For the head-only + BN-fold path, PyTorch eval-mode BN (frozen running stats) is
mathematically identical to the folded frozen Conv used on device, and this host loop was
calibrated bit-exact to the on-device result (batch-2 FT). See
`TrainDeeploy/DeeployTest/experiments/headonly_ondevice_ft_fixedwindow/SPEECHNET_ONDEVICE_FT_CONFIG_SPEC.md`.

On-device FT configuration (defaults here):
  scope      = head only (fc = Linear 32->9); conv+BN frozen (folded feature extractor)
  optimizer  = plain SGD (no momentum, no weight decay)
  lr         = 0.01, static (no schedule/decay)
  n_accum    = 4, SUM accumulation: w <- w - lr * sum_{i=1..4} grad_i  (no /n_accum)
  eff. batch = 1 (one window per forward/backward)
  epochs     = 40 (fixed, no early stopping)
  dropout    = none
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def balanced_accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Mean per-class recall over the eval windows (batch-1 forward, eval mode)."""
    model.eval()
    preds = np.array([int(model(torch.from_numpy(X[i:i + 1])).argmax(1).item())
                      for i in range(len(y))])
    classes = sorted(set(y.tolist()))
    return 100.0 * float(np.mean([(preds[y == c] == c).mean() for c in classes]))


def finetune_head(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    n_accum: int = 4,
    epochs: int = 40,
) -> nn.Module:
    """Head-only + BN-fold SGD fine-tune with SUM gradient accumulation (in place).

    BN is kept in eval() (frozen running stats == folded frozen Conv). Only fc trains.
    Deterministic fixed data order (sum-accumulation is order-dependent).
    """
    model.eval()                                    # freeze BN running stats (folded-equiv)
    for p in model.parameters():
        p.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    opt = torch.optim.SGD([model.fc.weight, model.fc.bias], lr=lr)  # no momentum / no wd
    crit = nn.CrossEntropyLoss()
    N = len(y)
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    for _ in range(epochs):
        opt.zero_grad()
        c = 0
        for j in range(N):                          # fixed order, effective batch 1
            crit(model(Xt[j:j + 1]), yt[j:j + 1]).backward()   # grads accumulate (sum)
            c += 1
            if c % n_accum == 0:
                opt.step(); opt.zero_grad()         # w <- w - lr * sum-of-n_accum grads
        if c % n_accum:
            opt.step(); opt.zero_grad()             # flush trailing partial group
    return model
