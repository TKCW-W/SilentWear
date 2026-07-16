# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
SpeechNet — on-device (Deeploy) deployment variant, PyTorch host copy.

This is a self-contained copy of the model as deployed on the Siracusa/Deeploy target
(`Onnx4Deeploy/onnx4deeploy/models/pytorch_models/speechnet/speechnet.py`). It is
architecturally identical to the trained SilentWear SpeechNet checkpoints — it loads
`leave_one_session_out_fold_*.pt` with `strict=True` and reproduces the paper's balanced
accuracy exactly.

Differences from the SilentWear training model (all inference-neutral):
  - input is 4-D (B,1,C,T); no `x[:,None]` unsqueeze in forward.
  - the (1,1) "no pooling" blocks are `nn.Identity` (a 1x1 stride-1 pool IS the identity),
    so no degenerate MaxPool nodes are emitted on device.
  - no Dropout (the paper trained with p=0.5; inactive at inference; the on-device head-only
    FT recipe also omits it — see README).

Default config == the exact `blocks_config` from the checkpoints' run_cfg.json:
  Block0 Conv(1->8,  k=(1,4))  BN ReLU MaxPool(1,8)
  Block1 Conv(8->16, k=(1,16)) BN ReLU MaxPool(1,4)
  Block2 Conv(16->16,k=(1,8))  BN ReLU MaxPool(1,4)
  Block3 Conv(16->32,k=(7,1))  BN ReLU (no pool)
  Block4 Conv(32->32,k=(7,1))  BN ReLU (no pool)
  GlobalAvgPool -> Linear(32 -> num_classes)
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class SpeechNetOnDevice(nn.Module):
    def __init__(
        self,
        num_channels: int = 14,
        time_steps: int = 700,
        num_classes: int = 9,
        blocks_config: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.num_classes = num_classes

        if blocks_config is None:
            blocks_config = [
                dict(out_channels=8, kernel=(1, 4), pool=(1, 8)),
                dict(out_channels=16, kernel=(1, 16), pool=(1, 4)),
                dict(out_channels=16, kernel=(1, 8), pool=(1, 4)),
                dict(out_channels=32, kernel=(7, 1), pool=(1, 1)),
                dict(out_channels=32, kernel=(7, 1), pool=(1, 1)),
            ]

        self.blocks = nn.ModuleList()
        in_ch = 1
        for cfg in blocks_config:
            out_ch = int(cfg["out_channels"])
            k_c, k_t = int(cfg["kernel"][0]), int(cfg["kernel"][1])
            pool_c, pool_t = cfg.get("pool", (1, 1))
            pool_c, pool_t = int(pool_c), int(pool_t)
            if pool_c == 1 and pool_t == 1:
                pool_layer: nn.Module = nn.Identity()
            else:
                pool_layer = nn.MaxPool2d(kernel_size=(pool_c, pool_t), stride=(pool_c, pool_t))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(k_c, k_t), stride=(1, 1),
                          padding=(0, k_t // 2), bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
                pool_layer,
            ))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc_in = in_ch
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,C,T). Handles any batch size (host copy; the deployed graph is batch-1)."""
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], self._fc_in)
        return self.fc(x)


def load_speechnet(ckpt_path: str, num_channels: int = 14, time_steps: int = 700,
                   num_classes: int = 9) -> SpeechNetOnDevice:
    """Load a leave_one_session_out_fold_*.pt checkpoint (strict)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    m = SpeechNetOnDevice(num_channels, time_steps, num_classes)
    m.load_state_dict(sd)  # strict: architecture is identical to the checkpoint
    m.eval()
    return m
