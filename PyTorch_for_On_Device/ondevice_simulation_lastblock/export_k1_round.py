# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Export ONE round of the K=1 (last-block + fc) on-device training fixture via Onnx4Deeploy's `custom`
training strategy: freeze blocks 0-3, train block-4 (conv W/b + BN gamma/beta) + fc (W/b). BN kept
SEPARATE (not folded) with frozen running stats. Then sanity-check the exported ORT reference against a
PyTorch K=1 run on the SAME 54 windows — if they match, the export's BN handling is frozen-stat-correct
and the chain can be trusted; if not, block-4 BN is training in live-batch mode (the batch-1 problem).

CLI:  python3 export_k1_round.py --batch 1 --carry <pretrained.pt> --out /app/.../speechnet_train_k1_b1_fold3
"""
import argparse, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "exp13_progressive_unfreeze"))
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter   # noqa: E402
from adabn_full_training import make_model                             # noqa: E402
from windowing import load_windows                                     # noqa: E402
from ondevice_ft import balanced_accuracy                             # noqa: E402
from run_progressive_unfreeze import train_lastk                       # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
K1_PARAMS = ["blocks_4_0_weight", "blocks_4_0_bias", "blocks_4_1_weight", "blocks_4_1_bias",
             "fc_weight", "fc_bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--carry", required=True)     # pretrained/carry .pt
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ex = SpeechNetExporter(save_path=a.out)
    ex._config_overrides = {
        "dataset": "silentwear", "data_path": DATA,
        "subject": "S01", "session": 3, "batch": a.batch, "condition": "vocalized",
        "stratified_sampling": True, "data_size": 54,
        "n_batches": 2160, "n_accum": 4, "learning_rate": 1e-3,
        "training_strategy": "custom", "custom_trainable_params": K1_PARAMS,
        "fold_bn": False, "bn_frozen_stats": True, "pretrained_weights": a.carry,
    }
    ex.export_training()
    print(f"[export] K=1 fixture written to {a.out}", flush=True)

    # --- sanity: does the ORT reference match PyTorch K=1 on the same 54 windows? ---
    out = np.load(os.path.join(a.out, "outputs.npz"))
    d = np.load(os.path.join(a.out, "inputs.npz"))
    Xs = [d["arr_0000"]]; ys = [np.atleast_1d(d["arr_0001"])]
    for i in range(1, 54):
        Xs.append(d[f"mb{i}_arr_0000"]); ys.append(np.atleast_1d(d[f"mb{i}_arr_0001"]))
    X = np.concatenate(Xs, 0).astype(np.float32); y = np.concatenate(ys).astype(np.int64)

    # ORT-trained weights -> eval on batch+1 (eval-mode/frozen-BN inference)
    m_ort = make_model(a.carry)
    with torch.no_grad():
        for nm, p in [("blocks.4.0.weight", "blocks_4_0_weight"), ("blocks.4.0.bias", "blocks_4_0_bias"),
                      ("blocks.4.1.weight", "blocks_4_1_weight"), ("blocks.4.1.bias", "blocks_4_1_bias"),
                      ("fc.weight", "fc_weight"), ("fc.bias", "fc_bias")]:
            dict(m_ort.named_parameters())[nm].copy_(torch.from_numpy(out[p]))
    eb = a.batch + 1
    if eb <= 5:
        Xe, ye = load_windows(DATA, "S01", 3, eb, "vocalized", downsample_rest=True)
        acc_ort = balanced_accuracy(m_ort, Xe, ye)
        # PyTorch K=1 on the SAME 54 windows
        m_pt = make_model(a.carry)
        train_lastk(m_pt, X, y, 1e-3, 1)
        acc_pt = balanced_accuracy(m_pt, Xe, ye)
        print(f"[check b{eb}] ORT-ref={acc_ort:.2f}  PyTorch-K1={acc_pt:.2f}  "
              f"{'MATCH (frozen-stat BN ok)' if abs(acc_ort-acc_pt) < 1.0 else 'MISMATCH -> BN training mode differs'}",
              flush=True)
        print(f"[loss] ORT ref first={out['loss'][0]:.3f} last={out['loss'][-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
