# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch K=1 (last-block+fc, frozen pretrained BN) incremental chain on the SAME 54-window Onnx4Deeploy
draws the GVSoC chain uses (extracted from each fixture's inputs.npz, fixture order, block4+fc carried).
This is the matched-data PyTorch reference the on-device K=1 chain is compared against — removing the
stratified-draw confound (Onnx4Deeploy draw vs windowing.stratified_draw(seed=42)).

S01 vocalized fold 3. K=1: train block4 conv+BN-affine + fc, eval-mode (frozen) BN, lr 1e-3, n_accum 4,
40 ep, fixed order, incremental. Matches the GVSoC BN_FROZEN_STATS recipe.
CLI:  python3 run_pytorch_k1_ondevicedata_chain.py
"""
import os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "exp13_progressive_unfreeze"))
from adabn_full_training import make_model            # noqa: E402
from windowing import load_windows                    # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402
from run_progressive_unfreeze import train_lastk      # noqa: E402

SN = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
# per round r: (fixture with batch-r's 54 Onnx4Deeploy FT windows, eval batch, on-device GVSoC acc so far)
ROUNDS = [
    (1, f"{SN}/speechnet_train_k1_b1_fold3", 2, 87.78),
    (2, f"{SN}/speechnet_train_k1_b2_fold3", 3, None),
    (3, f"{SN}/speechnet_train_k1_b3_dataprobe", 4, None),
    (4, f"{SN}/speechnet_train_k1_b4_dataprobe", 5, None),
]


def extract_windows(fixture):
    d = np.load(os.path.join(fixture, "inputs.npz"))
    Xs = [d["arr_0000"]]; ys = [np.atleast_1d(d["arr_0001"])]
    for i in range(1, 54):
        Xs.append(d[f"mb{i}_arr_0000"]); ys.append(np.atleast_1d(d[f"mb{i}_arr_0001"]))
    return np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys).astype(np.int64)


def main():
    import pandas as pd
    m = make_model(PRETRAINED)
    Xb1, yb1 = load_windows(DATA, "S01", 3, 1, "vocalized", downsample_rest=True)
    rows = [dict(batch=1, note="zero-shot", pytorch_k1=round(balanced_accuracy(m, Xb1, yb1), 2),
                 ondevice_gvsoc=80.56)]
    print(f"{'batch':>5} {'PyTorch K=1 (on-device data)':>28} {'on-device GVSoC':>16}")
    print(f"{1:>5} {rows[0]['pytorch_k1']:>28.2f} {80.56:>16.2f}   (zero-shot)")
    for r, fixture, eb, dev in ROUNDS:
        X, y = extract_windows(fixture)                       # exact Onnx4Deeploy 54 windows for batch r
        train_lastk(m, X, y, 1e-3, 1)                         # K=1: last block + fc, frozen BN, carry
        Xe, ye = load_windows(DATA, "S01", 3, eb, "vocalized", downsample_rest=True)
        acc = balanced_accuracy(m, Xe, ye)
        rows.append(dict(batch=eb, note=f"FT through b{r}", pytorch_k1=round(acc, 2), ondevice_gvsoc=dev))
        print(f"{eb:>5} {acc:>28.2f} {('%.2f' % dev) if dev else '(pending)':>16}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "pytorch_k1_ondevicedata.csv"), index=False)
    print("\nsaved pytorch_k1_ondevicedata.csv", flush=True)


if __name__ == "__main__":
    main()
