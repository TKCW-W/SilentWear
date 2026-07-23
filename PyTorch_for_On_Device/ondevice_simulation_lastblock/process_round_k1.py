# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""
Process one completed K=1 (last-block+fc) GVSoC training round:
  1. parse the 6 device tensors from [WDUMP] (block-4 conv W/b + BN gamma/beta + fc W/b),
  2. match each dumped tensor to its ORT reference (outputs.npz) by size + value proximity
     (the 32-length conv-bias / BN-gamma / BN-beta are size-ambiguous, so match by value),
  3. validate device == ORT (< 1e-4),
  4. build the carry checkpoint = frozen pretrained backbone + device block-4 + device fc,
  5. evaluate on the next batch (bit-exact host inference; BN frozen at pretrained stats),
  6. save the carry checkpoint for the next round.

CLI: python3 process_round_k1.py --round 1 --fixture <...>/speechnet_train_k1_b1_fold3 \
       --gvsoc-log logs/round1_gvsoc_train.log --eval-batch 2 --out-carry /tmp/carry_k1_b1_fold3.pt
"""
import argparse, os, re, struct, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from adabn_full_training import make_model            # noqa: E402
from windowing import load_windows                    # noqa: E402
from ondevice_ft import balanced_accuracy             # noqa: E402

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRETRAINED = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
              "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
# ONNX initializer name -> PyTorch state-dict key for the 6 K=1 trainable tensors
NAME_MAP = {"blocks_4_0_weight": "blocks.4.0.weight", "blocks_4_0_bias": "blocks.4.0.bias",
            "blocks_4_1_weight": "blocks.4.1.weight", "blocks_4_1_bias": "blocks.4.1.bias",
            "fc_weight": "fc.weight", "fc_bias": "fc.bias"}


def parse_wdump(logpath):
    pat = re.compile(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)")
    dumps = {}
    for line in open(logpath):
        m = pat.search(line)
        if not m:
            continue
        s, wi = int(m.group(1)), int(m.group(2))
        arr = np.array([struct.unpack("<f", struct.pack("<I", int(w, 16)))[0]
                        for w in m.group(4).split()], dtype=np.float32)
        dumps[(s, wi)] = arr
    last = max(s for s, _ in dumps)
    return [dumps[(s, wi)] for (s, wi) in sorted(dumps) if s == last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--gvsoc-log", required=True)
    ap.add_argument("--eval-batch", type=int, required=True)
    ap.add_argument("--out-carry", required=True)
    a = ap.parse_args()
    logdir = os.path.join(HERE, "logs"); os.makedirs(logdir, exist_ok=True)

    dumped = parse_wdump(a.gvsoc_log)
    ref = np.load(os.path.join(a.fixture, "outputs.npz"))
    # match each dumped tensor to a K=1 reference tensor by size + value proximity
    device = {}
    for arr in dumped:
        best, bestd = None, 1e9
        for onnx_n in NAME_MAP:
            r = ref[onnx_n]
            if r.size == arr.size:
                d = float(np.max(np.abs(arr.reshape(r.shape) - r)))
                if d < bestd:
                    best, bestd = onnx_n, d
        device[best] = (arr.reshape(ref[best].shape), bestd)
    print(f"round {a.round}: matched {len(device)}/6 device tensors:", flush=True)
    ok = True
    for onnx_n in NAME_MAP:
        arr, d = device[onnx_n]
        v = "OK" if d < 1e-4 else "DIVERGES"; ok &= d < 1e-4
        print(f"    {onnx_n:18s} max|device-ORT|={d:.2e}  {v}", flush=True)
        np.save(os.path.join(logdir, f"device_r{a.round}_{onnx_n}.npy"), arr)
    print(f"  ALL VALID (device==ORT<1e-4)" if ok else "  WARNING: some tensors diverge", flush=True)

    base = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    st = base.get("model_state_dict", base)
    ns = {k: (v.clone() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in st.items()}
    for onnx_n, pt_n in NAME_MAP.items():
        ns[pt_n] = torch.from_numpy(device[onnx_n][0]).float()
    torch.save({"model_state_dict": ns}, a.out_carry)

    m = make_model(a.out_carry)
    Xe, ye = load_windows(DATA, "S01", 3, a.eval_batch, "vocalized", downsample_rest=True)
    acc = balanced_accuracy(m, Xe, ye)
    print(f"round {a.round}: FT through b{a.round} -> eval b{a.eval_batch} = {acc:.2f}  "
          f"(K=1 on-device, bit-exact inference)", flush=True)
    with open(os.path.join(logdir, f"round{a.round}_accuracy.txt"), "w") as f:
        f.write(f"round {a.round}: eval b{a.eval_batch} = {acc:.2f}\n")


if __name__ == "__main__":
    main()
