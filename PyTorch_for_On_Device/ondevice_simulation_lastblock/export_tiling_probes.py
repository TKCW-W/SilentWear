# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""Export two probe fixtures to isolate the K=1 tiling infeasibility:
  A) K=1 custom strategy with LIVE BN (bn_frozen_stats=False)  -> tiles? -> if yes, freeze is the cause
  B) FULL strategy with FROZEN BN   (bn_frozen_stats=True)     -> fails? -> if yes, frozen BN is the cause
Fast export (n_batches=4) — only the graph structure matters for tiling."""
import sys
sys.path.insert(0, "/app/Onnx4Deeploy")
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter

DATA = "/app/SilentWear/SilentWear_data/data_raw_and_filt"
PRE = ("/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/"
       "speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt")
SN = "/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet"
K1 = ["blocks_4_0_weight", "blocks_4_0_bias", "blocks_4_1_weight", "blocks_4_1_bias", "fc_weight", "fc_bias"]


def export(out, strategy, custom_params, bn_frozen):
    ex = SpeechNetExporter(save_path=out)
    ex._config_overrides = {
        "dataset": "silentwear", "data_path": DATA, "subject": "S01", "session": 3, "batch": 1,
        "condition": "vocalized", "stratified_sampling": True, "data_size": 54,
        "n_batches": 54, "n_accum": 4, "learning_rate": 1e-3,
        "training_strategy": strategy, "custom_trainable_params": custom_params,
        "fold_bn": False, "bn_frozen_stats": bn_frozen, "pretrained_weights": PRE,
    }
    ex.export_training()
    print(f"[probe] exported {out}  (strategy={strategy}, bn_frozen_stats={bn_frozen})", flush=True)


# A: K=1 custom, LIVE BN
export(f"{SN}/speechnet_train_k1_livebn_probe", "custom", K1, False)
# B: FULL, FROZEN BN
export(f"{SN}/speechnet_train_full_frozenbn_probe", "full", [], True)
