"""Bounded ACT latent/dropout probes and data-coverage checks; no optimizer or source edits."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.constants import OBS_IMAGES

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    checkpoint = ROOT / "outputs/act20261023/fixed_scene_train/train/checkpoints/005000/pretrained_model"
    dataset_path = ROOT / "data/fixed_scene_seed200"
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda"
    cfg.pretrained_backbone_weights = None
    policy = get_policy_class("act").from_pretrained(checkpoint, config=cfg).cuda().eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint,
                                        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    ds = LeRobotDataset("local/fixed_scene", root=dataset_path, video_backend="pyav",
                        delta_timestamps={"action": [i / 33 for i in range(cfg.chunk_size)]})
    offsets = [0,1,2,5,10,20,30,60,100,104,127,167,300,500,700,1000,1200]
    items = [ds[i] for i in offsets]
    results = []
    with torch.inference_mode():
        for i, item in zip(offsets, items):
            target = item["action"].numpy().copy()
            batched = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                       for key, value in item.items()}
            batch = _preprocess_dataset_batch(batched, ds.meta.camera_keys, {}, pre)
            model_batch = dict(batch)
            model_batch[OBS_IMAGES] = [batch[key] for key in cfg.image_features]
            prior_zero = policy.model(model_batch)[0]
            # Enable only the root's VAE branch; all dropout submodules stay in eval.
            policy.model.training = True
            _, (mu, logvar) = policy.model(model_batch)
            policy.model.training = False

            def decode(z):
                hook = policy.model.encoder_latent_input_proj.register_forward_pre_hook(lambda module, inputs: (z,))
                try:
                    return policy.model(model_batch)[0]
                finally:
                    hook.remove()

            variants = {"prior_zero": [prior_zero], "posterior_mean": [decode(mu)]}
            variants["prior_samples"] = [decode(torch.randn_like(mu)) for _ in range(8)]
            variants["posterior_samples"] = [decode(mu + torch.exp(logvar/2)*torch.randn_like(mu)) for _ in range(8)]
            policy.train()
            variants["training_dropout_posterior"] = [policy.model(model_batch)[0] for _ in range(8)]
            policy.eval()
            if i in [0, 10, 100, 500]:
                # Isolate dropout from the action-conditioned VAE branch.
                dropouts = [m for m in policy.modules() if isinstance(m, torch.nn.Dropout)]
                attentions = [m for m in policy.modules() if isinstance(m, torch.nn.MultiheadAttention)]
                for mode in ["dropout_only_prior_zero", "feedforward_dropout_only", "attention_dropout_only"]:
                    for m in dropouts:
                        m.training = mode != "attention_dropout_only"
                    for m in attentions:
                        m.training = mode != "feedforward_dropout_only"
                    variants[mode] = [policy.model(model_batch)[0] for _ in range(16)]
                    policy.eval()
                saved_p = [m.p for m in dropouts]
                saved_attn_p = [m.dropout for m in attentions]
                policy.train()
                policy.model.training = False
                for m in dropouts:
                    m.p = 0
                for m in attentions:
                    m.dropout = 0
                variants["training_flags_but_zero_dropout"] = [policy.model(model_batch)[0]]
                for m, p in zip(dropouts, saved_p):
                    m.p = p
                for m, p in zip(attentions, saved_attn_p):
                    m.dropout = p
                policy.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    variants["eval_bfloat16"] = [policy.model(model_batch)[0].float()]
            record = {"frame": i, "posterior_mu_rms": float(mu.square().mean().sqrt()),
                      "posterior_sigma_mean": float(torch.exp(logvar/2).mean()), "modes": {}}
            mask = ~batch["action_is_pad"][0]
            for name, values in variants.items():
                predictions = np.stack([post(value).cpu().numpy()[0] for value in values])
                mean = predictions.mean(0)
                record["modes"][name] = {
                    "first_target_mean":mean[0].tolist(),
                    "first_arm_mae_of_mean":float(np.abs(mean[0,:5]-target[0,:5]).mean()),
                    "first_arm_mae_mean_of_samples":float(np.abs(predictions[:,0,:5]-target[None,0,:5]).mean()),
                    "normalized_mae_mean":float(torch.stack([(v-batch["action"]).abs()[0,mask].mean() for v in values]).mean()),
                }
            results.append(record)
            print("PROBE",i,record["modes"]["prior_zero"]["first_arm_mae_of_mean"],flush=True)
    rows = pd.concat([pd.read_parquet(p) for p in (dataset_path/"data").rglob("*.parquet")],ignore_index=True)
    states = np.stack(rows["observation.state"]);actions=np.stack(rows.action)
    coverage = {"frames":len(rows),"near_reset_frames":{},
                "expert_max_target_minus_state_degrees":np.abs(actions[:,:5]-states[:,:5]).max(0).tolist(),
                "expert_initial_target_minus_state_degrees":(actions[0,:5]-states[0,:5]).tolist(),
                "state_initial_zscore":((states[0]-states.mean(0))/states.std(0)).tolist()}
    for tolerance in [1,3,5,10]:
        coverage["near_reset_frames"][str(tolerance)] = int((np.max(np.abs(states[:,:5]-states[0,:5]),axis=1)<tolerance).sum())
    report={"checkpoint":str(checkpoint),"probes":results,"coverage":coverage,
            "limitations":"17 selected expert observations, 8 samples per stochastic mode. Posterior uses ground-truth future actions and is diagnostic only. Dropout/latent interventions are not deployable-policy improvements. No training."}
    (args.output/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=="__main__":
    main()
