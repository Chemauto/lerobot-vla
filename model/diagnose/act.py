"""Audit ACT wiring, replay recorded actions and measure errors by expert phase.

Diagnostics only: no training or upstream edits. Output stays under outputs/.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "myscript"))

import numpy as np
import pandas as pd
import torch
import mujoco
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from official_env import REVISION, make_official_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/official_tabletop_50")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/act20261023/train/checkpoints/015000/pretrained_model")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 40])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / "collection_manifest.json").read_text())
    assert manifest["env_revision"] == REVISION
    rows = pd.concat([pd.read_parquet(p) for p in sorted((args.dataset / "data").rglob("*.parquet"))], ignore_index=True)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cuda"
    config.pretrained_backbone_weights = None
    policy = get_policy_class("act").from_pretrained(args.checkpoint, config=config).eval().cuda()
    pre, post = make_pre_post_processors(config, pretrained_path=args.checkpoint,
                                        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    dataset = LeRobotDataset("local/so101_official_tabletop", root=args.dataset, video_backend="pyav",
                            delta_timestamps={"action": [i / manifest["fps"] for i in range(config.chunk_size)]})
    envs = make_official_env(config_path=args.dataset / "collector_sources/configs/official_tabletop.json",
                             initial_joints=manifest["initial_joints"])
    env = envs["so101_tabletop"][0].envs[0]
    summary = {"checkpoint": str(args.checkpoint.resolve()), "replays": [], "phase_frame_counts": {},
               "samples": [], "camera_mapping": {"front": "cam0", "wrist": "cam1"}}
    phase_counts = defaultdict(int)
    for ep in manifest["episodes"]:
        phases = ep["phases"]
        for i, phase in enumerate(phases):
            end = phases[i + 1]["frame"] if i + 1 < len(phases) else ep["frames"]
            phase_counts[phase["phase"]] += end - phase["frame"]
    summary["phase_frame_counts"] = dict(phase_counts)
    try:
        for ep_index in args.episodes:
            ep = manifest["episodes"][ep_index]
            part = rows[rows.episode_index == ep_index].reset_index(drop=True)
            start = int(dataset.meta.episodes[ep_index]["dataset_from_index"])
            obs, _ = env.reset(seed=ep["seed"])
            initial = dataset[start]
            reset_error = float(np.max(np.abs(obs["agent_pos"] - initial["observation.state"].numpy())))
            live = preprocess_observation({"pixels": {"front": obs["pixels"]["cam0"], "wrist": obs["pixels"]["cam1"]},
                                           "agent_pos": obs["agent_pos"]})
            saved = {key: initial[key].unsqueeze(0).clone() for key in config.input_features}
            live["task"] = saved["task"] = [env.task_description]
            live_pre = pre(live)
            saved_pre = pre(saved)
            input_diffs = {key: float((live_pre[key] - saved_pre[key]).abs().mean()) for key in config.input_features}
            with torch.inference_mode():
                live_action = post(policy.predict_action_chunk(live_pre))
                saved_action = post(policy.predict_action_chunk(saved_pre))
            first_action_difference = (live_action - saved_action).abs().mean(dim=(0, 1)).tolist()
            errors = []
            for step, row in part.iterrows():
                errors.append(float(np.max(np.abs(obs["agent_pos"] - np.asarray(row["observation.state"])))))
                obs, _, terminated, truncated, info = env.step(np.asarray(row["action"], dtype=np.float32))
            final_error = float(np.max(np.abs(env._world.data.qpos - np.asarray(ep["final_qpos"]))))
            replay = dict(episode=ep_index, seed=ep["seed"], reset_state_max_error=reset_error,
                          state_max_error=max(errors), final_qpos_max_error=final_error,
                          cubes_in_box=int(info["cubes_in_box"]), final_success=bool(terminated),
                          preprocessed_input_mean_abs_diff=input_diffs,
                          live_vs_recorded_prediction_mean_abs_diff=first_action_difference)
            summary["replays"].append(replay)
            print("REPLAY", json.dumps(replay), flush=True)
            world = env._world
            scratch = mujoco.MjData(world.model)
            site = world.model.site("gripperframe").id

            def target_pose(action):
                # Compare kinematic setpoints on scratch data; never change live physics.
                scratch.qpos[:] = world.data.qpos
                scratch.qpos[world.qadr] = world.to_ctrl(np.asarray(action))
                mujoco.mj_kinematics(world.model, scratch)
                return scratch.site_xpos[site].copy(), scratch.site_xmat[site].reshape(3, 3).copy()
            # Cover the first two pick/place sequences, plus the exact initial state.
            indices = {0: "initial"}
            phases = ep["phases"]
            for i, phase in enumerate(phases[:16]):
                end = phases[i + 1]["frame"] if i + 1 < len(phases) else ep["frames"]
                indices[phase["frame"]] = phase["phase"]
                indices[(phase["frame"] + end - 1) // 2] = phase["phase"]
            for index, phase in sorted(indices.items()):
                item = dataset[start + index]
                state = item["observation.state"].clone()
                target = item["action"].clone()
                batch = pre(item)
                target_normalized = batch["action"].clone()
                valid = ~batch["action_is_pad"][0]
                for condition in ["normal", "black_images", "zero_state"]:
                    inputs = dict(batch)
                    if condition == "black_images":
                        # Zero in normalized image space: mean-color images, not raw black RGB.
                        for key in config.image_features:
                            inputs[key] = torch.zeros_like(inputs[key])
                    elif condition == "zero_state":
                        inputs["observation.state"] = torch.zeros_like(inputs["observation.state"])
                    with torch.inference_mode():
                        prediction_normalized = policy.predict_action_chunk(inputs)
                        prediction = post(prediction_normalized).cpu()[0]
                    diff = (prediction - target).abs()
                    normalized_diff = (prediction_normalized - target_normalized).abs()[0]
                    record = dict(episode=ep_index, offset=index, phase=phase, condition=condition,
                                  normalized_mae=float(normalized_diff[valid].mean()),
                                  arm_mae_degrees=float(diff[:10, :5].mean()),
                                  gripper_mae=float(diff[:10, 5].mean()),
                                  state_copy_arm_mae_degrees=float((target[:10, :5] - state[:5]).abs().mean()),
                                  predicted_arm_change_degrees=float((prediction[:10, :5] - state[:5]).abs().mean()))
                    if condition == "normal":
                        pos_a, rot_a = target_pose(prediction[0].numpy())
                        pos_b, rot_b = target_pose(target[0].numpy())
                        record["setpoint_position_error_mm"] = float(np.linalg.norm(pos_a - pos_b) * 1000)
                        record["setpoint_orientation_error_deg"] = float(np.rad2deg(np.arccos(
                            np.clip((np.trace(rot_a.T @ rot_b) - 1) / 2, -1, 1))))
                    summary["samples"].append(record)
            (args.output / "audit.json").write_text(json.dumps(summary, indent=2))
    finally:
        envs["so101_tabletop"][0].close()
    grouped = {}
    for condition in ["normal", "black_images", "zero_state"]:
        values = [x for x in summary["samples"] if x["condition"] == condition]
        grouped[condition] = {key: float(np.mean([x[key] for x in values])) for key in
                              ["normalized_mae", "arm_mae_degrees", "gripper_mae"]}
    summary["sample_averages"] = grouped
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(grouped), flush=True)


if __name__ == "__main__":
    main()
