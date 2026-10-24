"""Compare framewise ACT predictions, queued predictions and an existing closed-loop trace."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "myscript"))

import mujoco
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from official_env import make_official_env


def first_run(mask, length=3):
    for i in range(len(mask) - length + 1):
        if np.all(mask[i:i + length]):
            return i
    return None


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.read_csv(output / "frame_comparison.csv")
    t = table.seconds.to_numpy()
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    mask = t < 1
    for label, column in [("expert", "expert_action_1"), ("fresh expert observation", "fresh_action_1"), ("closed loop", "closed_action_1")]:
        axs[0].plot(t[mask], table.loc[mask, column], label=label)
    axs[0].set_ylabel("Shoulder target (deg)")
    axs[0].set_title("First second: error exists before any accumulated drift")
    axs[0].legend()
    for label in ["fresh", "queued", "closed"]:
        axs[1].plot(t, table[label + "_position_error_mm"], label=label, alpha=.8)
    axs[1].set_ylabel("Target position error (mm)")
    axs[1].legend()
    for label, column in [("expert", "expert_action_5"), ("expert input + queue", "queued_action_5"), ("closed loop", "closed_action_5")]:
        axs[2].plot(t, table[column], label=label, alpha=.8)
    axs[2].set_ylabel("Gripper target (0-100)")
    axs[2].legend()
    axs[3].plot(t, table[[f"closed_state_error_{i}" for i in range(5)]].max(axis=1))
    axs[3].axhline(5, color="r", linestyle="--")
    axs[3].set_ylabel("Max arm state error (deg)")
    for ax in axs:
        ax.grid(alpha=.2)
        ax.set_xlabel("Seconds from reset")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/act20261023/fixed_scene_train/train/checkpoints/005000/pretrained_model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/fixed_scene_seed200")
    parser.add_argument("--closed-trace", type=Path, default=ROOT / "outputs/act20261023/fixed_scene_train/eval/seed_200_episode_0_trace.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the plot from saved CSV without rerunning inference")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.dataset / "collection_manifest.json").read_text())
    assert len(manifest["episodes"]) == 1
    episode = manifest["episodes"][0]
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = "cuda"
    cfg.pretrained_backbone_weights = None
    policy = get_policy_class("act").from_pretrained(args.checkpoint, config=cfg).cuda().eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint,
                                        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    ds = LeRobotDataset("local/fixed_scene", root=args.dataset, video_backend="pyav",
                        delta_timestamps={"action": [i / manifest["fps"] for i in range(cfg.chunk_size)]})
    loader = DataLoader(ds, batch_size=8, num_workers=2, shuffle=False)
    chunks, states, targets, norm_errors = [], [], [], []
    for i, batch in enumerate(loader):
        states.append(batch["observation.state"].numpy().copy())
        targets.append(batch["action"][:, 0].numpy().copy())
        batch = _preprocess_dataset_batch(batch, ds.meta.camera_keys, {}, pre)
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch)
            mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(pred)
            norm_errors.append(((pred - batch["action"]).abs()[mask].sum().item(), mask.sum().item()))
            chunks.append(post(pred).cpu().numpy().copy())
        if i % 25 == 0:
            print(f"EXPERT_INPUT {min((i+1)*8,len(ds))}/{len(ds)}", flush=True)
    chunks, states, targets = map(np.concatenate, (chunks, states, targets))
    n = len(states)
    direct = chunks[:, 0]
    horizon = cfg.n_action_steps
    queued = np.stack([chunks[(i // horizon) * horizon, i % horizon] for i in range(n)])
    closed = json.loads(args.closed_trace.read_text())[:n]
    actual_states = np.array([row["state"] for row in closed])
    actual_actions = np.array([row["action"] for row in closed])
    assert len(closed) == n
    phases = np.full(n, "", dtype=object)
    for i, phase in enumerate(episode["phases"]):
        end = episode["phases"][i+1]["frame"] if i+1 < len(episode["phases"]) else n
        phases[phase["frame"]:end] = phase["phase"]

    envs = make_official_env(config_path=args.dataset / "collector_sources/configs/official_tabletop.json",
                             initial_joints=manifest["initial_joints"])
    env = envs["so101_tabletop"][0].envs[0]
    env.reset(seed=episode["seed"])
    world = env._world
    scratch = mujoco.MjData(world.model)
    scratch.qpos[:] = world.data.qpos
    site = world.model.site("gripperframe").id

    def positions(actions):
        result = []
        for action in actions:
            scratch.qpos[world.qadr] = world.to_ctrl(action)
            mujoco.mj_kinematics(world.model, scratch)
            result.append(scratch.site_xpos[site].copy())
        return np.stack(result)

    truth_xyz = positions(targets)
    errors = {}
    columns = {"frame": np.arange(n), "seconds": np.arange(n)/manifest["fps"], "phase": phases}
    for label, prediction in [("fresh", direct), ("queued", queued), ("closed", actual_actions)]:
        diff = np.abs(prediction - targets)
        pos = np.linalg.norm(positions(prediction) - truth_xyz, axis=1)*1000
        errors[label] = {
            "arm_mae_degrees": float(diff[:, :5].mean()), "gripper_mae": float(diff[:, 5].mean()),
            "setpoint_position_error_mm_mean": float(pos.mean()), "setpoint_position_error_mm_p90": float(np.percentile(pos,90)),
            "first_3frame_run_arm_max_error_over_5deg": first_run(diff[:, :5].max(1)>5),
            "first_3frame_run_setpoint_error_over_5mm": first_run(pos>5),
            "initial_action_error": diff[0].tolist(), "initial_setpoint_error_mm": float(pos[0]),
        }
        columns[label+"_arm_mae_degrees"] = diff[:, :5].mean(1)
        columns[label+"_position_error_mm"] = pos
        for joint in range(6):columns[f"{label}_action_{joint}"] = prediction[:,joint]
    for joint in range(6):
        columns[f"expert_action_{joint}"] = targets[:,joint]
        columns[f"closed_state_error_{joint}"] = np.abs(actual_states[:,joint]-states[:,joint])
    first_state_error = first_run(np.abs(actual_states[:,:5]-states[:,:5]).max(1)>5)
    table = pd.DataFrame(columns)
    table.to_csv(args.output / "frame_comparison.csv",index=False)
    by_phase = []
    for phase in dict.fromkeys(phases):
        mask = phases==phase
        by_phase.append({"phase":phase,"frames":int(mask.sum()),**{label+"_position_error_mm":float(table.loc[mask,label+"_position_error_mm"].mean()) for label in errors}})

    # Real expert replay: compare a live observation to compressed recorded images
    # at every replan point in the first 120 frames, including the first grasp.
    live_checks = []
    from lerobot.envs.utils import preprocess_observation
    obs = env.reset(seed=episode["seed"])[0]
    max_state_error = 0.0
    for i in range(n):
        max_state_error = max(max_state_error,float(np.abs(obs["agent_pos"]-states[i]).max()))
        if i < 120 and i % horizon == 0:
            live = preprocess_observation({"pixels":{"front":obs["pixels"]["cam0"],"wrist":obs["pixels"]["cam1"]},"agent_pos":obs["agent_pos"]})
            live["task"] = [env.task_description]
            with torch.inference_mode():
                pred = post(policy.predict_action_chunk(pre(live))).cpu().numpy()[0,0]
            live_checks.append({"frame":i,"live_target_arm_mae_degrees":float(np.abs(pred[:5]-targets[i,:5]).mean()),
                                "live_vs_saved_arm_mae_degrees":float(np.abs(pred[:5]-direct[i,:5]).mean())})
        obs, _, terminated, _, info = env.step(targets[i].astype(np.float32))
    replay = {"state_max_error":max_state_error,"cubes_in_box":int(info['cubes_in_box']),"final_success":bool(terminated)}
    envs["so101_tabletop"][0].close()
    result = {"checkpoint":str(args.checkpoint.resolve()),"dataset":str(args.dataset.resolve()),"frames":n,"n_action_steps":horizon,
              "normalized_prior_inference_mae_full_chunk":sum(x[0] for x in norm_errors)/sum(x[1] for x in norm_errors),
              "errors":errors,"first_3frame_closed_state_arm_error_over_5deg":first_state_error,
              "by_phase":by_phase,"expert_replay":replay,"live_checks":live_checks,
              "threshold_note":"5mm/5deg for 3 consecutive frames are diagnostic thresholds, not calibrated grasp-success thresholds. Frame indices start at 0.",
              "method":"fresh: expert observation each frame, first predicted target. queued: predictions at frames 0,10,..., execute first 10 targets offline. closed: existing rollout vs time-aligned expert; comparisons after divergence are descriptive, not equal-state tests."}
    (args.output / "results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    plot_results(args.output)
    print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
