"""Run a LeRobot checkpoint in the pinned official simulator and save videos/results.

The checkpoint chooses the model class. Inference and normalization use official
LeRobot APIs; only environment/camera wiring and result recording are local.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "myscript"))

import av
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from official_env import REPO, REVISION, make_official_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint's pretrained_model directory")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Read collection manifest to reproduce camera settings and initial joints")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000, help="New seeds, separate from collection")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seeds (overrides episodes/seed)")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-count", type=int, default=None, help="Record only the first N evaluation episodes")
    parser.add_argument("--video-dir", type=Path, help="Separate directory for evaluation videos")
    parser.add_argument("--stable-finish", action="store_true", help="Continue policy after raw termination; require open-gripper stability for two seconds")
    parser.add_argument("--n-action-steps", type=int, help="Override execution horizon without retraining")
    parser.add_argument("--temporal-ensemble-coeff", type=float, help="ACT's official temporal ensemble; executes one step")
    parser.add_argument("--trace", action="store_true", help="Save state/action traces for diagnosis")
    parser.add_argument("--torch-threads", type=int, default=2, help="Limit CPU overhead during GPU inference")
    parser.add_argument("--allow-nondeterministic", action="store_true", help="Allow faster, less repeatable PyTorch kernels")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    if args.video_count is not None and args.video_count < 0:
        parser.error("video-count must be nonnegative")
    if args.torch_threads < 1:
        parser.error("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    if not args.allow_nondeterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = args.checkpoint.resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    if args.n_action_steps is not None:
        if hasattr(config, "chunk_size"):
            maximum = config.chunk_size
        elif hasattr(config, "horizon"):
            maximum = config.horizon - config.n_obs_steps + 1
        else:
            parser.error("This policy does not expose a supported action horizon")
        if not 1 <= args.n_action_steps <= maximum:
            parser.error(f"Require 1 <= n-action-steps <= {maximum} for this checkpoint")
        config.n_action_steps = args.n_action_steps
    if args.temporal_ensemble_coeff is not None:
        if config.type != "act" or args.temporal_ensemble_coeff < 0:
            parser.error("Temporal ensembling requires ACT and a nonnegative coefficient")
        config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        config.n_action_steps = 1
    # All parameters are restored from the checkpoint; avoid downloading ImageNet weights again.
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    manifest = json.loads((args.dataset / "collection_manifest.json").read_text())
    if manifest["env_revision"] != REVISION:
        raise ValueError("Dataset environment revision differs from the pinned playback environment")
    env_config = args.dataset / "collector_sources/configs/official_tabletop.json"
    envs = make_official_env(config_path=env_config, initial_joints=manifest["initial_joints"])
    env = envs["so101_tabletop"][0].envs[0]
    target_count = len(env._world.cube_bodies) if env._world is not None else manifest.get("target_cube_count", 4)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    batch = ROOT / "outputs" / f"{config.type}{now:%Y%m%d}"
    output = args.output or batch / f"eval_{now:%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    results = {
        "checkpoint": str(checkpoint), "policy_type": config.type,
        "env_repo": REPO, "env_revision": REVISION,
        "dataset": str(args.dataset.resolve()), "fps": env.fps,
        "seed": args.seed, "requested_episodes": args.episodes, "max_steps": args.max_steps,
        "execution": {"n_action_steps": config.n_action_steps,
                      "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
                      "torch_threads": args.torch_threads,
                      "deterministic_torch": not args.allow_nondeterministic,
                      "determinism_scope": "PyTorch only; OpenGL-rendered pixels can still vary"},
        "target_cube_count": target_count,
        "success_definition": ("All target cubes in box AND target/actual gripper>=95 for two consecutive seconds; policy actions only"
                               if args.stable_finish else "Environment terminated flag (all target cubes in box)"),
        "grasp_definition": "Proxy: unique cube lifted >=3cm above reset height, within8cm of gripperframe, actual gripper<50, for>=0.1s; denominator=target cubes, not grasp attempts",
        "episodes": [],
    }
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    results["requested_episodes"] = len(seeds)
    results["seeds"] = seeds
    try:
        for episode, seed in enumerate(seeds):
            set_seed(seed)
            obs, info = env.reset(seed=seed)
            world = env._world
            target_count = len(world.cube_bodies)
            results["target_cube_count"] = target_count
            initial_heights = world.data.xpos[world.cube_bodies, 2].copy()
            grasp_runs = np.zeros(target_count, dtype=int)
            grasped = np.zeros(target_count, dtype=bool)
            gripper_site = world.model.site("gripperframe").id
            stable_run = longest_stable = 0
            strict_success = official_seen = False
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            stem = f"seed_{seed}" if seeds.count(seed) == 1 else f"seed_{seed}_episode_{episode}"
            video = (args.video_dir or output) / f"{stem}.mp4"
            container = None
            record_video = not args.no_video and (args.video_count is None or episode < args.video_count)
            if record_video:
                video.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(video), "w")
                stream = container.add_stream("libx264", rate=env.fps)
                h, w = obs["pixels"]["cam0"].shape[:2]
                stream.width, stream.height, stream.pix_fmt = w * 2, h, "yuv420p"
            started = time.monotonic()
            terminated = truncated = False
            peak_cubes = 0
            reward_sum = 0.0
            trace = []
            try:
                for step in range(1, args.max_steps + 1):
                    state_before = obs["agent_pos"].copy()
                    batch = preprocess_observation({
                        "pixels": {"front": obs["pixels"]["cam0"], "wrist": obs["pixels"]["cam1"]},
                        "agent_pos": obs["agent_pos"],
                    })
                    batch["task"] = [env.task_description]
                    with torch.inference_mode():
                        action = postprocessor(policy.select_action(preprocessor(batch)))
                    action = action.detach().cpu().numpy()[0].astype(np.float32)
                    if action.shape != (6,) or not np.isfinite(action).all():
                        raise ValueError(f"Invalid policy action: {action}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += float(reward)
                    peak_cubes = max(peak_cubes, int(info.get("cubes_in_box", 0)))
                    official_seen |= bool(terminated)
                    good = (info.get("cubes_in_box", 0) == target_count and action[5] >= 95
                            and obs["agent_pos"][5] >= 95)
                    stable_run = stable_run + 1 if good else 0
                    longest_stable = max(longest_stable, stable_run)
                    strict_success = stable_run >= int(np.ceil(2 * env.fps))
                    positions = world.data.xpos[world.cube_bodies]
                    lifted = ((positions[:, 2] - initial_heights >= .03)
                              & (np.linalg.norm(positions - world.data.site_xpos[gripper_site], axis=1) < .08)
                              & (obs["agent_pos"][5] < 50))
                    grasp_runs = np.where(lifted, grasp_runs + 1, 0)
                    grasped |= grasp_runs >= int(np.ceil(.1 * env.fps))
                    if args.trace:
                        trace.append(dict(step=step, state=state_before.tolist(), action=action.tolist(),
                                          next_state=obs["agent_pos"].tolist(),
                                          cubes_in_box=int(info.get("cubes_in_box", 0))))
                    if container:
                        frame = np.concatenate([obs["pixels"]["cam0"], obs["pixels"]["cam1"]], axis=1)
                        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                            container.mux(packet)
                    if truncated or (strict_success if args.stable_finish else terminated):
                        break
            finally:
                if container:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
            success = strict_success if args.stable_finish else bool(terminated)
            record = dict(seed=seed, steps=step, success=success, strict_success=bool(strict_success),
                          official_success_seen=official_seen, longest_stable_frames=longest_stable,
                          grasped_cubes=int(grasped.sum()), target_cube_count=target_count,
                          truncated=bool(truncated), step_limit=not (success or truncated),
                          cubes_in_box=int(info.get("cubes_in_box", 0)), peak_cubes=peak_cubes,
                          reward_sum=reward_sum, wall_seconds=time.monotonic() - started,
                          video=str(video.resolve()) if record_video else None)
            results["episodes"].append(record)
            if args.trace:
                (output / f"{stem}_trace.json").write_text(json.dumps(trace))
            results["success_rate"] = sum(x["success"] for x in results["episodes"]) / len(results["episodes"])
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(record), flush=True)
        print(f"Success: {results['success_rate']:.1%}; results: {output.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
