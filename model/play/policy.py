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
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/official_tabletop_50",
                        help="Read collection manifest to reproduce camera settings and initial joints")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000, help="New seeds, separate from collection")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seeds (overrides episodes/seed)")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--n-action-steps", type=int, help="Override execution horizon without retraining")
    parser.add_argument("--temporal-ensemble-coeff", type=float, help="ACT's official temporal ensemble; executes one step")
    parser.add_argument("--trace", action="store_true", help="Save state/action traces for diagnosis")
    parser.add_argument("--torch-threads", type=int, default=2, help="Limit CPU overhead during GPU inference")
    parser.add_argument("--allow-nondeterministic", action="store_true", help="Allow faster, less repeatable PyTorch kernels")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
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
        if not 1 <= args.n_action_steps <= config.chunk_size:
            parser.error("Require 1 <= n-action-steps <= checkpoint chunk_size")
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
        "success_definition": "official terminated flag (all four cubes in box)",
        "episodes": [],
    }
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    results["requested_episodes"] = len(seeds)
    results["seeds"] = seeds
    try:
        for episode, seed in enumerate(seeds):
            set_seed(seed)
            obs, info = env.reset(seed=seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            stem = f"seed_{seed}" if seeds.count(seed) == 1 else f"seed_{seed}_episode_{episode}"
            video = output / f"{stem}.mp4"
            container = None
            if not args.no_video:
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
                    if args.trace:
                        trace.append(dict(step=step, state=state_before.tolist(), action=action.tolist(),
                                          next_state=obs["agent_pos"].tolist(),
                                          cubes_in_box=int(info.get("cubes_in_box", 0))))
                    if container:
                        frame = np.concatenate([obs["pixels"]["cam0"], obs["pixels"]["cam1"]], axis=1)
                        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                            container.mux(packet)
                    if terminated or truncated:
                        break
            finally:
                if container:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
            record = dict(seed=seed, steps=step, success=bool(terminated),
                          truncated=bool(truncated), step_limit=not (terminated or truncated),
                          cubes_in_box=int(info.get("cubes_in_box", 0)), peak_cubes=peak_cubes,
                          reward_sum=reward_sum, wall_seconds=time.monotonic() - started,
                          video=None if args.no_video else str(video.resolve()))
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
