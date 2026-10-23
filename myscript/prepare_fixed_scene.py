"""Extract one recorded scene with the official LeRobot dataset splitter; no simulation or training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/official_tabletop_50")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-parent", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    manifest = json.loads((args.source / "collection_manifest.json").read_text())
    if not 0 <= args.episode < len(manifest["episodes"]):
        parser.error("episode is outside the source dataset")
    selected = deepcopy(manifest["episodes"][args.episode])
    name = f"fixed_scene_seed{selected['seed']}"
    target = args.output_parent / name
    if target.exists():
        parser.error(f"Output already exists: {target}")
    source = LeRobotDataset("local/so101_official_tabletop", root=args.source, video_backend="pyav")
    split_dataset(source, {name: [args.episode]}, output_dir=args.output_parent)
    selected["source_episode_index"] = args.episode
    selected["episode_index"] = 0
    manifest["episodes"] = [selected]
    manifest["attempts"] = [a for a in manifest.get("attempts", []) if a.get("seed") == selected["seed"]]
    manifest["subset_provenance"] = {
        "source": str(args.source.resolve()), "episode_index": args.episode,
        "seed": selected["seed"], "method": "official lerobot.datasets.dataset_tools.split_dataset",
        "note": "Extracted existing demonstration; no recollection. Reset the simulator with this seed for playback.",
    }
    (target / "collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.copytree(args.source / "collector_sources", target / "collector_sources")
    (target / "README.md").write_text(
        f"# 🎯 固定场景：seed {selected['seed']}\n\n"
        f"来自 `{args.source.name}` 的第 {args.episode} 条（从 0 编号），仅 1 条成功示范、{selected['frames']} 帧。\n\n"
        "通过 LeRobot 官方拆分接口提取，没有重新采集。用于固定场景专项训练；不能用来证明泛化能力。\n\n"
        f"评估须同时指定 `--dataset data/{name} --seeds {selected['seed']}`；只换数据目录不会固定种子。"
        "固定种子保留官方重置流程，同样初始摆放可复现，不保证渲染与整条轨迹逐位一致。\n"
    )
    print(target)


if __name__ == "__main__":
    main()
