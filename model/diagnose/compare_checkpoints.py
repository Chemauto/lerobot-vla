"""Compare saved checkpoints in one fixed scene via the existing official-API playback entry."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "outputs/act20261023/train/checkpoints")
    parser.add_argument("--steps", type=int, nargs="+", default=[3000, 6000, 9000, 12000, 15000])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/fixed_scene_seed200")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--repeats", type=int, default=3, help="Additional repeats of the selected checkpoint")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_steps < 1:
        parser.error("repeats and max-steps must be positive")
    checkpoints = {step: args.checkpoints.resolve() / f"{step:06d}" / "pretrained_model" for step in args.steps}
    for path in checkpoints.values():
        if not (path / "model.safetensors").is_file():
            parser.error(f"Missing checkpoint: {path}")
    manifest = json.loads((args.dataset / "collection_manifest.json").read_text())
    if not any(ep["seed"] == args.seed for ep in manifest["episodes"]):
        parser.error("Selected seed is not represented in the diagnostic dataset")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": str(args.dataset.resolve()), "seed": args.seed, "max_steps": args.max_steps,
        "n_action_steps": 10, "results": [], "status": "running",
        "selection_rule": "full success, final cubes, peak cubes; tie: latest checkpoint",
        "limitations": "One training scene only; grasp/lift success is not inferred from gripper closure alone. OpenGL pixels can vary.",
    }

    def save():
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        lines = ["# 🎯 ACT 固定场景检查点对比", "", f"同一训练场景 seed {args.seed}；每回合最多{args.max_steps}步，每次执行10步。未训练、未采集新数据。", "",
                 "| 模型步数 | 四块全部成功 | 最终入盒 | 最多入盒 | 首次入盒步数 | 视频与记录 |", "|---|---|---:|---:|---|---|"]
        for record in summary["results"]:
            ep = record["episode"]
            lines.append(f"| {record['checkpoint_step']} | {'✅' if ep['success'] else '❌'} | {ep['cubes_in_box']} | {ep['peak_cubes']} | {record['first_cube_step'] or '无'} | [{record['folder']}]({record['folder']}/README.md) |")
        if "selected_step" in summary:
            lines.extend(["", f"按已声明规则选择 {summary['selected_step']} 步模型做额外 {args.repeats} 次重复；平手选择较晚检查点，不代表它更好。"])
        if "repeat_results" in summary:
            episodes = summary["repeat_results"]["episodes"]
            lines.extend(["", f"额外重复：完整成功 {sum(e['success'] for e in episodes)}/{len(episodes)}，最终入盒数 {[e['cubes_in_box'] for e in episodes]}。", "", "[重复测试视频与结果](repeat_selected/README.md)"])
        lines.extend(["", f"状态：{summary['status']}。", "", "单个训练场景不能证明泛化；重复场景也不能保证逐位相同。是否夹住/抬起需结合视频观察，不能只凭夹爪闭合判定。"])
        (output / "README.md").write_text("\n".join(lines) + "\n")

    def run(step, folder, seeds):
        destination = output / folder
        command = [sys.executable, str(ROOT / "model/play/policy.py"),
                   "--checkpoint", str(checkpoints[step]), "--dataset", str(args.dataset.resolve()),
                   "--seeds", *map(str, seeds), "--max-steps", str(args.max_steps),
                   "--n-action-steps", "10", "--trace", "--output", str(destination)]
        print(f"START checkpoint={step} seeds={seeds}", flush=True)
        with (output / f"{folder}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Keep the subprocess log in its own run folder after successful creation.
        (output / f"{folder}.log").rename(destination / "run.log")
        result = json.loads((destination / "results.json").read_text())
        links = "\n".join(f"- [回合{i+1}视频]({Path(ep['video']).name})：入盒{ep['cubes_in_box']}，完整成功{ep['success']}" for i, ep in enumerate(result["episodes"]))
        (destination / "README.md").write_text(f"# ACT {step}步 · seed {args.seed}\n\n{links}\n\n详细结果：results.json；动作：*_trace.json；运行日志：run.log。\n")
        print(f"DONE checkpoint={step} cubes={[ep['cubes_in_box'] for ep in result['episodes']]}", flush=True)
        return result

    save()
    try:
        for step in args.steps:
            folder = f"checkpoint_{step:06d}"
            result = run(step, folder, [args.seed])
            trace = json.loads((output / folder / f"seed_{args.seed}_trace.json").read_text())
            first = next((row["step"] for row in trace if row["cubes_in_box"] > 0), None)
            summary["results"].append({"checkpoint_step": step, "folder": folder, "episode": result["episodes"][0], "first_cube_step": first})
            save()
        selected = max(summary["results"], key=lambda r: (r["episode"]["success"], r["episode"]["cubes_in_box"], r["episode"]["peak_cubes"], r["checkpoint_step"]))
        summary["selected_step"] = selected["checkpoint_step"]
        save()
        summary["repeat_results"] = run(summary["selected_step"], "repeat_selected", [args.seed] * args.repeats)
        summary["status"] = "completed"
        save()
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
