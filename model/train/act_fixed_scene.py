"""Fine-tune ACT through the official trainer, then evaluate three fixed-scene episodes."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/act20261023/train/checkpoints/015000/pretrained_model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dropout", type=float, help="Override only ACT dropout through the official trainer CLI")
    parser.add_argument("--audit", action="store_true", help="After evaluation, also run the expert-observation framewise audit")
    parser.add_argument("--compare-with", type=Path, help="Previous fixed-scene run to include in the final comparison")
    args = parser.parse_args()
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error("dropout must be between 0 (inclusive) and 1 (exclusive)")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        parser.error(f"Checkpoint missing: {checkpoint}")
    day = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    label = "fixed_scene_train" if args.dropout is None else f"fixed_scene_dropout{args.dropout:g}"
    output = args.output or ROOT / "outputs" / f"act{day}" / label
    if args.output is None:
        number = 2
        while output.exists():
            output = ROOT / "outputs" / f"act{day}" / f"{label}_{number:02d}"
            number += 1
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = ROOT / "model/configs/act_fixed_scene.json"
    training = [sys.executable, str(ROOT / "model/train/act.py"), "--config", str(config),
                f"--policy.path={checkpoint}", f"--output_dir={output / 'train'}"]
    if args.dropout is not None:
        training.append(f"--policy.dropout={args.dropout}")
    evaluation = [sys.executable, str(ROOT / "model/play/policy.py"),
                  "--checkpoint", str(output / "train/checkpoints/005000/pretrained_model"),
                  "--dataset", str(ROOT / "data/fixed_scene_seed200"), "--seeds", "200", "200", "200",
                  "--max-steps", "3600", "--n-action-steps", "10", "--trace", "--output", str(output / "eval")]
    state = {
        "status": "starting", "pid": os.getpid(),
        "started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "initial_checkpoint": str(checkpoint), "training_command": training, "evaluation_command": evaluation,
        "steps": 5000, "scene_seed": 200, "repeats": 3,
        "dropout_override": args.dropout, "framewise_audit": args.audit,
        "compare_with": str(args.compare_with.resolve()) if args.compare_with else None,
        "note": "Warm-start weights, fresh optimizer. Official trainer uses the subset's normalization statistics. No held-out validation; no claim of generalization.",
    }

    def save(status):
        state["status"] = status
        state["updated"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output / "run.json")
        text = ("# 🎯 ACT 固定场景专项训练\n\n"
                f"状态：**{status}**\n\n"
                "仅训练 seed200 的1条示范（1,300帧），从原15,000步权重继续学习5,000步，使用新的优化器。"
                "归一化由官方训练入口按单条数据统计处理。没有独立验证集，不能作为泛化成绩。\n\n"
                "- `train/`：官方训练输出，2,500步与5,000步检查点。\n"
                "- `train.log`：训练过程与损失。\n"
                "- `eval/`：训练完成后自动重复测试seed200三次，保存视频、结果和动作记录。\n"
                "- `eval.log`：评估运行日志。\n"
                "- `run.json`：启动参数、状态和错误信息。\n")
        if args.dropout is not None:
            text += f"\n本次唯一训练参数改动：Dropout={args.dropout:g}。评估使用正常eval推理，不启用MC Dropout。\n"
        if args.audit:
            text += "\n`audit/`：评估后自动检查1,300帧专家观测的预测误差，输出CSV、JSON和曲线。\n"
        if "evaluation_results" in state:
            eps = state["evaluation_results"]["episodes"]
            text += f"\n完整成功：{sum(ep['success'] for ep in eps)}/{len(eps)}；最终入盒数：{[ep['cubes_in_box'] for ep in eps]}。\n"
            for i, ep in enumerate(eps):
                text += f"\n[第{i+1}次视频](eval/{Path(ep['video']).name})\n"
        if "audit_results" in state:
            errors = state["audit_results"]["errors"]["fresh"]
            text += (f"\n专家观测逐帧检查：手臂目标MAE {errors['arm_mae_degrees']:.3f}°，"
                     f"运动学目标位置平均误差 {errors['setpoint_position_error_mm_mean']:.3f}mm。\n"
                     "\n[误差曲线](audit/comparison.png) · [对照汇总](comparison.json)\n")
        (output / "README.md").write_text(text)
        print(f"STATUS {status}: {output}", flush=True)

    try:
        save("training")
        with (output / "train.log").open("w") as log:
            subprocess.run(training, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        save("evaluating")
        with (output / "eval.log").open("w") as log:
            subprocess.run(evaluation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["evaluation_results"] = json.loads((output / "eval/results.json").read_text())
        if args.audit:
            save("auditing")
            command = [sys.executable, str(ROOT / "model/diagnose/teacher_forcing.py"),
                       "--checkpoint", str(output / "train/checkpoints/005000/pretrained_model"),
                       "--closed-trace", str(output / "eval/seed_200_episode_0_trace.json"),
                       "--output", str(output / "audit")]
            with (output / "audit.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["audit_results"] = json.loads((output / "audit/results.json").read_text())
        comparison = {"dropout_override": args.dropout,
                      "current_final_cubes": [e["cubes_in_box"] for e in state["evaluation_results"]["episodes"]],
                      "current_full_success_rate": state["evaluation_results"]["success_rate"]}
        if args.compare_with:
            baseline = json.loads((args.compare_with / "eval/results.json").read_text())
            comparison["previous_final_cubes"] = [e["cubes_in_box"] for e in baseline["episodes"]]
            comparison["previous_full_success_rate"] = baseline["success_rate"]
        if "audit_results" in state:
            comparison["current_framewise_metrics"] = state["audit_results"]["errors"]["fresh"]
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
        save("completed")
    except Exception as exc:
        state["error"] = str(exc)
        save("failed")
        raise


if __name__ == "__main__":
    main()
