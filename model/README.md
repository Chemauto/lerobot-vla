# 🧠 策略训练与仿真

训练直接用 **LeRobot 官方 trainer**；本目录只保存配置、启动入口和仿真接线。

```text
model/
├── configs/act.json  # 数据、模型与训练参数
├── train/act.py      # ACT 启动入口，透传官方参数
├── play/policy.py    # 通用 checkpoint 播放与评估
└── diagnose/act.py   # 专家动作回放、输入核对和分阶段误差
```

## 🚀 训练 ACT

在仓库根目录运行：

```bash
.venv/bin/python model/train/act.py
```

默认：50 条示范按 episode 留出 20% 验证，batch 8、BF16、15,000 步，每 3,000 步保存。预测 50 步动作、执行 10 步后重新看图。验证损失每 1,000 步计算，最多取 128 个样本，**不代表抓取成功率**。首次训练会下载 ResNet18 权重。

🔧 所有官方参数均可覆盖，例如短跑：

```bash
.venv/bin/python model/train/act.py --steps=2 --num_workers=0 \
  --eval_steps=0 --log_freq=1 --output_dir=outputs/act20261023/smoke
```

🔄 断点续训（使用 checkpoint 中保存的完整配置）：

```bash
.venv/bin/python model/train/act.py \
  --config outputs/act20261023/train/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=30000
```

## 🎬 仿真评估与视频

```bash
.venv/bin/python model/play/policy.py \
  --checkpoint outputs/act20261023/train/checkpoints/last/pretrained_model \
  --episodes 10 --seed 10000
```

📷 输出到 `outputs/act年月日/eval_时分秒/`：双相机 MP4、每回合结果和成功率。使用采集时的相机、初始关节和 33Hz；新种子测试。模型只看图像、关节和任务，不读取方块真值。成功采用官方四方块入盒判定；到达步数上限也记失败。

🔎 排查时可加 `--trace` 保存动作与关节状态；`--seeds 200 240` 指定场景。重复种子会保存独立文件。

执行方式可通过 `--n-action-steps 50` 或 `--temporal-ensemble-coeff 0.01` 对照，二者调用官方 ACT 实现，均不需要重新训练。**目前这些设置仍未解决完整四方块任务，默认配置暂不改动。**

播放默认限制 CPU 线程并启用 PyTorch 确定性计算；这不保证整条仿真逐位一致，OpenGL 图像仍可能有极少数像素差异。`--allow-nondeterministic` 可放宽 PyTorch 限制。

## 🩺 排查

```bash
.venv/bin/python model/diagnose/act.py --output outputs/act20261023/diagnose_new
```

核对训练/播放输入，原样回放第 0、40 条专家动作，抽查分阶段动作、夹爪目标位姿误差和图像/状态依赖；不会训练，也不改官方环境。详见 [本轮排查结论](DIAGNOSIS.md)。

➕ 扩展 SmolVLA 时添加对应训练配置与入口，共用播放脚本和官方归一化处理。环境、数据、权重、视频、日志均不提交 Git。

## 📂 实验目录

按模型名＋北京时间日期分组，例如 `outputs/act20261023/`、`outputs/smolvla20261025/`。先看 [实验导航](../outputs/README.md)。新ACT训练默认保存到当日组内的 `train/`，已存在则使用 `train_02/` 等；新评估保存到组内的 `eval_时分秒/`。也可用 `--output_dir`（训练）或 `--output`（评估）指定组内目录。恢复训练沿用该配置保存的位置。

固定场景数据来自现有 50 条中的第 0 条：`data/fixed_scene_seed200`，准备入口 `myscript/prepare_fixed_scene.py`，不是重新采集。评估时同时传 `--dataset data/fixed_scene_seed200 --seeds 200`；只指定 dataset 不会固定场景。仅单条训练时应设 `--dataset.eval_split=0`，不能把它称为独立验证成绩。当前尚未启动新的训练。
