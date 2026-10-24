# 🧠 策略训练与仿真

训练直接用 **LeRobot 官方 trainer**；本目录保存本地配置、启动入口和仿真接线。**官方源码不等于所有参数都是官方默认值**，详见 [配置来源](CONFIG_PROVENANCE.md)。

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

🧪 ACT诊断选项 `--mc-dropout-samples 16`：仅启用普通Dropout（前馈/残差分支），16个随机预测取平均，VAE与attention内部dropout仍为eval。默认值0保持正常推理。这是定位训练/推理偏移的实验干预，不能据部分成功认定部署已修复。

## 🩺 排查

```bash
.venv/bin/python model/diagnose/act.py --output outputs/act20261023/diagnose_new
```

核对训练/播放输入，原样回放第 0、40 条专家动作，抽查分阶段动作、夹爪目标位姿误差和图像/状态依赖；不会训练，也不改官方环境。详见 [本轮排查结论](DIAGNOSIS.md)。

➕ ACT、Diffusion和SmolVLA共用官方训练启动器、播放脚本和官方归一化处理。环境、数据、权重、视频、日志均不提交 Git。

## 🤗 SmolVLA：200条数据微调

完整编排入口（正在运行的实验不要重复启动）：

```bash
.venv/bin/python model/run_smolvla.py --output outputs/smolvla20261024
```

🔗 原50条＋补采150条 → 官方API合并与验证 → 官方预训练SmolVLA微调 → 10个新场景评估与视频。数据准备调用 `myscript/expand_dataset.py`，权重准备调用 `model/download_smolvla.py`，训练和评估复用现有 `train_and_eval.py`。

🧠 使用官方 [smolvla_base](https://huggingface.co/lerobot/smolvla_base)，约450M总参数，按默认设置冻结视觉/语言主干，微调约100M参数与投影层。真实batch16已通过本机8GB显存检查；BF16、30,000次参数更新、480,000次样本呈现，200条按episode留出20%验证。任务指令、相机和动作单位沿用采集设置。算法与优化器继承官方base配置，不是官方验证过的本地SO101任务配方。

📷 官方 `rename_map` 将front/wrist映射为camera1/camera2；第三相机未提供，官方SmolVLA按可用相机处理。图像由官方模型缩放补边，统计由官方trainer按新数据处理，推理复用保存的映射与统计。

📂 `pipeline_status.json` 看采集/等待/训练阶段，`data_status.json` 和 `collect150.log` 看补采进度，训练开始后看 `RUN.md` / `train.log`，评估后看 `eval/`。关机后后台进程会停止，不会自动开机恢复。

数据和权重准备好后，单独训练入口：`.venv/bin/python model/train/smolvla.py`；完整官方微调说明见 [SmolVLA文档](https://huggingface.co/docs/lerobot/smolvla)。

## 📂 实验目录

### 🎯 单条固定场景专项训练

```bash
.venv/bin/python model/train/act_fixed_scene.py
```

使用 `configs/act_fixed_scene.json`，通过官方 trainer 从15,000步权重追加学习5,000步；仅用 `data/fixed_scene_seed200`，不划独立验证集。归一化按该子集统计处理，优化器重新初始化。训练完成后自动评估同一场景3次并保存视频，不代表泛化能力。

输出在当日模型目录下的 `fixed_scene_train/`，已有同名实验则新建带序号的目录；查看其中的 `README.md`、`run.json` 和 `train.log`。正式基线模型不覆盖。

Dropout单变量对照（同样从原15,000步权重出发）：

```bash
.venv/bin/python model/train/act_fixed_scene.py --dropout 0 --audit \
  --compare-with outputs/act20261023/fixed_scene_train
```

训练仍为5,000步，仅通过官方参数 `--policy.dropout=0.0` 覆盖Dropout。结束后自动正常推理3次、检查专家观测逐帧误差，并输出 `comparison.json`；不会使用MC Dropout评估。默认在当日目录建立 `fixed_scene_dropout0/`，已有同名目录则加序号。

固定场景比较入口：`model/diagnose/compare_checkpoints.py`。它依次调用现有播放脚本，比较5个已存模型，并额外重复测试选中的模型3次；不会训练。

```bash
.venv/bin/python model/diagnose/compare_checkpoints.py \
  --output outputs/act20261023/fixed_scene_compare
```

输出目录必须不存在，避免覆盖以前的结果。该实验只检查 seed 200 的训练场景，不代表泛化能力。

按模型名＋北京时间日期分组，例如 `outputs/act20261023/`、`outputs/smolvla20261025/`。先看 [实验导航](../outputs/README.md)。新ACT训练默认保存到当日组内的 `train/`，已存在则使用 `train_02/` 等；新评估保存到组内的 `eval_时分秒/`。也可用 `--output_dir`（训练）或 `--output`（评估）指定组内目录。恢复训练沿用该配置保存的位置。

固定场景数据来自现有 50 条中的第 0 条：`data/fixed_scene_seed200`，准备入口 `myscript/prepare_fixed_scene.py`，不是重新采集。评估时同时传 `--dataset data/fixed_scene_seed200 --seeds 200`；只指定 dataset 不会固定场景。仅单条训练时应设 `--dataset.eval_split=0`，不能把它称为独立验证成绩。当前尚未启动新的训练。

### 🔬 专家画面逐帧测试

`model/diagnose/teacher_forcing.py` 使用官方数据预处理与模型接口，将逐帧预测、10步队列和已有闭环记录对照；不会训练。默认检查固定场景专项训练的最终模型。

```bash
.venv/bin/python model/diagnose/teacher_forcing.py \
  --output outputs/act20261023/teacher_forcing_new
```

输出逐帧CSV、指标JSON和曲线PNG；输出目录不能存在。可对已有结果使用 `--plot-only --output 输出目录` 重画曲线。

训练/推理模式隔离入口：`model/diagnose/training_modes.py --output 新实验目录`。它仅对固定检查点的17个专家观测测试VAE、Dropout与BF16机制，不更新参数。当前结果见 [原因分析](../outputs/act20261023/cause_analysis/README.md)。

剩余失败的阶段定位：`model/diagnose/remaining_failures.py --output 新实验目录`，精确重放已有动作、测量物体轨迹，并由专家完成不同前缀后交给模型。`--prefixes 425 743 --skip-replay` 可只测试指定接管点。它是诊断，接管成功包含专家贡献，不能作为独立任务成功率。当前结果见 [阶段排查](../outputs/act20261023/remaining_failures/README.md)。

固定输入正对照：`model/diagnose/pipeline_control.py --output 新目录`，用官方trainer拟合1个观测/动作300步，另1个观测只防止统计方差退化，不参与训练或验证。原始300步及后续同预算学习率对照见 [完整记录](../outputs/act20261023/pipeline_positive_control/README.md)。这个检查不代表任务成功。

## 🌊 Diffusion Policy

推荐使用与原ACT主实验对齐训练预算的配置：

```bash
.venv/bin/python model/train_and_eval.py \
  --config model/configs/diffusion_batch8.json \
  --output outputs/diffusion20261023/batch8
```

📊 同一数据划分、BF16、有效batch8、15,000次参数更新、120,000次样本呈现。8GB显存使用batch4＋官方梯度累积2次；本版trainer的steps计micro-step，所以配置为30,000步，warmup同步换算为1000步。训练后自动评估seed10000—10009并保存视频。目录存在时拒绝覆盖，重跑请换组内子目录。累积时BatchNorm每批只见4条，不能声称与真实batch8数值完全相同。

🔄 如果训练已完成、评估被关机等情况中断，原命令追加 `--eval-only` 即可只重跑评估。必须指定原 `--output`，会核对最终checkpoint；旧评估文件移入 `interrupted/` 保留，不会重训或覆盖权重。机器关机后脚本不会自行恢复。

下面是此前预算较短的batch1入口，保留用于复现历史结果：

```bash
.venv/bin/python model/train/diffusion.py
```

完整训练后自动评估10个新场景：

```bash
.venv/bin/python model/train_and_eval.py --config model/configs/diffusion.json
```

统一保存到当日模型目录，状态见 `RUN.md` / `run.json`，进度见 `train.log`；已有正式训练目录不会覆盖。默认15,000步、batch1、BF16，模型算法参数继承官方默认。训练完成后加载last checkpoint正常推理，不启用ACT专用诊断参数。

模型参数继承官方DiffusionConfig默认；数据与运行参数见 configs/diffusion.json。该命令默认正式训练15,000步、batch1，仅约0.28遍训练数据；已有正式结果与预算限制见 [结果说明](../outputs/diffusion20261023/REPORT.md)。接入测试另见 [接口检查](../outputs/diffusion20261023/interface_check/README.md)。

ACT与Diffusion入口共用 train/policy.py 启动器，都调用官方 lerobot_train；播放不再假设所有模型都有 chunk_size，Diffusion使用 horizon 与自身历史队列。
