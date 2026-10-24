# 🤖 SO-101 仿真数据工厂

基于 [LeRobot 官方 MuJoCo 环境](https://huggingface.co/lerobot/so101-tabletop-mujoco)，自动完成四方块入盒，生成可训练的专家演示。

![官方场景演示](myscript/assets/official_success.jpg)

🧠 方块位姿 → 有界IK → 平滑抓放 → 失败重试 → 成功检查 → LeRobot数据集。

- 🦾 对齐夹爪与方块方向，补偿放置偏差；不改官方源码或物理参数。
- 📷 前视＋腕部RGB、6维关节状态与动作，33Hz，严格对齐动作前观测。
- ✅ 20个测试种子全部完成；另已采集50条成功演示、66,451帧。结果属于脚本专家，不是训练策略，亦非任意场景保证。
- 📦 环境、数据、视频、模型和日志仅保存在本地，不进入Git。

## 🚀 安装

Python **3.12**；Ubuntu需具备EGL/OpenGL运行库（`libegl1 libgl1 libglib2.0-0`）。

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python myscript/download_assets.py
```

LeRobot源码固定到commit，官方环境资产固定到独立revision，下载在Hugging Face缓存中。`HF_ENDPOINT`可按网络条件配置，不会关闭TLS校验。

## 🎬 运行

```bash
.venv/bin/python myscript/official_teacher.py --episodes 3 --seed 42 --video
.venv/bin/python myscript/collect_official.py --episodes 5 --preview --root data/my_demos
.venv/bin/python myscript/validate_official_data.py --root data/my_demos
```

详见 [快速开始](myscript/QUICKSTART.md) 与 [脚本说明](myscript/README.md)。已有数据位于 `data/official_tabletop_50/`，不会随仓库提交；换机器需重新采集或自行复制数据。

## 🧠 训练与评估

```bash
.venv/bin/python model/train/act.py
.venv/bin/python model/play/policy.py --checkpoint outputs/act20261023/train/checkpoints/last/pretrained_model
```

ACT 使用官方 LeRobot 训练器，默认训练 15,000 步。目录结构、断点续训和视频评估见 [模型说明](model/README.md)。

🤗 **200条数据＋SmolVLA微调**（2026-10-24已完成，复现实验请使用新的输出目录）：

```bash
.venv/bin/python model/run_smolvla.py --output outputs/smolvla20261024/reproduce
```

原50条保留，补采150条 → 官方API合并验证 → 微调官方SmolVLA → 自动评估与录像。本机结果保存在 `outputs/smolvla20261024/`，运行输出不上传Git；可随仓库查看的结果摘要见 [流程验收说明](PIPELINE_STATUS.md)。

## 🔗 当前流程状态

自动采集、数据检查、官方ACT/Diffusion训练、模型加载、闭环评估与视频记录均已运行验证；完整四方块任务均为0/10。最新Diffusion平均最终入盒0.9块，结果仍未达标。现有数据未发现损坏或格式问题，不能将失败单独归因于数据或模型。

200条数据已完成合并与检查，SmolVLA已微调30,000步并评估10场：完整成功0/10，平均最终入盒0.8块。没有修改官方场景物理，也没有继续ACT任务专项调参。

项目重点是可复用的采集→训练→评估流程。证据、限制与问题分类见 [流程验收说明](PIPELINE_STATUS.md)。
