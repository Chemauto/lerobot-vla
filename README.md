# 🤖 SO-101 仿真数据工厂

基于 [LeRobot 官方 MuJoCo 环境](https://huggingface.co/lerobot/so101-tabletop-mujoco)，自动完成四方块入盒，生成可训练的专家演示。

![官方场景演示](myscript/assets/official_success.jpg)

🧠 方块位姿 → 有界IK → 平滑抓放 → 失败重试 → 成功检查 → LeRobot数据集。

- 🦾 对齐夹爪与方块方向，补偿放置偏差；不改官方源码或物理参数。
- 📷 前视＋腕部RGB、6维关节状态与动作，33Hz，严格对齐动作前观测。
- ✅ 20个测试种子全部完成；单/四方块、固定/随机四组各200条已采集并通过检查。结果属于脚本专家，不是训练策略，亦非任意场景保证。
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

详见 [快速开始](myscript/QUICKSTART.md) 与 [脚本说明](myscript/README.md)。当前四组数据见 [数据说明](data/README.md)，不会随仓库提交；换机器需重新采集或自行复制数据。

## 🧠 训练与评估

```bash
# 指定数据集训练ACT；多个数据集会依次训练
.venv/bin/python model/run_experiment.py --model act \
  --datasets one_fixed_200 one_random_200 four_fixed_200 four_random_200
# 微调SmolVLA时，把act换成smolvla；Diffusion用diffusion
```

数据须完成验证后才能训练。每组保存latest/best两个checkpoint、3段视频、README结果表和CSV；统一位于 `outputs/模型名日期/数据集名/`。详细用法见 [模型说明](model/README.md)。旧实验和权重已移至 `archive/experiments_before_20261028/`，旧数据已删除。

## 🔗 当前流程状态

自动采集、数据检查、官方ACT/Diffusion训练、模型加载、闭环评估与视频记录均已运行验证；完整四方块任务均为0/10。最新Diffusion平均最终入盒0.9块，结果仍未达标。现有数据未发现损坏或格式问题，不能将失败单独归因于数据或模型。

此前的200条数据已完成合并与检查，SmolVLA曾微调30,000步并评估10场：完整成功0/10，平均最终入盒0.8块。没有修改官方场景物理，也没有继续ACT任务专项调参。

项目重点是可复用的采集→训练→评估流程。证据、限制与问题分类见 [流程验收说明](PIPELINE_STATUS.md)。

📦 四组各200条均已验证，ACT正在依次训练。项目分工：`myscript/`负责通用采集与IK，`model/`负责通用训练评估。
