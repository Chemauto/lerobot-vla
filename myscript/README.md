# 🤖 SO-101 仿真数据工厂

在 **[LeRobot 官方 MuJoCo 场景](https://huggingface.co/lerobot/so101-tabletop-mujoco)**中，让机械臂自动把四个方块放进绿盒子，再保存为可训练的数据。

🧠 **流程**：读取方块位置 → 求解抓取姿态 → 平滑抓放 → 失败重试 → 只保留成功回合。

- 🦾 夹爪朝下，并对齐方块方向，减少推走和掉落。
- 📷 保存前视/腕部画面、关节状态、动作和任务指令。
- 📦 输出 LeRobot 格式，33Hz；记录随机种子，便于复现。
- 🔒 不改官方源码、XML或摩擦参数；这是脚本专家，还不是训练好的策略。

![官方环境自动抓放](assets/official_success.jpg)

🧪 **本轮验证：20个测试种子全部成功**（小样本，不代表任意场景都能成功）。

🚀 **[快速开始](QUICKSTART.md)** · 📄 **[诊断记录](../outputs/diagnosis/REPORT.md)**

| 脚本 | 用途 |
| --- | --- |
| `official_teacher.py` | 自动抓放、重试与视频演示 |
| `collect_official.py` | 采集成功演示数据 |
| `official_env.py` | 加载固定版本官方环境 |
| `official_ik.py` | 计算关节航点 |
| `validate_official_data.py` | 检查训练数据和视频 |

📦 **已采集50条成功演示**，共66,451帧：[数据目录](../data/official_tabletop_50)。
