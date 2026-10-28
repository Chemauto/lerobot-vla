# 🦾 通用仿真数据工具

这里放 **IK、脚本专家、数据采集和检查**。指定场景配置即可复用；本目录不编排某次训练，也不保留失误排查代码。

| 文件 | 用途 |
| --- | --- |
| ⭐ [collect_official.py](collect_official.py) | 自动执行专家，保存成功示范为LeRobot数据集 |
| [official_teacher.py](official_teacher.py) | 接近 → 抓取 → 搬运 → 松爪 → 检查，支持重试 |
| [official_ik.py](official_ik.py) | 有界IK，求机械臂关节目标 |
| [official_env.py](official_env.py) | 加载固定版本官方MuJoCo环境 |
| [task_variants.py](task_variants.py) | 单/四方块、固定/随机场景适配 |
| [validate_official_data.py](validate_official_data.py) | 格式、完整视频解码、物理回放检查 |
| [media_utils.py](media_utils.py) | 视频编码公共函数 |
| [download_assets.py](download_assets.py) | 下载固定版本官方资产 |
| `configs/`、[configs/scenes/](configs/scenes/README.md) | 场景参数与自动生成的XML |

## 📷 采数据

在仓库根目录运行，例如使用单方块固定场景采10条：

```bash
.venv/bin/python myscript/collect_official.py \
  --config myscript/configs/one_fixed.json \
  --episodes 10 --max-attempts 30 --seed 30000 --preview \
  --root data/my_demos

.venv/bin/python myscript/validate_official_data.py \
  --root data/my_demos --decode-all --replay-physics
```

📦 每帧记录“动作前双相机图像＋6维状态 → 专家动作”，33Hz；关节角为度，夹爪0–100。目标数量、随机种子和初末状态写入manifest，源代码与配置随数据保存。已有数据目录拒绝覆盖。

🔒 官方源码和接触物理不改。单方块和固定摆放属于本地场景适配；随机化、相机等由JSON指定。验证完成的数据才交给模型训练。

🚀 [快速开始](QUICKSTART.md) · 📦 [当前数据说明](../data/README.md)
