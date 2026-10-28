# 🚀 快速开始

在仓库根目录使用 `.venv/bin/python`，首次安装见 [项目README](../README.md)。

### ① 准备场景 🧩

```bash
.venv/bin/python myscript/download_assets.py --offline
```

首次下载去掉 `--offline`。场景JSON在 `myscript/configs/`，派生XML自动生成到 `configs/scenes/`。

### ② 自动采集 📷

```bash
.venv/bin/python myscript/collect_official.py \
  --config myscript/configs/one_fixed.json \
  --episodes 10 --max-attempts 30 --seed 30000 --preview \
  --root data/my_demos
```

`--episodes`是要保留的成功回合数，失败回合丢弃；已有目录不覆盖。图像/状态记录在动作执行前。

### ③ 检查 ✅

```bash
.venv/bin/python myscript/validate_official_data.py \
  --root data/my_demos --decode-all --replay-physics
```

### ④ 训练 🧠

```bash
.venv/bin/python model/train/act.py --dataset data/my_demos
```

采集与训练是独立工具。替换场景JSON与数据集目录即可用于其他场景。
