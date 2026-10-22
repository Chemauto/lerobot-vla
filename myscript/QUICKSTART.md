# 🚀 快速开始

## ① 进入项目 📂

```bash
# 在仓库根目录执行
```

使用 `.venv`；首次安装见根目录README。官方环境资产可用 `myscript/download_assets.py` 下载。

## ② 看机械臂自动抓放 🦾

```bash
.venv/bin/python myscript/official_teacher.py \
  --episodes 3 --seed 42 --video \
  --out-dir outputs/official_teacher_demo
```

🎬 首个回合的视频和逐回合结果保存在输出目录。

## ③ 自动采数据 📷

```bash
.venv/bin/python myscript/collect_official.py \
  --episodes 5 --max-attempts 30 --seed 42 --preview \
  --root data/my_official_demos
```

📦 `--episodes` 是**要保留的成功回合数**；失败回合自动丢弃。已有目录不会覆盖。

每帧记录：**动作之前的双相机画面 + 6维关节状态 → 实际发出的6维动作**。前5维为度，夹爪为0–100，33Hz。

✅ 放下、开爪并等待后，四个方块仍在盒内，才保存。种子、初始物体状态和控制器版本记录在 `collection_manifest.json` 与 `collector_sources/`。

## ④ 检查数据 ✅

```bash
.venv/bin/python myscript/validate_official_data.py --root data/my_official_demos
```

📦 已有数据：`data/official_tabletop_50/`（50条、66,451帧）。

## 接下来 🎓
