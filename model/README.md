# 🧠 通用模型训练与评估

这里只放可复用工具，训练和模型推理调用固定版本 **LeRobot官方API**。数据集由命令行指定，模型参数集中在已有configs/，不再额外拆实验目录。

| 位置 | 用途 |
| --- | --- |
| `train/act.py`、`diffusion.py`、`smolvla.py` | 按模型名称启动，共用统一流程 |
| [run_experiment.py](run_experiment.py) | 指定数据集 → 官方训练 → 选best → 仿真评估 |
| [play/policy.py](play/policy.py) | 加载checkpoint，评估并录视频 |
| `configs/` | 模型基础参数，不绑定具体数据集 |
| [artifacts.py](artifacts.py) | latest/best保留、README表格与CSV |
| [download_smolvla.py](download_smolvla.py) | 下载固定版本预训练权重 |

## 🚀 常用命令

在仓库根目录运行。将 `data/my_demos` 换成自己的、已通过完整检查的本地SO101数据：

```bash
# 可指定任意数据集，不要求200条；也能一次传多个
.venv/bin/python model/train/act.py --dataset data/my_demos --steps 30000
.venv/bin/python model/train/smolvla.py --dataset data/my_demos

# 仅查看配置和命令，不训练
.venv/bin/python model/run_experiment.py --model act --dataset data/my_demos --dry-run

# 单独评估已有模型
.venv/bin/python model/play/policy.py \
  --checkpoint outputs/actYYYYMMDD/my_demos/checkpoints/best/pretrained_model \
  --dataset data/my_demos --episodes 10 --video-count 3 --stable-finish \
  --output outputs/actYYYYMMDD/my_demos/metadata/evaluation
```

可覆盖：`--steps`、`--batch-size`、`--checkpoint-every`、`--config`；中断续训追加 `--resume`，跨天需指定原 `--output`。同一输出不静默覆盖。入口默认包括训练与仿真评估；如只需官方原始训练CLI，可直接使用 `python -m lerobot.scripts.lerobot_train` 和完整配置。

## 📂 结果

`outputs/模型名日期/数据集名/` 下统一保存 `checkpoints/latest/`、`checkpoints/best/`、`videos/`、`results.csv`、`README.md`、`loss.png`、`loss.csv` 和 `metadata/`。

📌 best按同模型同数据集的官方验证损失选，闭环成功率另外测。默认评估10回合，只录前3回合；完整成功要求全部入盒、松爪并稳定2秒，指标定义写进结果README。当前评估器适配SO101环境与图像字段。

📌 当前ACT配置为30,000步、batch12、BF16、6个worker；实际运行参数保存在每组metadata/train_config.json。单元检查在tests/，显存工具在tools/。

📉 loss曲线由已有artifacts.py从官方日志生成：训练总loss、验证loss、L1/KL分量分图展示，PNG与原始CSV在每组结果目录，README直接展示。训练时每分钟更新，不额外平滑；曲线不代表抓取成功率。
