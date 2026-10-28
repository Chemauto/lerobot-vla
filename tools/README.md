# 🛠️ 开发工具

[benchmark_act_batch.py](benchmark_act_batch.py)：以真实数据元信息确定形状，用合成张量测官方ACT的BF16前反向和AdamW显存/计算吞吐。不保存模型，不代表训练或任务评估。

```bash
.venv/bin/python tools/benchmark_act_batch.py \
  --dataset data/one_fixed_200 --batches 8 12 16 \
  --output .cache/my_act_batch_check
```

正式训练运行时不要重复做显存探测。
