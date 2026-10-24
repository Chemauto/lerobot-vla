# 🔍 官方源码与本地配置的区别

## 源码来源

实际安装使用的LeRobot固定在commit `8a125584153240d7511dddfbda440bbee56b134a`。本机核对了训练器、ACT/Diffusion模型与配置、数据集/策略factory的7个文件，与该干净checkout逐文件SHA256一致，见 `OFFICIAL_SOURCE_CHECK.json`。没有修改这些官方文件。

2026-10-24再次核对：实际导入的LeRobot包内 **592个Python文件** 全部与该commit的Git对象一致，无缺失、额外或修改的Python文件；官方远端commit已核实。完整记录见 [OFFICIAL_SOURCE_AUDIT.json](OFFICIAL_SOURCE_AUDIT.json)。本地采集控制、配置与评估适配不属于这项官方源码一致性检查，因此该检查不证明整条适配流程没有bug。

官方SO101场景固定revision `12e504e860c369aad1a9a9eab3fdfbac658ed6dd`，本地脚本负责配置、动作来源、数据与评估记录。

## 以前的ACT实验不是全默认配置

| 项目 | 官方ACT默认 | 本地ACT主实验 |
|---|---|---|
| chunk_size | 100 | 50 |
| n_action_steps | 100 | 10 |
| Dropout | 0.1 | 原主实验0.1；后续诊断改0 |
| 优化器基础学习率 | 1e-5 | 原主实验1e-5；最小正对照另试1e-6 |
| 数据 | 由使用者指定 | 本地50条或单条子集 |
| 训练预算 | 训练器默认100,000步 | 15,000或5,000步 |
| 精度 | 运行参数 | 本地主要用BF16，部分诊断FP32 |

这叫“调用官方代码，使用本地实验配置”，不能称为完整照搬官方已验证任务配方。相机映射、33Hz、初始关节等也是本地数据适配参数，虽通过官方接口配置，仍是适配选择。

## 新Diffusion对照

`configs/diffusion.json`的policy部分仅指定type、device和不上传模型；架构、归一化、历史长度、预测长度、执行长度、噪声过程、学习率与scheduler继承已安装的官方DiffusionConfig默认值。当前默认n_obs_steps=2、horizon=64、n_action_steps=32。

数据路径、划分、训练步数、batch=1、BF16、worker与日志保存由本地设置，主要适配显存与工程运行。这些运行设置并不是“官方已验证SO101训练配方”。官方Diffusion默认配置原本面向PushT，不能据其来源保证本地SO101成功。

新模型先做训练→保存→加载→仿真短程接口测试；极少训练步数的结果不评价抓取能力。不继续ACT任务专项调参。

## Diffusion 与 ACT 对齐训练预算

`configs/diffusion_batch8.json`：同一50条数据、20% episode验证划分、seed1000、BF16；batch4、官方梯度累积2次，30,000个micro-step对应15,000次参数更新、120,000次样本呈现，与原ACT主实验预算一致。实际batch8及开启官方U-Net checkpointing均发生OOM，容量测试记录保存在当日实验的`batch8/setup/`。

固定版本trainer的总步数与scheduler均按micro-step计算，因此warmup由默认500改为1000个micro-step，保持500次参数更新的warmup；cosine总长度由trainer自动使用30,000。其余Diffusion算法参数继承官方默认，从头训练，不接续此前已衰减学习率的15,000步短预算模型。没有改官方源码。

这是训练预算对照：不同模型的结构、优化器预设与动作窗口仍不同；梯度累积时BatchNorm批统计来自4条样本，不能当作真实batch8的完全等价实现。自动评估仍使用seed10000—10009、官方成功判定和正常推理。

## SmolVLA 200条微调

`configs/smolvla.json`从官方`lerobot/smolvla_base`权重初始化，base revision为`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`，VLM revision为`7b375e1b73b11138ff12fe22c8f2822d8fe03467`。下载器固定这两个版本，保存到不进Git的`checkpoints/`；实际来源记录在实验的`assets_status.json`。

默认chunk50、执行50、冻结视觉/语言主干、训练expert与state投影，模型约450M/可训练约100M；归一化、512补边、flow matching、优化器与学习率调度均继承官方base配置。替换本地VLM文件路径、通过官方rename_map映射两路相机，不修改官方实现。接口检查核对完整checkpoint键、抽样冻结主干权重和4个安装源码文件，记录见`interface_check/READY.json`。

数据由原50条＋新增150条构成，官方merge_datasets合并，保留来源与唯一随机种子；原50条的动作、状态和时间戳逐项校验一致。160条训练/40条验证，seed1000、真实batch16、BF16、30,000步（无梯度累积），480,000次样本呈现。这些是本地预算选择，不是官方SO101成功配方；本次同时扩大数据和更换预训练模型，不能将效果变化单独归因于其中一个因素。
