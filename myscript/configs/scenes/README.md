# 🧩 仿真场景

场景统一放这里，不再写到outputs或task_assets目录：

| 文件 | 用途 |
| --- | --- |
| `one_fixed.xml` | 单方块，固定摆放 |
| `one_random.xml` | 单方块，随机摆放 |
| `four_fixed.xml` | 四方块，固定摆放 |

四方块随机组直接加载固定版本的官方XML，无需再复制一份。相机、随机化范围和控制频率由上一级四个JSON配置设置。

这些XML由 [task_variants.py](../../task_variants.py) 从固定版本官方场景生成，首次运行自动创建，删掉后可以重建。XML引用本机Hugging Face缓存中的纹理，因此不加入Git；生成代码和本说明加入Git。官方文件和物理参数不变。
