# YOLO26 任务流程入口使用说明

本文档说明如何使用当前项目中已经实现的三个正式入口：

```text
src.predict_one
src.predict_batch
src.evaluate_val
```

这些入口用于先跑通第四节的完整流程：单张图片预测、批量生成 JSON、对验证集评分。

## 1. 环境准备

在项目根目录运行命令。本文档中的命令都假设当前目录是：

```text
C:\Users\Rainluck\Desktop\Final Project
```

建议使用项目自带虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

当前默认模型是：

```text
yolo26n.pt
```

如果项目根目录下已经有 `yolo26n.pt`，运行时会直接使用本地权重。如果没有，Ultralytics 会尝试自动下载。

## 2. 单张图片预测

用于测试某一张图片的预测结果。

命令：

```powershell
.\.venv\Scripts\python.exe -m src.predict_one --image val_set\val_1.png --model yolo26n.pt --conf 0.25
```

参数说明：

```text
--image   输入图片路径
--model   YOLO 权重路径或模型名称，默认推荐 yolo26n.pt
--conf    置信度阈值，默认 0.25
```

输出示例：

```json
{"cat": 2, "sheep": 2}
```

输出含义是：模型在该图片中检测到 2 只猫和 2 只羊。

注意：当前使用的是 COCO 预训练 YOLO26，所以只能较好识别部分动物类别。没有被 COCO 覆盖的类别，例如 `fox`、`wolf`、`goose`、`duck`、`tiger`、`lion` 等，后续需要通过自建数据训练来提升。

## 3. 批量预测图片文件夹

用于处理一个文件夹中的所有图片，并生成最终评测需要的 JSON 格式。

命令：

```powershell
.\.venv\Scripts\python.exe -m src.predict_batch --input val_set --output outputs\val_predictions.json --model yolo26n.pt --conf 0.25
```

参数说明：

```text
--input    输入图片文件夹
--output   输出 JSON 文件路径
--model    YOLO 权重路径或模型名称
--conf     置信度阈值
```

支持的图片后缀包括：

```text
.png
.jpg
.jpeg
.bmp
.webp
```

输出 JSON 示例：

```json
{
  "val_1.png": {
    "cat": 2,
    "sheep": 2
  },
  "val_2.png": {}
}
```

如果某张图片没有检测到支持的动物类别，会输出空字典 `{}`。这样可以保证每张输入图片都有一条预测结果。

## 4. 验证集评分

用于对比预测 JSON 和老师提供的 `ground_truth.json`，按照 PDF 中的评分规则计算每张图得分和平均分。

命令：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_val --pred outputs\val_predictions.json --gt val_set\ground_truth.json
```

参数说明：

```text
--pred   模型生成的预测 JSON
--gt     标准答案 JSON
```

输出示例：

```text
val_1.png: 7.50
val_10.png: 0.00
...
Average: 27.08
```

评分规则来自项目 PDF：

1. 每张图片 100 分。
2. 如果 ground truth 中有 `N` 个动物类别，每个类别占 `100 / N` 分。
3. 类别识别正确，得到该类别一半分数。
4. 数量也预测正确，得到另一半分数。
5. 多预测一个 ground truth 中不存在的动物类别，扣 5 分。
6. 每张图片最低分为 0。

## 5. 推荐完整流程

第一次跑通项目时，建议按下面顺序执行。

先测试单张图片：

```powershell
.\.venv\Scripts\python.exe -m src.predict_one --image val_set\val_1.png --model yolo26n.pt --conf 0.25
```

再批量预测验证集：

```powershell
.\.venv\Scripts\python.exe -m src.predict_batch --input val_set --output outputs\val_predictions.json --model yolo26n.pt --conf 0.25
```

最后计算验证集分数：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_val --pred outputs\val_predictions.json --gt val_set\ground_truth.json
```

## 6. 调整置信度阈值

可以用不同 `--conf` 值测试效果。例如：

```powershell
.\.venv\Scripts\python.exe -m src.predict_batch --input val_set --output outputs\val_predictions_conf040.json --model yolo26n.pt --conf 0.40
```

然后评分：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_val --pred outputs\val_predictions_conf040.json --gt val_set\ground_truth.json
```

一般来说：

1. 阈值较低，可能检测更多目标，但误检也更多。
2. 阈值较高，误检可能减少，但漏检也可能增加。

当前验证集中，`0.40` 的结果比 `0.25` 略好，但这不代表最终评测一定也是这样。最终提交前应多测试几个阈值。

## 7. 最终评测时怎么用

假设老师发布的最终评测图片放在：

```text
eval_set/
```

可以运行：

```powershell
.\.venv\Scripts\python.exe -m src.predict_batch --input eval_set --output outputs\eval_predictions.json --model yolo26n.pt --conf 0.40
```

生成后检查：

```powershell
.\.venv\Scripts\python.exe -m json.tool outputs\eval_predictions.json
```

确认：

1. JSON 能正常解析。
2. 每张评测图片都有结果。
3. key 是图片文件名。
4. value 是动物计数字典。
5. 没有非动物类别。
6. 没有数量为 0 的类别。

## 8. 当前实现的限制

当前流程已经能完整跑通，但它还是 baseline：

1. 使用的是 `yolo26n.pt` 预训练模型，不是针对本项目 20 类动物训练的模型。
2. 只保守映射 COCO 中明确支持的动物类别。
3. 不会把 `bird` 强行映射成 `duck`、`goose` 或 `chicken`。
4. 对 `fox`、`wolf`、`tiger`、`lion`、`pig`、`rabbit`、`monkey` 等类别，当前预训练模型可能无法正确输出。

要提高最终准确率，需要继续完成数据收集、标注和 YOLO26 微调训练。

