# Deep Learning Final Project Step-by-Step Guide

本文档是一份从零到提交的执行指南，目标是帮助你完成 `Final_Project_task.pdf` 中的动物检测与计数项目。

项目最终要做的事情很明确：输入一批图片，输出每张图片中指定动物类别的数量，保存为 JSON 文件。核心难点不是把 YOLO 跑起来，而是让结果稳定满足评测格式，并在最终评测当天半小时内可靠生成提交文件。

## 1. 项目目标

你需要完成一个动物检测与计数系统：

```text
输入：单张图片
输出：Python/JSON 字典风格的动物计数结果
```

示例：

```json
{
  "cat": 2,
  "duck": 1,
  "deer": 1
}
```

最终批量提交格式：

```json
{
  "eval_000001.png": {"cat": 2, "duck": 1, "deer": 1},
  "eval_000002.png": {"cat": 2, "bear": 2, "deer": 1}
}
```

输出要求：

1. 只输出动物类别。
2. 只输出图片中出现的动物。
3. 不要输出数量为 0 的类别。
4. 每个数量必须是大于等于 1 的整数。
5. JSON 中不要写自然语言解释。

## 2. 评测类别

最终答案只能使用以下英文类别名作为 JSON key：

```text
cat
dog
horse
cow
sheep
goat
pig
rabbit
chicken
duck
goose
deer
monkey
fox
wolf
bear
tiger
lion
zebra
giraffe
```

建议在代码中定义一个固定白名单：

```python
ANIMAL_CLASSES = {
    "cat", "dog", "horse", "cow", "sheep", "goat", "pig", "rabbit",
    "chicken", "duck", "goose", "deer", "monkey", "fox", "wolf",
    "bear", "tiger", "lion", "zebra", "giraffe"
}
```

任何不在白名单里的类别都不要写入最终 JSON。

## 3. 推荐项目目录

建议整理成下面的结构：

```text
Final Project/
├── Final_Project_task.pdf
├── 报告模板.docx
├── PROJECT_GUIDE.md
├── val_set/
│   ├── ground_truth.json
│   ├── val_1.png
│   ├── ...
│   └── val_10.png
├── src/
│   ├── predict_one.py
│   ├── predict_batch.py
│   ├── evaluate_val.py
│   ├── classes.py
│   └── utils.py
├── models/
│   └── best.pt
├── datasets/
│   ├── raw/
│   ├── yolo/
│   └── data.yaml
├── outputs/
│   ├── val_predictions.json
│   └── eval_predictions.json
├── reports/
│   └── project_report.docx
└── presentation/
    └── project_presentation.pptx
```

如果你暂时不训练自己的模型，也至少保留 `src/`、`models/`、`outputs/`，方便最终打包。

## 4. 环境准备

建议使用 Python + Ultralytics YOLO。

### 4.1 安装 Python

推荐 Python 3.10 或 3.11。

检查命令：

```powershell
python --version
```

如果本机没有 Python，需要先安装。安装后重新打开 PowerShell。

### 4.2 创建虚拟环境

在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4.3 安装依赖

```powershell
pip install ultralytics opencv-python pillow tqdm
```

如果要做标注或数据处理，可以再装：

```powershell
pip install label-studio fiftyone albumentations
```

最小可运行依赖通常是：

```text
ultralytics
opencv-python
pillow
tqdm
```

## 5. 第一阶段：先跑通验证集流程

不要一开始就纠结训练。先保证系统能读取图片、输出 JSON、计算验证集分数。

当前目录已有：

```text
val_set/
├── ground_truth.json
├── val_1.png
├── ...
└── val_10.png
```

`ground_truth.json` 的格式是：

```json
{
  "val_1.png": {
    "lion": 2,
    "fox": 1,
    "goose": 2,
    "cat": 1
  }
}
```

你应该先实现三个脚本：

1. `predict_one.py`：输入一张图片，输出一个字典。
2. `predict_batch.py`：输入一个文件夹，输出完整 JSON。
3. `evaluate_val.py`：对比 `ground_truth.json`，计算验证集分数。

## 6. 第二阶段：实现基础 YOLO 推理

### 6.1 先使用预训练模型

可以先用 YOLO 预训练模型测试流程，例如：

```python
from ultralytics import YOLO

model = YOLO("yolo11x.pt")
results = model("val_set/val_1.png")
```

注意：COCO 预训练模型并不覆盖全部项目类别，例如 fox、wolf、goose、deer 可能识别不好或没有对应类别。所以预训练模型主要用于跑通框架，不一定能取得好分数。

### 6.2 类别映射

YOLO 模型输出的类别名不一定和项目要求完全一致。需要写一个映射表。

示例：

```python
CLASS_MAP = {
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "bird": "duck"
}
```

不要盲目把 `bird` 全部映射成 `duck`，这只是临时方案。最终应该用训练数据或更强模型区分 chicken、duck、goose。

### 6.3 计数逻辑

推理后对检测结果做：

1. 读取每个检测框的类别名。
2. 映射成项目类别名。
3. 如果不在白名单，跳过。
4. 如果置信度低于阈值，跳过。
5. 按类别计数。
6. 删除数量为 0 的类别。

伪代码：

```python
counts = {}

for detection in detections:
    raw_name = detection.class_name
    conf = detection.confidence

    if conf < CONF_THRESHOLD:
        continue

    name = CLASS_MAP.get(raw_name)
    if name not in ANIMAL_CLASSES:
        continue

    counts[name] = counts.get(name, 0) + 1
```

## 7. 第三阶段：提高准确率

预训练 COCO 模型不够用时，有三条路线。

### 路线 A：直接用更强的开放词汇检测模型辅助

如果时间很紧，可以考虑用支持文本提示的检测模型辅助，例如 Grounding DINO、OWL-ViT 或一些视觉语言模型。但最终输出仍必须是 JSON。

优点：

1. 不一定需要大量标注。
2. 对项目指定类别可能更灵活。

缺点：

1. 环境更复杂。
2. 推理速度可能慢。
3. 最终半小时评测要确保能跑完。

### 路线 B：收集数据并微调 YOLO

这是最稳妥、也最符合报告要求的路线。

你需要为 20 个动物类别收集图片并标注 bounding box。

推荐每类至少：

```text
最低目标：30-50 张
较好目标：100+ 张
```

图片类型要覆盖：

1. 单只动物。
2. 多只同类动物。
3. 多种动物同图。
4. 小目标。
5. 遮挡。
6. 背景复杂。
7. 合成图风格，如果最终评测是 synthetic set，这点尤其重要。

### 路线 C：混合方案

可以用 YOLO 检测常见类别，再用手工规则或第二模型补充难类别。

例如：

1. YOLO 检测 cat、dog、horse、cow、sheep、bear、zebra、giraffe。
2. 自训练模型检测 fox、wolf、goose、deer、monkey、tiger、lion 等。
3. 最后统一合并为一个 JSON。

注意合并时要避免重复计数。

## 8. 数据集构建

### 8.1 YOLO 数据格式

YOLO 检测数据通常是：

```text
datasets/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

每张图片对应一个同名 `.txt` 标注文件。

例如：

```text
images/train/cat_001.jpg
labels/train/cat_001.txt
```

标注文件内容：

```text
class_id x_center y_center width height
```

坐标都是 0 到 1 之间的归一化值。

### 8.2 data.yaml

`datasets/data.yaml` 示例：

```yaml
path: datasets/yolo
train: images/train
val: images/val
names:
  0: cat
  1: dog
  2: horse
  3: cow
  4: sheep
  5: goat
  6: pig
  7: rabbit
  8: chicken
  9: duck
  10: goose
  11: deer
  12: monkey
  13: fox
  14: wolf
  15: bear
  16: tiger
  17: lion
  18: zebra
  19: giraffe
```

类别顺序必须和标注文件中的 `class_id` 一致。

### 8.3 标注建议

标注时只框动物主体：

1. 尽量完整包住身体。
2. 遮挡时框住可见部分。
3. 多只动物分别框。
4. 不要把背景、笼子、食盆、人、树框进去。
5. 同一只动物不要重复标注。

## 9. 模型训练

用 Ultralytics 训练示例：

```powershell
yolo detect train model=yolo11m.pt data=datasets/data.yaml epochs=80 imgsz=640 batch=8
```

如果显存不够：

```powershell
yolo detect train model=yolo11s.pt data=datasets/data.yaml epochs=80 imgsz=640 batch=4
```

如果时间充足且显卡较强：

```powershell
yolo detect train model=yolo11l.pt data=datasets/data.yaml epochs=100 imgsz=768 batch=4
```

训练完成后通常会生成：

```text
runs/detect/train/weights/best.pt
```

把它复制到：

```text
models/best.pt
```

报告里要记录：

1. 使用的基础模型，例如 YOLO11m。
2. 训练轮数。
3. 图片尺寸。
4. batch size。
5. 数据量。
6. 数据增强方法。
7. 验证集表现。

## 10. 验证集评估

### 10.1 生成验证集预测

目标命令应该类似：

```powershell
python src/predict_batch.py --input val_set --output outputs/val_predictions.json --model models/best.pt
```

输出格式必须是：

```json
{
  "val_1.png": {"lion": 2, "fox": 1, "goose": 2, "cat": 1},
  "val_2.png": {"lion": 3, "cat": 2, "duck": 1}
}
```

### 10.2 计算分数

目标命令应该类似：

```powershell
python src/evaluate_val.py --pred outputs/val_predictions.json --gt val_set/ground_truth.json
```

评分规则：

1. 每张图 100 分。
2. 若真实答案有 N 个类别，每个类别占 `100 / N`。
3. 预测到类别，得该类别一半分。
4. 数量也正确，得另一半。
5. 每多预测一个真实答案里没有的类别，扣 5 分。
6. 每张图最低 0 分。

你应该在验证脚本中输出：

```text
val_1.png: 83.33
val_2.png: 66.67
...
Average: 75.20
```

### 10.3 错误分析

每次验证后记录错误类型：

1. 漏检：真实有动物但没预测。
2. 误检：预测了不存在的动物。
3. 数量错误：类别对了但数量不对。
4. 混淆：例如 goose 识别成 duck，wolf 识别成 dog。
5. 小目标失败。
6. 遮挡失败。

这些内容可以直接写进报告 Analysis 部分。

## 11. 阈值调参

置信度阈值会直接影响分数。

阈值太高：

```text
漏检变多，recall 下降。
```

阈值太低：

```text
误检变多，额外错误类别每个扣 5 分。
```

建议测试这些值：

```text
0.20
0.25
0.30
0.35
0.40
0.50
```

对验证集分别生成 JSON 并计算平均分，选择验证集上最好的阈值。

如果模型容易多预测错误类别，宁可稍微提高阈值，因为额外类别会扣分。

## 12. 最终评测当天流程

最终评测时间非常紧：

```text
Jun 14 14:00 发布 evaluation image set
Jun 14 14:30 提交 evaluation response JSON
```

你必须提前准备好一条命令，拿到图片后直接运行。

### 12.1 评测前一天检查

提前完成：

1. 虚拟环境可启动。
2. `models/best.pt` 存在。
3. `predict_batch.py` 能跑通 `val_set`。
4. 输出 JSON 格式正确。
5. 代码不会依赖绝对路径。
6. 如果没有 GPU，也能在 CPU 上跑完，或者确认评测当天机器有 GPU。
7. 准备一个空目录 `eval_set/`。

### 12.2 拿到评测图片后

把评测图片放入：

```text
eval_set/
```

运行：

```powershell
python src/predict_batch.py --input eval_set --output outputs/eval_predictions.json --model models/best.pt
```

然后立刻检查 JSON：

```powershell
python -m json.tool outputs/eval_predictions.json
```

检查重点：

1. 文件名是否和评测图片一致。
2. 是否每张图都有结果。
3. 是否有非动物类别。
4. 是否有数量为 0 的类别。
5. 是否 JSON 语法有效。

### 12.3 提交文件

提交：

```text
outputs/eval_predictions.json
```

如果老师要求特定文件名，按老师要求改名。

## 13. 报告写作结构

报告建议按下面结构写。

### 13.1 Introduction

写清楚：

1. 项目目标：动物检测与计数。
2. 输入输出定义。
3. 为什么使用 YOLO。
4. 最终系统能处理多类别、多实例动物图片。

### 13.2 Methodology

写清楚：

1. 整体 pipeline。
2. 数据来源。
3. 数据清洗方法。
4. 标注方法。
5. 数据增强。
6. 模型结构和基础模型。
7. 训练参数。
8. 推理流程。
9. 类别白名单。
10. 置信度过滤。
11. NMS 或重复检测处理。
12. JSON 输出格式。

### 13.3 Experiments

写清楚：

1. 使用的数据量。
2. 训练/验证划分。
3. 不同模型或阈值对比。
4. 验证集结果。
5. 最终评测结果，如果已公布。

可以放一个表：

```text
Model       Img Size   Conf   Val Score
YOLO11s     640        0.25   xx.xx
YOLO11m     640        0.30   xx.xx
YOLO11m     768        0.30   xx.xx
```

### 13.4 Analysis

写清楚：

1. 哪些类别最容易错。
2. 常见混淆，例如 duck/goose/chicken、dog/wolf/fox、lion/tiger。
3. 遮挡和小目标影响。
4. 多动物重叠导致重复计数或漏计数。
5. 你们如何解决这些问题。

### 13.5 Contribution

必须写每个组员的姓名、学号、贡献。

示例：

```text
Student A: dataset collection, annotation, model training.
Student B: inference pipeline, JSON generation, evaluation script.
Student C: report writing, presentation, error analysis.
```

## 14. PPT 展示建议

PPT 建议 8-12 页：

1. 项目目标。
2. 任务定义和输出格式。
3. 数据集构建。
4. 模型方法。
5. 推理 pipeline。
6. 验证集结果。
7. 错误案例分析。
8. 创新点。
9. 分工。
10. 总结。

展示时重点讲你们实际做了什么，不要只讲 YOLO 原理。

## 15. 最终提交清单

最终 zip 命名：

```text
StudentName-1_StudentName-2_project.zip
```

zip 内包含：

```text
StudentName-1_StudentName-2_project_report.pdf
StudentName-1_StudentName-2_project_code.zip
StudentName-1_StudentName-2_project_presentation.ppt
```

代码包建议包含：

```text
src/
models/
datasets/data.yaml
requirements.txt
README.md
outputs/eval_predictions.json
```

不要把非常大的原始数据全部塞进代码包，除非老师明确要求。可以在报告里说明数据来源和构建方式。

## 16. 推荐执行时间线

### 第 1 天

1. 安装 Python 和依赖。
2. 跑通 YOLO 预训练模型。
3. 写出 `predict_one.py` 和 `predict_batch.py`。
4. 能对 `val_set` 生成 JSON。

### 第 2-3 天

1. 写 `evaluate_val.py`。
2. 根据 `ground_truth.json` 计算验证集分数。
3. 做错误分析。
4. 调 confidence threshold。

### 第 4-7 天

1. 收集和标注训练数据。
2. 整理 YOLO 数据格式。
3. 训练第一版自定义模型。
4. 用验证集评估。

### 第 8-10 天

1. 补充难类别数据。
2. 处理易混淆类别。
3. 训练第二版模型。
4. 固定最终推理命令。

### 最终评测前

1. 冻结代码。
2. 冻结模型。
3. 在 `val_set` 上完整演练一次。
4. 准备最终提交 JSON 的命令。

## 17. 高风险点

最容易丢分的地方：

1. JSON 格式错误。
2. 文件名不匹配。
3. 输出了非动物类别。
4. 输出了数量为 0 的类别。
5. 把不在 20 类列表中的动物写进结果。
6. 评测当天环境跑不起来。
7. 模型路径写成了自己电脑上的绝对路径。
8. 只在单图上测试，没有批量测试。
9. 过度依赖 COCO 预训练模型，导致很多项目类别无法识别。
10. 报告没有写清楚数据构建和成员贡献。

## 18. 最小可交付版本

如果时间非常紧，至少完成：

1. 一个能运行的 `predict_batch.py`。
2. 一个固定的动物白名单。
3. 一个有效 JSON 输出。
4. 一个验证集评分脚本。
5. 一份说明清楚方法、问题和分工的报告。
6. 一份 PPT。

即使模型准确率一般，只要系统完整、报告清楚、代码可运行，仍然能拿到报告、工作量、代码和展示部分的分数。

## 19. 理想版本

理想版本应该做到：

1. 自建或扩充了 20 类动物数据。
2. 微调了 YOLO 模型。
3. 针对 duck/goose/chicken、dog/wolf/fox、lion/tiger 做了额外优化。
4. 有完整自动化批量推理脚本。
5. 有验证集评分和错误分析。
6. 最终评测当天可以一条命令生成 JSON。
7. 报告中有清楚的数据、模型、实验和分析。

## 20. 下一步建议

从现在开始，优先做这三件事：

1. 建立 `src/` 目录并写出批量预测脚本。
2. 用 `val_set` 生成 `outputs/val_predictions.json`。
3. 写评分脚本，对照 `val_set/ground_truth.json` 计算分数。

先把整个系统闭环跑通，再考虑训练更强模型。
