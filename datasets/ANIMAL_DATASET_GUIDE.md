# Animal YOLO Dataset Build Guide

本文档说明如何为本项目构建补充训练数据集。目标是微调 YOLO26，让模型能识别 PDF 要求的 20 类动物，并减少当前 COCO 预训练模型的漏检、误检和计数错误。

## 1. 当前模型问题总结

当前 `yolo26n.pt` 已经跑通第四节流程，但它是 COCO 预训练模型，只能直接覆盖部分动物类别。

在验证集上，`--conf 0.40` 的整体平均分约为 `30.58`。如果只看 YOLO26/COCO 当前可直接识别并映射的类别：

```text
支持类别 GT 出现次数: 16
类别命中: 13/16 = 81.25%
数量正确: 12/16 = 75.00%
多报错误类别: 5 个
```

表现较好的类别：

```text
horse, cow, zebra, dog
```

还需要补强的已有类别：

```text
bear, cat
```

当前主要缺失或弱识别类别：

```text
duck, goose, chicken, pig, rabbit, monkey,
fox, wolf, tiger, lion, deer, goat
```

当前主要误报问题：

```text
sheep, dog, zebra
```

因此，第五节的数据集建设不应该只补 COCO 没有的类别。最终模型要变成 20 类检测模型，所以训练集中必须包含全部 20 类。只是样本数量和精力应向缺失类别、弱类别和易混淆类别倾斜。

## 2. 数据集目录结构

本项目已经预先创建以下目录：

```text
datasets/
  animal_yolo/
    data.yaml
    images/
      train/
      val/
    labels/
      train/
      val/
    raw/
      by_class/
        cat/
        dog/
        horse/
        cow/
        sheep/
        goat/
        pig/
        rabbit/
        chicken/
        duck/
        goose/
        deer/
        monkey/
        fox/
        wolf/
        bear/
        tiger/
        lion/
        zebra/
        giraffe/
      hard_negatives/
      mixed_species/
    notes/
```

各目录用途：

```text
raw/by_class/      原始图片，按主要动物类别暂存
raw/mixed_species/ 多动物、多类别同图的原始图片
raw/hard_negatives 容易误检的背景或混淆图片
images/train/      YOLO 训练图片
images/val/        YOLO 验证图片
labels/train/      YOLO 训练标签
labels/val/        YOLO 验证标签
notes/             数据来源、标注记录、问题样例记录
data.yaml          YOLO26 训练配置
```

`data.yaml` 中的类别顺序已经固定，标注时必须保持一致。

## 3. 类别优先级和建议数量

建议按下面的优先级收集数据。

### 3.1 第一优先级：当前缺失或弱识别类别

这些类别当前几乎不能依靠 COCO 预训练模型稳定识别，应重点补充：

```text
duck, goose, chicken, pig, rabbit, monkey,
fox, wolf, tiger, lion, deer, goat
```

建议每类：

```text
最低目标: 80 张
较好目标: 150 张以上
```

这些类别对最终得分提升最大，因为当前主要丢分来自类别漏检。

### 3.2 第二优先级：已有但不稳定类别

这些类别 YOLO26 能识别，但验证集中仍有漏检或计数错误：

```text
bear, cat
```

建议每类：

```text
最低目标: 80 张
较好目标: 120 张以上
```

重点收集多实例、遮挡、小目标和复杂背景图片。

### 3.3 第三优先级：已有且表现较好的类别

这些类别当前表现相对较好，但仍应保留训练数据，避免 fine-tuning 后退化：

```text
dog, horse, cow, zebra, giraffe
```

建议每类：

```text
最低目标: 30-60 张
较好目标: 80 张以上
```

其中 `dog` 和 `zebra` 还容易作为误报类别出现，应加入更多混淆场景。

### 3.4 特别处理：sheep

`sheep` 当前在验证集中没有真实出现，但模型经常误报 `sheep`。因此 `sheep` 数据集应包含两类样本：

1. 真实 sheep 图片，帮助模型保留 sheep 能力。
2. 与 sheep 容易混淆但不是 sheep 的图片，例如 goose、duck、pig、cow、白色动物、复杂草地背景。

第二类图片不要标成 sheep，而要标成真实类别，或作为无目标 hard negative 使用。

## 4. 易混淆类别成组收集

不要只按单类孤立收集。应专门补充以下混淆组：

```text
duck / goose / chicken
dog / fox / wolf
cat / tiger / lion
pig / sheep / cow
bear / dog / wolf
horse / zebra
```

每个混淆组都应包含：

1. 单只动物图片。
2. 同类多只动物图片。
3. 多类别同图图片。
4. 相似姿态和相似背景图片。
5. 小目标、遮挡、侧面、背面、远景图片。

这样训练出的模型更容易区分类似动物，而不是把所有鸟类都当成同一类，或把 fox/wolf/tiger/lion 误报成 dog。

每个易混淆组建议收集三类图片：

1. 单类别图片，例如只有 duck、只有 goose、只有 chicken。
2. 同组多类别同图，例如 duck + goose、dog + fox、lion + tiger、pig + sheep。
3. 容易误判的背景或姿态，例如白色 goose 在草地上、远景 wolf、侧身 fox、泥地里的 pig。

建议数量：

```text
单类别图: 每类 30-50 张
同组多类别图: 每组 30-80 张
复杂背景/遮挡/远景图: 每组 30-50 张
```

如果时间紧，优先收集：

```text
duck / goose / chicken
dog / fox / wolf
cat / tiger / lion
pig / sheep / cow
```

这些组最容易影响最终分数，也最容易产生错误类别扣分。

## 5. 图片收集规范

收集原始图片时，先放入：

```text
datasets/animal_yolo/raw/by_class/<class_name>/
```

如果一张图中有多个类别，放入：

```text
datasets/animal_yolo/raw/mixed_species/
```

建议命名格式：

```text
class_source_index.jpg
duck_web_0001.jpg
wolf_web_0001.jpg
mixed_web_0001.jpg
```

收集时注意：

1. 图片必须清晰可见，动物类别明确。
2. 避免大量重复图片或近似重复图片。
3. 每类都要覆盖不同背景、姿态、大小和光照。
4. 不要只收集动物大头照，要包含完整身体和真实场景。
5. 多动物图片非常重要，因为最终任务需要 counting。
6. 保留来源记录，可写入 `datasets/animal_yolo/notes/source_log.md`。

不要把 `val_set/` 当作训练数据。它保留给项目流程检查和计数评分对比，不写入 YOLO `data.yaml`。

## 6. 标注规范

本项目是目标检测任务，必须标注 bounding box。

每张图片对应一个 YOLO 标签文件：

```text
images/train/duck_0001.jpg
labels/train/duck_0001.txt
```

YOLO 标签格式：

```text
class_id x_center y_center width height
```

坐标均为 0 到 1 之间的归一化值。

标注要求：

1. 每只可见动物都单独标注一个框。
2. 同一张图中同类多只动物不能只标一个大框。
3. 框尽量贴合动物主体，不包含过多背景。
4. 遮挡动物如果仍可识别，应标注可见主体区域。
5. 非动物目标不要标注。
6. 不在 20 类列表里的动物不要强行标成相似类别。
7. 类别编号必须和 `data.yaml` 一致。

### 6.1 多类别图片怎么标注

一张图里有多个类别时，仍然按目标检测方式标注：每只动物一个框，每个框一个真实类别。

例如一张图里有：

```text
2 只 duck
1 只 goose
1 只 fox
```

对应标签文件应有 4 行：

```text
9  x_center y_center width height   # duck
9  x_center y_center width height   # duck
10 x_center y_center width height   # goose
13 x_center y_center width height   # fox
```

不要写成：

```text
duck: 2
goose: 1
fox: 1
```

YOLO 训练时学习的是每个目标框的位置和类别，不直接学习数量。数量由推理后的 `src/predictor.py` 按预测框统计出来。

多类别图片标注时注意：

1. 同类多只动物也要分别画框。
2. 不同类别不能合并成一个大框。
3. 框的类别必须是真实类别，不要标成模型容易识别的类别。
4. 遮挡动物如果还能判断类别，就标注其可见部分。
5. 太模糊、无法判断类别的动物，不建议标注。
6. 不在 20 类里的动物不要强行标成相似类别。

例如一张图里有 1 只 goose 和 2 只 duck：

```text
10 x_center y_center width height   # goose
9  x_center y_center width height   # duck
9  x_center y_center width height   # duck
```

例如一张图里有 1 只 fox 和 1 只 dog：

```text
13 x_center y_center width height   # fox
1  x_center y_center width height   # dog
```

例如一张图里有 1 只 tiger 和 1 只 lion：

```text
16 x_center y_center width height   # tiger
17 x_center y_center width height   # lion
```

原始多类别图片可以先放入：

```text
datasets/animal_yolo/raw/mixed_species/
```

真正训练时，再把标注好的图片和同名标签整理到：

```text
datasets/animal_yolo/images/train/
datasets/animal_yolo/labels/train/
```

示例：

```text
images/train/duck_goose_0001.jpg
labels/train/duck_goose_0001.txt
```

标签内容示例：

```text
9  0.312 0.480 0.180 0.220
10 0.661 0.515 0.210 0.260
```

### 6.2 易混淆类别怎么标注

易混淆类别的关键原则是：永远标真实类别，不要按模型预测结果改标签。

例如：

1. 模型把 goose 预测成 sheep，标注时仍然标 `goose`。
2. 模型把 fox 预测成 dog，标注时仍然标 `fox`。
3. 模型把 tiger 预测成 zebra，标注时仍然标 `tiger`。
4. 模型把 pig 预测成 sheep，标注时仍然标 `pig`。

这样模型才会学会纠正错误。

易混淆样本可以先放入：

```text
datasets/animal_yolo/raw/hard_negatives/
```

如果图片中有真实动物，就按真实动物类别标注。如果图片中没有任何 20 类动物，可以作为无目标负样本，并创建同名空标签文件。是否使用纯负样本取决于后续训练实验效果；如果不确定，优先收集“有真实动物但容易混淆”的图片。

可用标注工具：

```text
LabelImg
CVAT
Roboflow
Label Studio
```

导出格式选择 YOLO detection format。

## 7. 训练集和验证集结构

当前项目按每个动物类别约 8:2 划分：

```text
datasets/animal_yolo/images/train + labels/train  训练集
datasets/animal_yolo/images/val   + labels/val    YOLO 验证集
```

整理时注意：

1. 每个类别按图片数量约 80% 放入 train，20% 放入 val。
2. 每个类别都要在 train 中出现。
3. 尽量让每个类别也在 val 中出现。
4. 同一来源的近似重复图片不要大量同时出现在 train 和 val。
5. `val_set/` 保持独立，只用于项目级计数检查，不作为 YOLO 验证集。

如果某类样本很少，优先保证 train 中有足够样本。

## 8. Hard Negative 和误报控制

当前模型常见误报是：

```text
sheep, dog, zebra
```

因此需要收集 hard negative 或混淆样本，例如：

1. 白色 goose/duck 被误报成 sheep。
2. fox/wolf/tiger/lion 被误报成 dog。
3. 条纹背景或其他动物被误报成 zebra。
4. 草地、围栏、笼子、树干等背景干扰物。

如果图片中没有任何 20 类动物，可以作为负样本使用，并创建同名空标签文件。是否使用纯负样本取决于训练工具和实验效果；如果不确定，优先使用“有真实动物但容易混淆”的图片，并正确标注真实类别。

## 9. 质量检查清单

训练前检查：

1. 每张训练图片都有同名 `.txt` 标签。
2. 每个标签文件中的 `class_id` 在 0 到 19 之间。
3. 标签坐标都在 0 到 1 之间。
4. `images/train` 和 `labels/train` 文件名一一对应。
5. `images/val` 和 `labels/val` 文件名一一对应。
6. 类别顺序和 `data.yaml` 完全一致。
7. 没有把 `duck/goose/chicken` 混标。
8. 没有把 `fox/wolf/tiger/lion` 误标成 dog。
9. 没有把不支持类别强行标成支持类别。
10. `val_set` 保持独立，没有混入 YOLO 训练或验证目录。

## 10. 训练和验证建议

第一版建议使用轻量模型快速试错：

```powershell
yolo detect train model=yolo26n.pt data=datasets/animal_yolo/data.yaml epochs=80 imgsz=640 batch=8
```

如果显存不足：

```powershell
yolo detect train model=yolo26n.pt data=datasets/animal_yolo/data.yaml epochs=80 imgsz=640 batch=4
```

如果第一版效果稳定，再尝试更大的模型：

```powershell
yolo detect train model=yolo26s.pt data=datasets/animal_yolo/data.yaml epochs=100 imgsz=640 batch=4
```

训练完成后，把最佳权重复制到：

```text
models/best.pt
```

然后使用已有第四节脚本验证：

```powershell
.\.venv\Scripts\python.exe -m src.predict_batch --input val_set --output outputs\val_predictions_custom.json --model models\best.pt --conf 0.40
.\.venv\Scripts\python.exe -m src.evaluate_val --pred outputs\val_predictions_custom.json --gt val_set\ground_truth.json
```

## 11. 迭代策略

每次训练后都做一次错误分析：

1. 哪些真实类别仍然漏检。
2. 哪些类别数量经常错。
3. 哪些类别经常被误报。
4. 哪些混淆组最严重。
5. 小目标、遮挡、多动物图片是否仍然失败。

下一轮数据补充应围绕错误最多的类别进行，而不是平均给每类加图片。

优先迭代顺序：

```text
1. 补 duck/goose/chicken
2. 补 fox/wolf/lion/tiger
3. 补 pig/rabbit/monkey/deer/goat
4. 补 bear/cat
5. 补 hard negatives 降低 sheep/dog/zebra 误报
```

## 12. 最终目标

最终数据集应满足：

1. 20 类动物全部有训练样本。
2. 缺失或弱识别类别有更多样本。
3. 已有 COCO 类别不会因为 fine-tuning 退化。
4. 多动物和多类别图片足够多。
5. 标注格式能直接用于 YOLO26 训练。
6. 训练后的模型能接入现有 `src.predict_batch` 和 `src.evaluate_val` 流程。
