import os

# 项目根目录
base = "datasets/animal_yolo"

# 20 个动物类别（顺序与 data.yaml 中保持一致）
classes = [
    "cat", "dog", "horse", "cow", "sheep", "goat", "pig", "rabbit",
    "chicken", "duck", "goose", "deer", "monkey", "fox", "wolf",
    "bear", "tiger", "lion", "zebra", "giraffe"
]

# 需要创建的所有子目录
subdirs = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val",
    "raw/hard_negatives",
    "raw/mixed_species",
    "notes",
] + [f"raw/by_class/{name}" for name in classes]

# 创建目录
for sub in subdirs:
    os.makedirs(os.path.join(base, sub), exist_ok=True)

# 创建占位 data.yaml（内容后续按需修改）
yaml_path = os.path.join(base, "data.yaml")
if not os.path.exists(yaml_path):
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLO dataset config\n")
        f.write(f"# number of classes: {len(classes)}\n")
        f.write("# class names will be added here\n")

print("Folder structure created successfully.")