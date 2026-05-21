import os
import shutil
from pathlib import Path

# ================== 配置区域 ==================
# 1. 你下载解压后的数据集根目录（里面包含 train, valid, test 文件夹）
SRC_ROOT_DIR = Path(r"C:\Users\25896\Desktop\Animal_Detection_yolo26") 

# 2. 你的目标项目路径
TARGET_DIR = Path("datasets/animal_yolo")


# 3. 【关键】建立映射关系：原数据集的 class_id -> 新的 class_id
# 请根据你下载的数据集的 data.yaml 修改下面的字典！
# 例如：如果原数据集中 'dog' 的 id 是 2，新项目里是 1，则写为 2: 1
# ================== 完美的 20 类动物标签映射表 ==================
# 键(Key) 为你下载的数据集原 ID（基于你给出的 64 类列表索引）
# 值(Value) 为你项目中固定的 20 类动物目标 ID (0-19)
CLASS_MAPPING = {
    0: 15,   # Bear -> bear
    1: 15,   # Brown-bear -> bear (相似物种合并)
    2: 3,    # Bull -> cow (相似物种合并)
    
    6: 0,    # Cat -> cat
    8: 3,    # Cattle -> cow (相似物种合并)
    
    11: 8,   # Chicken -> chicken
    12: 11,  # Deer -> deer
    13: 1,   # Dog -> dog
    14: 9,   # Duck -> duck
    
    17: 13,  # Fox -> fox
    19: 19,  # Giraffe -> giraffe
    20: 5,   # Goat -> goat
    21: 10,  # Goose -> goose
    
    25: 2,   # Horse -> horse
    31: 17,  # Lion -> lion
    35: 12,  # Monkey -> monkey
    
    45: 6,   # Pig -> pig
    46: 15,  # Polar-bear -> bear (相似物种合并)
    47: 7,   # Rabbit -> rabbit
    
    53: 4,   # Sheep -> sheep
    58: 16,  # Tiger -> tiger
    60: 6,   # Wild Boar -> pig (野猪合并到猪，绝佳的补充数据)
    61: 14,  # Wolf -> wolf
    63: 18,  # Zebra -> zebra
}

# 4. 数据集拆分对应关系
# 将原数据集的 train 映射到项目的 train
# 将原数据集的 valid 和 test 统统合并映射到项目的 val
SPLIT_MAPPING = {
    'train': 'train',
    'valid': 'val',
    'test': 'val'      # 如果不想合并 test 集，直接注释掉这一行即可
}

# 是否保留不包含20类动物的图片作为负样本(Hard Negative)
KEEP_HARD_NEGATIVE = True 
# =============================================

# 创建目标目录
for target_split in ['train', 'val']:
    (TARGET_DIR / 'images' / target_split).mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / 'labels' / target_split).mkdir(parents=True, exist_ok=True)

def process_split(src_split_name, target_split_name):
    src_img_dir = SRC_ROOT_DIR / src_split_name / 'images'
    src_txt_dir = SRC_ROOT_DIR / src_split_name / 'labels'
    
    if not src_img_dir.exists():
        print(f"提示: 未找到源目录 {src_img_dir}，跳过。")
        return

    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in img_extensions]
    
    print(f"正在处理原 [{src_split_name}] 数据集 -> 目标 [{target_split_name}] 集，共 {len(image_files)} 张图片...")
    
    copy_count = 0
    for img_path in image_files:
        txt_path = src_txt_dir / f"{img_path.stem}.txt"
        
        new_lines = []
        has_target_class = False
        
        # 如果对应的标签文件存在
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                
                # 如果该类别在我们需要的映射表里
                if old_cls in CLASS_MAPPING:
                    new_cls = CLASS_MAPPING[old_cls]
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts) + "\n")
                    has_target_class = True
        
        # 决定是否复制这张图
        if has_target_class or KEEP_HARD_NEGATIVE:
            # 防止合并 valid 和 test 时发生文件名冲突，在文件名前加上原 split 前缀
            new_file_name = f"{src_split_name}_{img_path.name}"
            new_txt_name = f"{src_split_name}_{img_path.stem}.txt"
            
            dst_img = TARGET_DIR / 'images' / target_split_name / new_file_name
            dst_txt = TARGET_DIR / 'labels' / target_split_name / new_txt_name
            
            # 复制图片
            shutil.copy(img_path, dst_img)
            
            # 写入新标签（如果没有目标类，则写入空文件作为负样本）
            with open(dst_txt, 'w') as f:
                f.writelines(new_lines)
            copy_count += 1

    print(f"完成！实际写入目标 [{target_split_name}] 集共 {copy_count} 张图片。")

# 循环执行处理
for src_split, target_split in SPLIT_MAPPING.items():
    process_split(src_split, target_split)

print("\n数据集结构整理完毕！可以开始按照教程进行质量检查与训练。")