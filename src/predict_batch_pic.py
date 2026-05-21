"""Command-line entrypoint for predicting all images in a folder and saving annotated outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 与原始脚本一致的默认值（可根据你的项目调整）
DEFAULT_MODEL = "yolov8n.pt"      # 模型路径或名称
DEFAULT_CONFIDENCE = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and annotate all images in a folder with YOLO."
    )
    parser.add_argument(
        "--input", required=True, help="Folder containing input images."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for annotated images (default: outputs/<timestamp>).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="YOLO model path or weight name."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold.",
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    images = find_images(input_dir)

    if not images:
        print("No supported images found in the input folder.")
        return

    # 确定输出文件夹
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"detections_{input_dir.name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型（与 AnimalCounter 内部使用相同 YOLO 检测）
    model = YOLO(args.model)

    saved_count = 0
    for image_path in tqdm(images, desc="Detecting & drawing", unit="image"):
        # 推理
        results = model(str(image_path), conf=args.conf, verbose=False)

        # 绘制检测框（包含类别名与置信度）
        annotated = results[0].plot()  # 返回 BGR 格式的 numpy 数组

        # 保存图片（保留原始文件名，可添加后缀以免覆盖）
        out_name = f"{image_path.stem}_annotated{image_path.suffix}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), annotated)
        saved_count += 1

    print(f"Saved {saved_count} annotated images to {output_dir}")


if __name__ == "__main__":
    main()
