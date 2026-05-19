"""Command-line entrypoint for predicting all images in a folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .classes import DEFAULT_CONFIDENCE, DEFAULT_MODEL, IMAGE_EXTENSIONS
from .predictor import AnimalCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict animal counts for an image folder.")
    parser.add_argument("--input", required=True, help="Folder containing input images.")
    parser.add_argument("--output", required=True, help="Path to write prediction JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path or weight name.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold.")
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
    images = find_images(Path(args.input))

    counter = AnimalCounter(model_path=args.model, conf=args.conf)
    predictions: dict[str, dict[str, int]] = {}
    for image in tqdm(images, desc="Predicting", unit="image"):
        predictions[image.name] = counter.predict_image(image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()

