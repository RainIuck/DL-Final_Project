"""Command-line entrypoint for predicting one image."""

from __future__ import annotations

import argparse
import json

from .classes import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from .predictor import AnimalCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict animal counts for one image.")
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path or weight name.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counter = AnimalCounter(model_path=args.model, conf=args.conf)
    prediction = counter.predict_image(args.image)
    print(json.dumps(prediction, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

