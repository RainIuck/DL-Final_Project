"""Evaluate prediction JSON with the project scoring rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .classes import ANIMAL_CLASS_SET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score predictions against ground truth JSON.")
    parser.add_argument("--pred", required=True, help="Prediction JSON path.")
    parser.add_argument("--gt", required=True, help="Ground-truth JSON path.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, dict[str, int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


def clean_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}

    clean: dict[str, int] = {}
    for key, value in raw.items():
        if key not in ANIMAL_CLASS_SET:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            continue
        clean[key] = value
    return clean


def score_sample(prediction: dict[str, int], truth: dict[str, int]) -> float:
    if not truth:
        return 100.0 if not prediction else max(0.0, 100.0 - 5.0 * len(prediction))

    n_categories = len(truth)
    category_points = 100.0 / n_categories
    score = 0.0

    for category, true_count in truth.items():
        if category in prediction:
            score += category_points / 2.0
            if prediction[category] == true_count:
                score += category_points / 2.0

    extra_categories = set(prediction) - set(truth)
    score -= 5.0 * len(extra_categories)
    return max(0.0, score)


def main() -> None:
    args = parse_args()
    predictions = load_json(args.pred)
    ground_truth = load_json(args.gt)

    scores: list[float] = []
    for filename in sorted(ground_truth):
        truth = clean_counts(ground_truth[filename])
        prediction = clean_counts(predictions.get(filename, {}))
        score = score_sample(prediction, truth)
        scores.append(score)
        print(f"{filename}: {score:.2f}")

    missing = sorted(set(ground_truth) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truth))
    if missing:
        print(f"Missing predictions: {', '.join(missing)}")
    if extra:
        print(f"Extra prediction files: {', '.join(extra)}")

    average = sum(scores) / len(scores) if scores else 0.0
    print(f"Average: {average:.2f}")


if __name__ == "__main__":
    main()

