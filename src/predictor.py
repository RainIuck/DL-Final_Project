"""YOLO26-based single-image prediction utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .classes import DEFAULT_CONFIDENCE, DEFAULT_MODEL, map_yolo_class


class AnimalCounter:
    """Wrap a YOLO detector and return project-format animal counts."""

    def __init__(self, model_path: str = DEFAULT_MODEL, conf: float = DEFAULT_CONFIDENCE) -> None:
        self.model_path = model_path
        self.conf = conf
        self.model = YOLO(model_path)

    def predict_image(self, image_path: str | Path) -> dict[str, int]:
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(f"Image not found: {image}")

        results = self.model.predict(source=str(image), conf=self.conf, verbose=False)
        if not results:
            return {}

        names = getattr(results[0], "names", None) or getattr(self.model, "names", {})
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return {}

        counts: Counter[str] = Counter()
        for box in boxes:
            raw_name = _class_name(names, int(box.cls.item()))
            project_name = map_yolo_class(raw_name)
            if project_name is None:
                continue
            counts[project_name] += 1

        return dict(sorted(counts.items()))


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)

