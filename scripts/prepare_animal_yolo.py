"""Build the unified 20-class YOLO26 animal dataset.

The source datasets under ``datasets/`` use different folder layouts and class
ids. This script keeps those sources untouched and writes a clean merged dataset
to ``datasets/animal_yolo`` with the class order required by the project.
Images are split into train and validation sets by class.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_CLASSES = (
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep",
    "goat",
    "pig",
    "rabbit",
    "chicken",
    "duck",
    "goose",
    "deer",
    "monkey",
    "fox",
    "wolf",
    "bear",
    "tiger",
    "lion",
    "zebra",
    "giraffe",
)

CLASS_ID = {name: index for index, name in enumerate(PROJECT_CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_IMAGE_LIMITS = {
    "deer": 400,
    "giraffe": 300,
}


@dataclass(frozen=True)
class SourceDataset:
    path: str
    target_class: str
    label_map: dict[int, str]


SOURCE_DATASETS = (
    SourceDataset("cat.v1i.yolo26", "cat", {0: "cat"}),
    SourceDataset("Dog.v1i.yolo26", "dog", {0: "dog"}),
    SourceDataset("horse.v1i.yolo26", "horse", {0: "horse"}),
    SourceDataset("Cow Detection.v3i.yolo26", "cow", {0: "cow", 1: "cow"}),
    SourceDataset("Sheep.v2i.yolo26", "sheep", {0: "sheep"}),
    SourceDataset("goat.v1i.yolo26", "goat", {0: "goat"}),
    SourceDataset("Pig_merged.yolo26", "pig", {0: "pig"}),
    SourceDataset("Rabbit_merged.yolo26", "rabbit", {0: "rabbit"}),
    SourceDataset("Chicken.v2i.yolo26", "chicken", {0: "chicken"}),
    SourceDataset("Duck.v1i.yolo26", "duck", {0: "duck"}),
    SourceDataset("goose.v1-goose.yolo26", "goose", {0: "goose"}),
    SourceDataset("deer.v3i.yolo26", "deer", {0: "deer"}),
    SourceDataset("monkey.v5i.yolo26", "monkey", {0: "monkey"}),
    SourceDataset("Fox.v3i.yolo26", "fox", {0: "fox"}),
    SourceDataset("wolf.v1i.yolo26", "wolf", {0: "wolf"}),
    SourceDataset("Bear.v3i.yolo26", "bear", {0: "bear"}),
    SourceDataset("Tiger.v2i.yolo26", "tiger", {0: "tiger"}),
    SourceDataset("Lion.v2-lion.yolo26", "lion", {0: "lion"}),
    SourceDataset("Zebra.v2i.yolo26", "zebra", {0: "zebra"}),
    SourceDataset("Giraffe.v1i.yolo26", "giraffe", {0: "giraffe"}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge 20 animal datasets into one YOLO26 dataset.")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory containing source datasets.")
    parser.add_argument("--output", default="datasets/animal_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--seed", type=int, default=26, help="Seed used for deterministic class image limiting.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for each target class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output)

    if not datasets_dir.is_dir():
        raise NotADirectoryError(f"Datasets directory not found: {datasets_dir}")

    missing = [src.path for src in SOURCE_DATASETS if not (datasets_dir / src.path).is_dir()]
    if missing:
        raise FileNotFoundError("Missing source datasets: " + ", ".join(missing))

    rebuild_output(output_dir)

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_labels: Counter[str] = Counter()
    missing_labels: Counter[str] = Counter()

    for source in SOURCE_DATASETS:
        source_root = datasets_dir / source.path
        pairs = collect_image_label_pairs(source_root)
        if not pairs:
            raise RuntimeError(f"No images found for {source.path}")
        pairs = limit_pairs(source, pairs, args.seed)
        val_paths = select_validation_images(source, pairs, args.seed, args.val_ratio)

        for image_path, label_path, _source_split in pairs:
            split = "val" if image_path in val_paths else "train"
            label_lines, skipped = rewrite_label(label_path, source.label_map)
            if label_path is None:
                missing_labels[source.target_class] += 1
            if skipped:
                skipped_labels[source.target_class] += skipped

            target_name = unique_target_name(source.target_class, image_path, source_root)
            image_out = output_dir / "images" / split / target_name
            label_out = output_dir / "labels" / split / f"{Path(target_name).stem}.txt"

            link_or_copy(image_path, image_out)
            label_out.write_text("".join(label_lines), encoding="utf-8")
            stats[source.target_class][split] += 1

    write_data_yaml(output_dir)
    write_summary(output_dir, stats, skipped_labels, missing_labels)
    print_summary(output_dir, stats, skipped_labels, missing_labels)


def rebuild_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (output_dir / kind / split).mkdir(parents=True, exist_ok=True)


def collect_image_label_pairs(source_root: Path) -> list[tuple[Path, Path | None, str | None]]:
    pairs: list[tuple[Path, Path | None, str | None]] = []
    for image_path in sorted(source_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "annotated_images" in image_path.parts:
            continue

        image_dir_parts = set(image_path.parent.parts)
        label_path = matching_label_path(source_root, image_path)
        source_split = next((split for split in ("train", "valid", "val", "test") if split in image_dir_parts), None)
        pairs.append((image_path, label_path, source_split))
    return pairs


def matching_label_path(source_root: Path, image_path: Path) -> Path | None:
    relative = image_path.relative_to(source_root)
    candidates: list[Path] = []

    parts = list(relative.parts)
    if "images" in parts:
        image_index = parts.index("images")
        label_parts = parts.copy()
        label_parts[image_index] = "labels"
        candidates.append(source_root / Path(*label_parts).with_suffix(".txt"))

    if parts and parts[0] in {"train", "valid", "val", "test"}:
        candidates.append(source_root / parts[0] / "labels" / f"{image_path.stem}.txt")

    candidates.append(source_root / "labels" / f"{image_path.stem}.txt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def limit_pairs(
    source: SourceDataset,
    pairs: list[tuple[Path, Path | None, str | None]],
    seed: int,
) -> list[tuple[Path, Path | None, str | None]]:
    limit = CLASS_IMAGE_LIMITS.get(source.target_class)
    if limit is None or len(pairs) <= limit:
        return pairs

    selected = sorted(
        pairs,
        key=lambda pair: hashlib.sha1(
            f"{seed}:limit:{source.path}:{pair[0].as_posix()}".encode("utf-8")
        ).hexdigest(),
    )[:limit]
    return sorted(selected, key=lambda pair: pair[0].as_posix())


def select_validation_images(
    source: SourceDataset,
    pairs: list[tuple[Path, Path | None, str | None]],
    seed: int,
    val_ratio: float,
) -> set[Path]:
    if not pairs:
        return set()
    n_val = round(len(pairs) * val_ratio)
    if len(pairs) > 1:
        n_val = max(1, min(len(pairs) - 1, n_val))
    else:
        n_val = 0
    selected = sorted(
        pairs,
        key=lambda pair: hashlib.sha1(
            f"{seed}:val:{source.path}:{pair[0].as_posix()}".encode("utf-8")
        ).hexdigest(),
    )[:n_val]
    return {image_path for image_path, _, _ in selected}


def rewrite_label(label_path: Path | None, label_map: dict[int, str]) -> tuple[list[str], int]:
    if label_path is None:
        return [], 0

    output: list[str] = []
    skipped = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            skipped += 1
            continue
        try:
            source_id = int(float(fields[0]))
        except (ValueError, IndexError):
            skipped += 1
            continue

        target_class = label_map.get(source_id)
        if target_class is None:
            skipped += 1
            continue

        fields[0] = str(CLASS_ID[target_class])
        output.append(" ".join(fields[:5]) + "\n")
    return output, skipped


def unique_target_name(target_class: str, image_path: Path, source_root: Path) -> str:
    relative = image_path.relative_to(source_root).with_suffix("")
    safe_relative = "__".join(relative.parts)
    return f"{target_class}__{safe_relative}{image_path.suffix.lower()}"


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_data_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in PROJECT_CLASSES)
    content = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(PROJECT_CLASSES)}\n"
        f"names: [{names}]\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def write_summary(
    output_dir: Path,
    stats: dict[str, Counter[str]],
    skipped_labels: Counter[str],
    missing_labels: Counter[str],
) -> None:
    lines = ["# Unified Animal YOLO Dataset Summary", ""]
    lines.append("The YOLO dataset under `datasets/animal_yolo` is split into train and val.")
    lines.append("Each animal class is split at approximately 8:2 by image count.")
    lines.append("")
    lines.append("| class | train | val | total | skipped_label_rows | missing_label_files |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for class_name in PROJECT_CLASSES:
        train = stats[class_name]["train"]
        val = stats[class_name]["val"]
        total = train + val
        lines.append(
            f"| {class_name} | {train} | {val} | {total} | "
            f"{skipped_labels[class_name]} | {missing_labels[class_name]} |"
        )
    lines.append("")
    lines.append("Class order:")
    for index, class_name in enumerate(PROJECT_CLASSES):
        lines.append(f"{index}: {class_name}")
    lines.append("")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(
    output_dir: Path,
    stats: dict[str, Counter[str]],
    skipped_labels: Counter[str],
    missing_labels: Counter[str],
) -> None:
    total_images = sum(sum(counter.values()) for counter in stats.values())
    print(f"Wrote unified dataset to {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Train images: {sum(counter['train'] for counter in stats.values())}")
    print(f"Val images: {sum(counter['val'] for counter in stats.values())}")
    skipped_total = sum(skipped_labels.values())
    missing_total = sum(missing_labels.values())
    if skipped_total:
        print(f"Skipped label rows: {skipped_total}")
    if missing_total:
        print(f"Images without label files: {missing_total}")


if __name__ == "__main__":
    main()
