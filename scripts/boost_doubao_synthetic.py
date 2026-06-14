"""Append focused synthetic samples to the DOUBAO YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from generate_doubao_synthetic import (
    CLASS_ID,
    PROJECT_CLASSES,
    IMAGE_EXTENSIONS,
    Placement,
    compose_image,
    deterministic_sample,
    format_yolo_labels,
    load_assets,
    load_backgrounds,
    write_data_yaml,
    write_visualizations,
)


FOCUS_CLASSES = ("cow", "goat", "chicken", "duck", "zebra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append focused synthetic samples to DOUBAO YOLO dataset.")
    parser.add_argument("--source", default="datasets/DOUBAO")
    parser.add_argument("--output", default="datasets/doubao_synthetic_yolo")
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=126)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--focus-classes", default=",".join(FOCUS_CLASSES))
    parser.add_argument("--preview-output", default="outputs/doubao_synthetic_boost_preview")
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument("--sample-vis-count", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    focus_classes = tuple(class_name.strip() for class_name in args.focus_classes.split(",") if class_name.strip())
    unknown = sorted(set(focus_classes) - set(PROJECT_CLASSES))
    if unknown:
        raise ValueError(f"Unknown focus classes: {', '.join(unknown)}")

    backgrounds, background_errors = load_backgrounds(source_dir / "background")
    assets_by_class, asset_errors, mask_skips = load_assets(source_dir)
    for class_name in focus_classes:
        if not assets_by_class[class_name]:
            raise RuntimeError(f"No usable assets for focus class: {class_name}")

    existing_records = collect_existing_as_all(output_dir)
    start_index = max_existing_index(existing_records) + 1
    rng = random.Random(args.seed)
    all_image_dir = output_dir / "images" / "all"
    all_label_dir = output_dir / "labels" / "all"

    generated_records: list[tuple[Path, list[Placement]]] = []
    failed_images = 0
    attempts = 0
    max_attempts = args.count * 8
    while len(generated_records) < args.count and attempts < max_attempts:
        attempts += 1
        image, placements = compose_image(
            rng=rng,
            backgrounds=backgrounds,
            assets_by_class=assets_by_class,
            available_classes=list(focus_classes),
            imgsz=args.imgsz,
            min_animals=3,
            max_animals=6,
        )
        placements = ensure_focus_count_scene(placements, focus_classes)
        if not placements:
            failed_images += 1
            continue

        stem = f"doubao_boost_{start_index + len(generated_records):06d}"
        image_path = all_image_dir / f"{stem}.jpg"
        label_path = all_label_dir / f"{stem}.txt"
        image.convert("RGB").save(image_path, quality=92)
        label_path.write_text(format_yolo_labels(placements, args.imgsz), encoding="utf-8")
        generated_records.append((image_path, placements))

    if len(generated_records) < args.count:
        raise RuntimeError(f"Only generated {len(generated_records)} focused samples after {attempts} attempts.")

    write_visualizations(generated_records[: args.preview_count], Path(args.preview_output), args.imgsz)
    all_records = existing_records + generated_records
    split_records = split_existing_all(output_dir, all_records, args.seed, args.val_ratio)
    write_data_yaml(output_dir)
    write_visualizations(
        deterministic_sample(split_records["train"] + split_records["val"], args.sample_vis_count, args.seed),
        output_dir / "preview_labels",
        args.imgsz,
    )
    write_boost_summary(
        output_dir=output_dir,
        source_dir=source_dir,
        focus_classes=focus_classes,
        existing_count=len(existing_records),
        generated_records=generated_records,
        split_records=split_records,
        backgrounds=len(backgrounds),
        background_errors=background_errors,
        asset_errors=asset_errors,
        mask_skips=mask_skips,
        failed_images=failed_images,
        args=args,
    )
    print(f"Appended {len(generated_records)} focused samples to {output_dir}")
    print(f"Total images after split: {len(split_records['train']) + len(split_records['val'])}")
    if failed_images:
        print(f"Skipped {failed_images} failed focused attempts")


def collect_existing_as_all(output_dir: Path) -> list[tuple[Path, list[Placement]]]:
    all_image_dir = output_dir / "images" / "all"
    all_label_dir = output_dir / "labels" / "all"
    if all_image_dir.exists():
        shutil.rmtree(all_image_dir)
    if all_label_dir.exists():
        shutil.rmtree(all_label_dir)
    all_image_dir.mkdir(parents=True, exist_ok=True)
    all_label_dir.mkdir(parents=True, exist_ok=True)

    records: list[tuple[Path, list[Placement]]] = []
    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            target_image = all_image_dir / image_path.name
            target_label = all_label_dir / label_path.name
            shutil.move(str(image_path), target_image)
            shutil.move(str(label_path), target_label)
            records.append((target_image, parse_label_file(target_label)))

    for split in ("train", "val"):
        for cache_path in (output_dir / "labels" / split).glob("*.cache"):
            cache_path.unlink()
        shutil.rmtree(output_dir / "images" / split, ignore_errors=True)
        shutil.rmtree(output_dir / "labels" / split, ignore_errors=True)
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    return records


def parse_label_file(label_path: Path) -> list[Placement]:
    placements: list[Placement] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        left = int(round((x_center - width / 2) * 640))
        top = int(round((y_center - height / 2) * 640))
        right = int(round((x_center + width / 2) * 640))
        bottom = int(round((y_center + height / 2) * 640))
        placements.append(Placement(PROJECT_CLASSES[class_id], (left, top, right, bottom)))
    return placements


def ensure_focus_count_scene(placements: list[Placement], focus_classes: tuple[str, ...]) -> list[Placement]:
    if len(placements) < 2:
        return []
    counts = Counter(placement.class_name for placement in placements)
    if not any(count >= 2 for class_name, count in counts.items() if class_name in focus_classes):
        return []
    return placements


def max_existing_index(records: list[tuple[Path, list[Placement]]]) -> int:
    best = 0
    for path, _ in records:
        digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
        if digits:
            best = max(best, int(digits[-1]))
    return best


def split_existing_all(
    output_dir: Path,
    records: list[tuple[Path, list[Placement]]],
    seed: int,
    val_ratio: float,
) -> dict[str, list[tuple[Path, list[Placement]]]]:
    rng = random.Random(seed + 202)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    n_val = max(1, min(len(shuffled) - 1, n_val)) if len(shuffled) > 1 and val_ratio > 0 else 0
    val_paths = {path for path, _ in shuffled[:n_val]}

    split_records = {"train": [], "val": []}
    for image_path, placements in records:
        split = "val" if image_path in val_paths else "train"
        target_image = output_dir / "images" / split / image_path.name
        source_label = output_dir / "labels" / "all" / f"{image_path.stem}.txt"
        target_label = output_dir / "labels" / split / source_label.name
        shutil.move(str(image_path), target_image)
        shutil.move(str(source_label), target_label)
        split_records[split].append((target_image, placements))

    shutil.rmtree(output_dir / "images" / "all")
    shutil.rmtree(output_dir / "labels" / "all")
    return split_records


def write_boost_summary(
    *,
    output_dir: Path,
    source_dir: Path,
    focus_classes: tuple[str, ...],
    existing_count: int,
    generated_records: list[tuple[Path, list[Placement]]],
    split_records: dict[str, list[tuple[Path, list[Placement]]]],
    backgrounds: int,
    background_errors: list[str],
    asset_errors: list[str],
    mask_skips: list[str],
    failed_images: int,
    args: argparse.Namespace,
) -> None:
    split_counters = {split: count_instances(records) for split, records in split_records.items()}
    total_instances = sum(sum(counter.values()) for counter in split_counters.values())
    lines = [
        "# DOUBAO Synthetic YOLO Dataset",
        "",
        f"- Source: `{source_dir}`",
        f"- Output: `{output_dir}`",
        f"- Existing images before boost: {existing_count}",
        f"- Focus boost images added: {len(generated_records)}",
        f"- Total images: {len(split_records['train']) + len(split_records['val'])}",
        f"- Split: train {len(split_records['train'])} / val {len(split_records['val'])} (8:2)",
        f"- Total instances: {total_instances}",
        f"- Focus classes: {', '.join(focus_classes)}",
        f"- Image size: {args.imgsz}x{args.imgsz}",
        f"- Seed: {args.seed}",
        f"- Backgrounds loaded: {backgrounds}",
        f"- Failed/no-placement focused attempts: {failed_images}",
        "",
        "## Split Counts",
        "",
        "| split | images | labels | instances |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        label_count = len(list((output_dir / "labels" / split).glob("*.txt")))
        lines.append(f"| {split} | {len(split_records[split])} | {label_count} | {sum(split_counters[split].values())} |")

    lines.extend(["", "## Class Counts", "", "| class | train instances | val instances | total instances |", "|---|---:|---:|---:|"])
    for class_name in PROJECT_CLASSES:
        train_count = split_counters["train"][class_name]
        val_count = split_counters["val"][class_name]
        lines.append(f"| {class_name} | {train_count} | {val_count} | {train_count + val_count} |")

    lines.extend(
        [
            "",
            "## Skipped Files",
            "",
            f"- Unreadable backgrounds: {len(background_errors)}",
            f"- Unreadable animal images: {len(asset_errors)}",
            f"- Mask-quality skips: {len(mask_skips)}",
        ]
    )
    for title, entries in (
        ("Unreadable Backgrounds", background_errors),
        ("Unreadable Animal Images", asset_errors),
        ("Mask-Quality Skips", mask_skips),
    ):
        if entries:
            lines.extend(["", f"### {title}", ""])
            lines.extend(f"- {entry}" for entry in entries[:200])
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_instances(records: list[tuple[Path, list[Placement]]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for _, placements in records:
        for placement in placements:
            counter[placement.class_name] += 1
    return counter


if __name__ == "__main__":
    main()
