"""Build a pure-animal YOLO dataset from DOUBAO source images.

This script keeps the original DOUBAO animal images as single-animal training
samples, estimates a subject bounding box for each image, and writes a clean
20-class YOLO dataset split by class at roughly 8:2.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from generate_doubao_synthetic import (
    CLASS_ID,
    IMAGE_EXTENSIONS,
    PROJECT_CLASSES,
    estimate_subject_mask,
    valid_mask,
    write_data_yaml,
)


@dataclass(frozen=True)
class PureRecord:
    class_name: str
    source_path: Path
    output_path: Path
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pure DOUBAO animal YOLO dataset.")
    parser.add_argument("--source", default="datasets/DOUBAO", help="DOUBAO source directory.")
    parser.add_argument("--output", default="datasets/doubao_pure_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--seed", type=int, default=26, help="Seed for deterministic train/val split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per class.")
    parser.add_argument("--preview-count", type=int, default=60, help="Number of labelled preview images.")
    parser.add_argument("--keep-output", action="store_true", help="Do not delete an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    prepare_output(output_dir, args.keep_output)

    records: list[PureRecord] = []
    skipped: list[str] = []
    unreadable: list[str] = []
    source_counts: Counter[str] = Counter()

    for class_name in PROJECT_CLASSES:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            skipped.append(f"{class_name}: missing source directory {class_dir}")
            continue

        image_paths = sorted(iter_image_files(class_dir))
        source_counts[class_name] = len(image_paths)
        split_by_path = split_class_images(image_paths, class_name, args.seed, args.val_ratio)

        for class_index, image_path in enumerate(image_paths, start=1):
            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                unreadable.append(f"{image_path}: {exc}")
                continue

            mask = estimate_subject_mask(image)
            bbox = mask.getbbox()
            if bbox is None:
                skipped.append(f"{image_path}: empty mask")
                continue
            if not valid_mask(mask, bbox):
                skipped.append(f"{image_path}: weak or abnormal mask bbox={bbox}")
                continue

            split = split_by_path[image_path]
            output_name = f"{class_name}__{class_index:04d}{normalized_suffix(image_path)}"
            output_path = output_dir / "images" / split / output_name
            label_path = output_dir / "labels" / split / f"{Path(output_name).stem}.txt"

            save_image(image, output_path)
            label_path.write_text(format_label(class_name, bbox, image.width, image.height), encoding="utf-8")
            records.append(
                PureRecord(
                    class_name=class_name,
                    source_path=image_path,
                    output_path=output_path,
                    bbox=bbox,
                    width=image.width,
                    height=image.height,
                    split=split,
                )
            )

    write_data_yaml(output_dir)
    write_previews(records, output_dir / "preview_labels", args.preview_count, args.seed)
    write_summary(
        output_dir=output_dir,
        source_dir=source_dir,
        records=records,
        source_counts=source_counts,
        unreadable=unreadable,
        skipped=skipped,
        args=args,
    )

    print(f"Wrote pure DOUBAO YOLO dataset to {output_dir}")
    print(f"Usable images: {len(records)}")
    print(f"Train images: {sum(1 for record in records if record.split == 'train')}")
    print(f"Val images: {sum(1 for record in records if record.split == 'val')}")
    if unreadable or skipped:
        print(f"Skipped files: {len(unreadable) + len(skipped)}")


def prepare_output(output_dir: Path, keep_output: bool) -> None:
    if output_dir.exists() and not keep_output:
        shutil.rmtree(output_dir)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (output_dir / kind / split).mkdir(parents=True, exist_ok=True)


def iter_image_files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def split_class_images(
    image_paths: list[Path],
    class_name: str,
    seed: int,
    val_ratio: float,
) -> dict[Path, str]:
    if not image_paths:
        return {}
    n_val = round(len(image_paths) * val_ratio)
    if len(image_paths) > 1 and val_ratio > 0:
        n_val = max(1, min(len(image_paths) - 1, n_val))
    else:
        n_val = 0
    ordered = sorted(
        image_paths,
        key=lambda path: hashlib.sha1(
            f"{seed}:pure-doubao:{class_name}:{path.name}".encode("utf-8")
        ).hexdigest(),
    )
    val_paths = set(ordered[:n_val])
    return {path: "val" if path in val_paths else "train" for path in image_paths}


def normalized_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return ".jpg" if suffix == ".jpeg" else suffix


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def format_label(class_name: str, bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    left, top, right, bottom = bbox
    x_center = ((left + right) / 2) / width
    y_center = ((top + bottom) / 2) / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    return f"{CLASS_ID[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def write_previews(records: list[PureRecord], output_dir: Path, count: int, seed: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0 or not records:
        return

    selected = sorted(
        records,
        key=lambda record: hashlib.sha1(
            f"{seed}:preview:{record.output_path.name}".encode("utf-8")
        ).hexdigest(),
    )[: min(count, len(records))]

    for record in selected:
        with Image.open(record.output_path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(record.bbox, outline=(230, 57, 70), width=max(2, min(image.size) // 180))
        text_bbox = draw.textbbox((0, 0), record.class_name)
        text_width = text_bbox[2] - text_bbox[0] + 8
        text_height = text_bbox[3] - text_bbox[1] + 6
        left, top, _, _ = record.bbox
        label_top = max(0, top - text_height)
        draw.rectangle(
            (left, label_top, min(image.width, left + text_width), label_top + text_height),
            fill=(230, 57, 70),
        )
        draw.text((left + 4, label_top + 3), record.class_name, fill=(255, 255, 255))
        image.save(output_dir / record.output_path.name, quality=92)


def write_summary(
    *,
    output_dir: Path,
    source_dir: Path,
    records: list[PureRecord],
    source_counts: Counter[str],
    unreadable: list[str],
    skipped: list[str],
    args: argparse.Namespace,
) -> None:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split_counts[record.split][record.class_name] += 1

    lines = [
        "# DOUBAO Pure Animal YOLO Dataset",
        "",
        f"- Source: `{source_dir}`",
        f"- Output: `{output_dir}`",
        "- Background directory: ignored",
        "- Annotation style: original image + auto-estimated subject bbox",
        f"- Seed: {args.seed}",
        f"- Validation ratio: {args.val_ratio}",
        f"- Preview images: {min(args.preview_count, len(records))}",
        f"- Total usable images: {len(records)}",
        f"- Total skipped files: {len(unreadable) + len(skipped)}",
        "",
        "## Class Counts",
        "",
        "| class | source images | train | val | usable | skipped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    skipped_by_class = count_skips_by_class(source_dir, unreadable + skipped)
    for class_name in PROJECT_CLASSES:
        train = split_counts["train"][class_name]
        val = split_counts["val"][class_name]
        usable = train + val
        lines.append(
            f"| {class_name} | {source_counts[class_name]} | {train} | {val} | "
            f"{usable} | {skipped_by_class[class_name]} |"
        )

    lines.extend(
        [
            "",
            "## Skipped Files",
            "",
            f"- Unreadable images: {len(unreadable)}",
            f"- Mask-quality skips: {len(skipped)}",
        ]
    )
    for title, entries in (("Unreadable Images", unreadable), ("Mask-Quality Skips", skipped)):
        if not entries:
            continue
        lines.extend(["", f"### {title}", ""])
        lines.extend(f"- {entry}" for entry in entries[:300])
        if len(entries) > 300:
            lines.append(f"- ... and {len(entries) - 300} more")

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_skips_by_class(source_dir: Path, entries: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in entries:
        for class_name in PROJECT_CLASSES:
            marker = f"{source_dir / class_name}/"
            if marker in entry:
                counts[class_name] += 1
                break
    return counts


if __name__ == "__main__":
    main()
