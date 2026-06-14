"""Generate a synthetic YOLO dataset from DOUBAO animal and background images.

The DOUBAO animal assets are RGB images with mostly light backgrounds rather
than transparent cutouts. This script estimates a subject mask from border
colors, composites animals onto generated backgrounds, and writes YOLO labels
from the visible mask extents.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


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


@dataclass(frozen=True)
class Asset:
    class_name: str
    path: Path
    image: Image.Image
    mask: Image.Image
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Placement:
    class_name: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Candidate:
    class_name: str
    image: Image.Image
    alpha: Image.Image
    x: int
    y: int
    bbox: tuple[int, int, int, int]
    full_mask: Image.Image
    mask_area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DOUBAO synthetic YOLO dataset.")
    parser.add_argument("--source", default="datasets/DOUBAO", help="Directory with animal class folders and background.")
    parser.add_argument("--output", default="datasets/doubao_synthetic_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--count", type=int, default=2000, help="Number of synthetic training images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Output image size.")
    parser.add_argument("--seed", type=int, default=26, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--min-animals", type=int, default=1, help="Minimum animals per image.")
    parser.add_argument("--max-animals", type=int, default=6, help="Maximum animals per image.")
    parser.add_argument("--preview-output", default="outputs/doubao_synthetic_preview", help="Directory for labelled previews.")
    parser.add_argument("--preview-count", type=int, default=20, help="Number of labelled preview images to write.")
    parser.add_argument("--sample-vis-count", type=int, default=50, help="Number of labelled full-dataset samples to write.")
    parser.add_argument("--no-preview", action="store_true", help="Skip preview visualizations.")
    parser.add_argument("--keep-output", action="store_true", help="Do not delete an existing output directory before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.min_animals <= 0 or args.max_animals < args.min_animals:
        raise ValueError("Animal count range is invalid")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    rng = random.Random(args.seed)
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    preview_dir = Path(args.preview_output)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    backgrounds, background_errors = load_backgrounds(source_dir / "background")
    assets_by_class, asset_errors, mask_skips = load_assets(source_dir)
    available_classes = [name for name in PROJECT_CLASSES if assets_by_class[name]]
    if not backgrounds:
        raise RuntimeError("No readable backgrounds found.")
    if not available_classes:
        raise RuntimeError("No usable animal assets found.")

    prepare_output(output_dir, args.keep_output)
    image_dir = output_dir / "images" / "all"
    label_dir = output_dir / "labels" / "all"

    image_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    generated_records: list[tuple[Path, list[Placement]]] = []
    failed_images = 0

    attempts = 0
    max_attempts = args.count * 5
    while len(generated_records) < args.count and attempts < max_attempts:
        attempts += 1
        image, placements = compose_image(
            rng=rng,
            backgrounds=backgrounds,
            assets_by_class=assets_by_class,
            available_classes=available_classes,
            imgsz=args.imgsz,
            min_animals=args.min_animals,
            max_animals=args.max_animals,
        )
        if not placements:
            failed_images += 1
            continue

        stem = f"doubao_synth_{len(generated_records) + 1:06d}"
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        image.convert("RGB").save(image_path, quality=92)
        label_path.write_text(format_yolo_labels(placements, args.imgsz), encoding="utf-8")
        generated_records.append((image_path, placements))

        seen_classes = {placement.class_name for placement in placements}
        for class_name in seen_classes:
            image_counts[class_name] += 1
        for placement in placements:
            instance_counts[placement.class_name] += 1

    if len(generated_records) < args.count:
        raise RuntimeError(
            f"Only generated {len(generated_records)} images after {attempts} attempts; "
            "relax overlap/visibility constraints or add more assets."
        )

    if not args.no_preview:
        write_visualizations(generated_records[: args.preview_count], preview_dir, args.imgsz)

    split_records = split_dataset(output_dir, generated_records, args.seed, args.val_ratio)
    write_data_yaml(output_dir)

    if not args.no_preview:
        sample_records = deterministic_sample(split_records["train"] + split_records["val"], args.sample_vis_count, args.seed)
        write_visualizations(sample_records, output_dir / "preview_labels", args.imgsz)

    write_summary(
        output_dir=output_dir,
        source_dir=source_dir,
        count=args.count,
        generated_records=generated_records,
        backgrounds=backgrounds,
        assets_by_class=assets_by_class,
        background_errors=background_errors,
        asset_errors=asset_errors,
        mask_skips=mask_skips,
        image_counts=image_counts,
        instance_counts=instance_counts,
        failed_images=failed_images,
        args=args,
    )

    print(f"Generated {len(generated_records)} images in {output_dir}")
    print(f"Preview labels: {preview_dir if not args.no_preview else 'skipped'}")
    if failed_images:
        print(f"Skipped {failed_images} images with no valid placements")


def load_backgrounds(background_dir: Path) -> tuple[list[Image.Image], list[str]]:
    backgrounds: list[Image.Image] = []
    errors: list[str] = []
    for path in sorted(iter_image_files(background_dir)):
        try:
            with Image.open(path) as image:
                backgrounds.append(ImageOps.exif_transpose(image).convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"{path}: {exc}")
    return backgrounds, errors


def load_assets(source_dir: Path) -> tuple[dict[str, list[Asset]], list[str], list[str]]:
    assets_by_class: dict[str, list[Asset]] = {name: [] for name in PROJECT_CLASSES}
    errors: list[str] = []
    mask_skips: list[str] = []

    for class_name in PROJECT_CLASSES:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            errors.append(f"{class_name}: missing directory {class_dir}")
            continue
        for path in sorted(iter_image_files(class_dir)):
            try:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{path}: {exc}")
                continue

            mask = estimate_subject_mask(image)
            bbox = mask.getbbox()
            if bbox is None:
                mask_skips.append(f"{path}: empty mask")
                continue
            if not valid_mask(mask, bbox):
                mask_skips.append(f"{path}: weak or abnormal mask bbox={bbox}")
                continue

            cropped_mask = refine_alpha_mask(mask.crop(bbox))
            cropped_image = remove_light_edge_fringe(image.crop(bbox), cropped_mask)
            assets_by_class[class_name].append(Asset(class_name, path, cropped_image, cropped_mask, bbox))

    return assets_by_class, errors, mask_skips


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def estimate_subject_mask(image: Image.Image) -> Image.Image:
    """Estimate non-background pixels using border color flood fill."""
    work = image.convert("RGB").resize((192, 192), Image.Resampling.LANCZOS)
    width, height = work.size
    pixels = work.load()
    corners = [
        pixels[0, 0],
        pixels[width - 1, 0],
        pixels[0, height - 1],
        pixels[width - 1, height - 1],
        pixels[width // 2, 0],
        pixels[width // 2, height - 1],
        pixels[0, height // 2],
        pixels[width - 1, height // 2],
    ]
    bg = tuple(sorted(channel)[len(channel) // 2] for channel in zip(*corners))

    background = Image.new("RGB", work.size, bg)
    diff = ImageChops.difference(work, background).convert("L")
    flood = flood_background_mask(work, bg, tolerance=42)
    candidate = diff.point(lambda value: 255 if value > 22 else 0)
    mask = ImageChops.multiply(candidate, ImageOps.invert(flood))

    mask = mask.filter(ImageFilter.MaxFilter(9))
    mask = mask.filter(ImageFilter.MinFilter(5))
    mask = fill_bbox_region(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    mask = mask.point(lambda value: 255 if value > 18 else 0)
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    return mask


def flood_background_mask(image: Image.Image, bg: tuple[int, int, int], tolerance: int) -> Image.Image:
    width, height = image.size
    pixels = image.load()
    mask = Image.new("L", image.size, 0)
    mask_pixels = mask.load()
    stack = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= width or y >= height or mask_pixels[x, y]:
            continue
        if color_distance(pixels[x, y], bg) > tolerance:
            continue
        mask_pixels[x, y] = 255
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    return mask


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def fill_bbox_region(mask: Image.Image) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return mask
    width, height = mask.size
    left, top, right, bottom = bbox
    pad_x = max(2, int((right - left) * 0.06))
    pad_y = max(2, int((bottom - top) * 0.06))
    expanded = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    output = Image.new("L", mask.size, 0)
    output.paste(mask.crop(expanded), expanded)
    return output


def valid_mask(mask: Image.Image, bbox: tuple[int, int, int, int]) -> bool:
    width, height = mask.size
    left, top, right, bottom = bbox
    bbox_width = right - left
    bbox_height = bottom - top
    if bbox_width < width * 0.08 or bbox_height < height * 0.08:
        return False
    if bbox_width > width * 0.98 and bbox_height > height * 0.98:
        return False
    alpha_pixels = mask.point(lambda value: 1 if value > 30 else 0)
    area = sum(alpha_pixels.getdata())
    if area < width * height * 0.008:
        return False
    fill_ratio = area / max(1, bbox_width * bbox_height)
    return fill_ratio >= 0.035


def refine_alpha_mask(mask: Image.Image) -> Image.Image:
    """Shrink hard light-background residue, then feather the visible edge."""
    solid = mask.point(lambda value: 255 if value > 80 else 0)
    solid = solid.filter(ImageFilter.MinFilter(3))
    feather = solid.filter(ImageFilter.GaussianBlur(1.4))
    return feather.point(lambda value: 0 if value < 10 else value)


def remove_light_edge_fringe(image: Image.Image, alpha: Image.Image) -> Image.Image:
    """Replace semi-transparent light edge pixels with nearby subject color."""
    rgb = image.convert("RGB")
    subject_color = median_subject_color(rgb, alpha)
    pixels = rgb.load()
    alpha_pixels = alpha.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            a = alpha_pixels[x, y]
            if a == 0:
                continue
            r, g, b = pixels[x, y]
            whiteness = min(1.0, max(0.0, (r + g + b - 570) / 195))
            edge = 1.0 - min(1.0, a / 220)
            mix = max(0.0, min(0.75, whiteness * 0.65 + edge * 0.45))
            if mix <= 0:
                continue
            sr, sg, sb = subject_color
            pixels[x, y] = (
                int(r * (1 - mix) + sr * mix),
                int(g * (1 - mix) + sg * mix),
                int(b * (1 - mix) + sb * mix),
            )
    output = rgb.convert("RGBA")
    output.putalpha(alpha)
    return output


def median_subject_color(image: Image.Image, alpha: Image.Image) -> tuple[int, int, int]:
    pixels = image.load()
    alpha_pixels = alpha.load()
    width, height = image.size
    samples: list[tuple[int, int, int]] = []
    step = max(1, min(width, height) // 80)
    for y in range(0, height, step):
        for x in range(0, width, step):
            if alpha_pixels[x, y] < 230:
                continue
            r, g, b = pixels[x, y]
            if r + g + b > 690:
                continue
            samples.append((r, g, b))
    if not samples:
        return (96, 96, 88)
    channels = []
    for index in range(3):
        values = sorted(sample[index] for sample in samples)
        channels.append(values[len(values) // 2])
    return tuple(channels)  # type: ignore[return-value]


def prepare_output(output_dir: Path, keep_output: bool) -> None:
    if output_dir.exists() and not keep_output:
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "all").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "all").mkdir(parents=True, exist_ok=True)


def split_dataset(
    output_dir: Path,
    records: list[tuple[Path, list[Placement]]],
    seed: int,
    val_ratio: float,
) -> dict[str, list[tuple[Path, list[Placement]]]]:
    rng = random.Random(seed + 202)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    if len(shuffled) > 1 and val_ratio > 0:
        n_val = max(1, min(len(shuffled) - 1, n_val))
    else:
        n_val = 0
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


def compose_image(
    *,
    rng: random.Random,
    backgrounds: list[Image.Image],
    assets_by_class: dict[str, list[Asset]],
    available_classes: list[str],
    imgsz: int,
    min_animals: int,
    max_animals: int,
) -> tuple[Image.Image, list[Placement]]:
    background = prepare_background(rng.choice(backgrounds), rng, imgsz).convert("RGBA")
    target_count = weighted_animal_count(rng, min_animals, max_animals)
    classes = choose_classes(rng, available_classes, target_count)
    candidates: list[Candidate] = []

    for class_name in classes:
        asset = rng.choice(assets_by_class[class_name])
        transformed, alpha = transform_asset(asset, rng, imgsz, target_count)
        if transformed.width < 12 or transformed.height < 12:
            continue

        candidate = find_candidate_position(rng, class_name, transformed, alpha, imgsz, candidates)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return background.convert("RGB"), []

    placements = final_visible_placements(candidates, imgsz)
    if not placements:
        return background.convert("RGB"), []

    for candidate in candidates:
        paste_with_shadow(background, candidate.image, candidate.alpha, candidate.x, candidate.y)

    return background.convert("RGB"), placements


def weighted_animal_count(rng: random.Random, min_animals: int, max_animals: int) -> int:
    values = list(range(min_animals, max_animals + 1))
    weights = []
    for value in values:
        if value == 1:
            weights.append(0.8)
        elif value == 2:
            weights.append(1.2)
        elif value <= 4:
            weights.append(1.8)
        else:
            weights.append(1.4)
    return rng.choices(values, weights=weights, k=1)[0]


def choose_classes(rng: random.Random, available_classes: list[str], count: int) -> list[str]:
    if count <= len(available_classes):
        if count == 1:
            return [rng.choice(available_classes)]
        unique_count = max(2, min(count, len(available_classes)))
        return rng.sample(available_classes, unique_count)

    classes = rng.sample(available_classes, len(available_classes))
    while len(classes) < count:
        classes.append(rng.choice(available_classes))
    rng.shuffle(classes)
    return classes[:count]


def find_candidate_position(
    rng: random.Random,
    class_name: str,
    image: Image.Image,
    alpha: Image.Image,
    imgsz: int,
    existing: list[Candidate],
) -> Candidate | None:
    alpha_area = mask_area(alpha)
    if alpha_area <= 0:
        return None

    best: Candidate | None = None
    best_score = float("inf")
    for _ in range(60):
        x = rng.randint(-image.width // 12, max(-image.width // 12, imgsz - image.width + image.width // 12))
        y = rng.randint(-image.height // 12, max(-image.height // 12, imgsz - image.height + image.height // 12))
        full_mask = Image.new("L", (imgsz, imgsz), 0)
        full_mask.paste(alpha, (x, y), alpha)
        bbox = full_mask.getbbox()
        if bbox is None or not valid_visible_bbox(bbox, imgsz):
            continue

        clipped_area = mask_area(full_mask)
        if clipped_area / alpha_area < 0.82:
            continue

        overlap_ratio = max_overlap_ratio(full_mask, clipped_area, existing)
        max_iou = max_bbox_iou(bbox, existing)
        if overlap_ratio <= 0.18 and max_iou <= 0.14:
            return Candidate(class_name, image, alpha, x, y, bbox, full_mask, clipped_area)

        score = overlap_ratio + max_iou
        if score < best_score and overlap_ratio <= 0.28 and max_iou <= 0.24:
            best_score = score
            best = Candidate(class_name, image, alpha, x, y, bbox, full_mask, clipped_area)

    return best


def final_visible_placements(candidates: list[Candidate], imgsz: int) -> list[Placement]:
    placements: list[Placement] = []
    for index, candidate in enumerate(candidates):
        visible = candidate.full_mask.copy()
        for occluder in candidates[index + 1 :]:
            visible = subtract_mask(visible, occluder.full_mask)
        bbox = visible.getbbox()
        if bbox is None:
            continue
        visible_area = mask_area(visible)
        visible_ratio = visible_area / max(1, candidate.mask_area)
        if visible_ratio < 0.55:
            continue
        if not valid_visible_bbox(bbox, imgsz):
            continue
        placements.append(Placement(candidate.class_name, bbox))
    return placements


def max_overlap_ratio(mask: Image.Image, area: int, existing: list[Candidate]) -> float:
    if not existing or area <= 0:
        return 0.0
    worst = 0.0
    for candidate in existing:
        intersection = ImageChops.multiply(mask, candidate.full_mask)
        overlap = mask_area(intersection)
        worst = max(worst, overlap / max(1, min(area, candidate.mask_area)))
    return worst


def max_bbox_iou(bbox: tuple[int, int, int, int], existing: list[Candidate]) -> float:
    if not existing:
        return 0.0
    return max(bbox_iou(bbox, candidate.bbox) for candidate in existing)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def subtract_mask(mask: Image.Image, occluder: Image.Image) -> Image.Image:
    blocker = occluder.point(lambda value: 255 if value > 24 else 0)
    return ImageChops.multiply(mask, ImageOps.invert(blocker))


def mask_area(mask: Image.Image) -> int:
    return sum(mask.point(lambda value: 1 if value > 24 else 0).getdata())


def prepare_background(background: Image.Image, rng: random.Random, imgsz: int) -> Image.Image:
    width, height = background.size
    crop_ratio = rng.uniform(0.62, 1.0)
    side = int(min(width, height) * crop_ratio)
    left = rng.randint(0, max(0, width - side))
    top = rng.randint(0, max(0, height - side))
    cropped = background.crop((left, top, left + side, top + side)).resize((imgsz, imgsz), Image.Resampling.LANCZOS)
    cropped = adjust_image(cropped, rng, brightness=(0.86, 1.12), contrast=(0.9, 1.1), color=(0.9, 1.1))
    return cropped


def transform_asset(asset: Asset, rng: random.Random, imgsz: int, animal_count: int) -> tuple[Image.Image, Image.Image]:
    image = asset.image.copy()
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    image = adjust_image(image, rng, brightness=(0.82, 1.18), contrast=(0.86, 1.18), color=(0.85, 1.18))
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)
        alpha = alpha.crop(bbox)

    if animal_count <= 2:
        scale = rng.uniform(0.23, 0.48)
    elif animal_count <= 4:
        scale = rng.uniform(0.17, 0.36)
    else:
        scale = rng.uniform(0.12, 0.27)
    if asset.class_name in {"chicken", "duck", "goose", "rabbit", "cat"}:
        scale *= rng.uniform(0.72, 0.95)
    if asset.class_name in {"giraffe", "horse", "cow", "bear", "lion", "tiger"}:
        scale *= rng.uniform(0.92, 1.08)
    target_width = max(16, int(imgsz * scale))
    ratio = target_width / max(1, image.width)
    target_height = max(16, int(image.height * ratio))
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    angle = rng.uniform(-8, 8)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    alpha = image.getchannel("A")
    alpha = alpha.filter(ImageFilter.MinFilter(3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(rng.uniform(0.65, 1.25)))
    alpha = alpha.point(lambda value: 0 if value < 18 else value)
    image.putalpha(alpha)
    return image, alpha


def adjust_image(
    image: Image.Image,
    rng: random.Random,
    *,
    brightness: tuple[float, float],
    contrast: tuple[float, float],
    color: tuple[float, float],
) -> Image.Image:
    mode = image.mode
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(*brightness))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(*contrast))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(*color))
    if alpha is not None:
        rgba = rgb.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba
    return rgb.convert(mode) if mode != "RGB" else rgb


def visible_alpha(alpha: Image.Image, x: int, y: int, imgsz: int, placed_masks: list[Image.Image]) -> Image.Image:
    full = Image.new("L", (imgsz, imgsz), 0)
    full.paste(alpha, (x, y), alpha)
    if not placed_masks:
        return full
    occlusion = Image.new("L", (imgsz, imgsz), 0)
    for mask in placed_masks:
        occlusion = ImageChops.lighter(occlusion, mask)
    visible = ImageChops.multiply(full, ImageOps.invert(occlusion.point(lambda value: 180 if value > 24 else 0)))
    return visible


def valid_visible_bbox(bbox: tuple[int, int, int, int], imgsz: int) -> bool:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width < imgsz * 0.035 or height < imgsz * 0.035:
        return False
    if width > imgsz * 0.95 and height > imgsz * 0.95:
        return False
    return True


def paste_with_shadow(canvas: Image.Image, image: Image.Image, alpha: Image.Image, x: int, y: int) -> None:
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(8)).point(lambda value: int(value * 0.35))
    shadow.putalpha(shadow_alpha)
    canvas.alpha_composite(shadow, (x + 8, y + 10))
    blended = blend_edge_with_canvas(canvas, image, alpha, x, y)
    canvas.alpha_composite(blended, (x, y))


def blend_edge_with_canvas(canvas: Image.Image, image: Image.Image, alpha: Image.Image, x: int, y: int) -> Image.Image:
    output = image.copy()
    out_pixels = output.load()
    canvas_pixels = canvas.load()
    alpha_pixels = alpha.load()
    width, height = output.size
    canvas_width, canvas_height = canvas.size
    for py in range(height):
        cy = y + py
        if cy < 0 or cy >= canvas_height:
            continue
        for px in range(width):
            cx = x + px
            if cx < 0 or cx >= canvas_width:
                continue
            a = alpha_pixels[px, py]
            if a <= 0 or a >= 210:
                continue
            edge_mix = min(0.35, (210 - a) / 210 * 0.35)
            r, g, b, old_a = out_pixels[px, py]
            br, bg, bb, _ = canvas_pixels[cx, cy]
            out_pixels[px, py] = (
                int(r * (1 - edge_mix) + br * edge_mix),
                int(g * (1 - edge_mix) + bg * edge_mix),
                int(b * (1 - edge_mix) + bb * edge_mix),
                old_a,
            )
    return output


def format_yolo_labels(placements: list[Placement], imgsz: int) -> str:
    lines = []
    for placement in placements:
        left, top, right, bottom = placement.bbox
        x_center = ((left + right) / 2) / imgsz
        y_center = ((top + bottom) / 2) / imgsz
        width = (right - left) / imgsz
        height = (bottom - top) / imgsz
        lines.append(f"{CLASS_ID[placement.class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return "".join(lines)


def write_data_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in PROJECT_CLASSES)
    dataset_path = output_dir.as_posix()
    content = (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(PROJECT_CLASSES)}\n"
        f"names: [{names}]\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def deterministic_sample(records: list[tuple[Path, list[Placement]]], count: int, seed: int) -> list[tuple[Path, list[Placement]]]:
    if count <= 0 or not records:
        return []
    rng = random.Random(seed + 999)
    if len(records) <= count:
        return records
    indices = sorted(rng.sample(range(len(records)), count))
    return [records[index] for index in indices]


def write_visualizations(records: list[tuple[Path, list[Placement]]], output_dir: Path, imgsz: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (0, 119, 182),
    ]
    for image_path, placements in records:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, placement in enumerate(placements):
            color = colors[index % len(colors)]
            draw.rectangle(placement.bbox, outline=color, width=3)
            label = placement.class_name
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0] + 8
            text_height = text_bbox[3] - text_bbox[1] + 6
            left, top, _, _ = placement.bbox
            label_top = max(0, top - text_height)
            draw.rectangle((left, label_top, min(imgsz, left + text_width), label_top + text_height), fill=color)
            draw.text((left + 4, label_top + 3), label, fill=(255, 255, 255))
        image.save(output_dir / image_path.name, quality=92)


def write_summary(
    *,
    output_dir: Path,
    source_dir: Path,
    count: int,
    generated_records: list[tuple[Path, list[Placement]]],
    backgrounds: list[Image.Image],
    assets_by_class: dict[str, list[Asset]],
    background_errors: list[str],
    asset_errors: list[str],
    mask_skips: list[str],
    image_counts: Counter[str],
    instance_counts: Counter[str],
    failed_images: int,
    args: argparse.Namespace,
) -> None:
    total_instances = sum(instance_counts.values())
    lines = [
        "# DOUBAO Synthetic YOLO Dataset",
        "",
        f"- Source: `{source_dir}`",
        f"- Output: `{output_dir}`",
        f"- Requested images: {count}",
        f"- Generated images: {len(generated_records)}",
        f"- Generated instances: {total_instances}",
        f"- Image size: {args.imgsz}x{args.imgsz}",
        f"- Seed: {args.seed}",
        f"- Animals per image: {args.min_animals}-{args.max_animals}",
        f"- Backgrounds loaded: {len(backgrounds)}",
        f"- Failed/no-placement images: {failed_images}",
        "",
        "## Class Counts",
        "",
        "| class | usable assets | images | instances |",
        "|---|---:|---:|---:|",
    ]
    for class_name in PROJECT_CLASSES:
        lines.append(
            f"| {class_name} | {len(assets_by_class[class_name])} | "
            f"{image_counts[class_name]} | {instance_counts[class_name]} |"
        )

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
        if not entries:
            continue
        lines.extend(["", f"### {title}", ""])
        for entry in entries[:200]:
            lines.append(f"- {entry}")
        if len(entries) > 200:
            lines.append(f"- ... and {len(entries) - 200} more")

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
