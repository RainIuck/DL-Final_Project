"""Project animal classes and conservative YOLO class mapping."""

DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_CONFIDENCE = 0.25

ANIMAL_CLASSES = (
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

ANIMAL_CLASS_SET = set(ANIMAL_CLASSES)

# COCO-pretrained YOLO class names only cover a subset of the project labels.
# Keep this mapping conservative to avoid extra wrong categories in scoring.
YOLO_TO_PROJECT_CLASS = {
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def map_yolo_class(raw_name: str) -> str | None:
    """Return a supported project class for a YOLO class name, or None."""
    return YOLO_TO_PROJECT_CLASS.get(raw_name.strip().lower())

