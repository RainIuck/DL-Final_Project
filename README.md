# Final Project Inference Guide

This project uses YOLO to predict animal counts from the course validation images.

## Environment Setup With uv

Install `uv` if it is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and synchronize the Python environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

After this step, run commands through `uv run` from the project root.

## Run Inference

Use `models/best.pt` with `conf=0.3` on the course validation set:

```bash
uv run python -m src.predict_batch \
  --input "DL课程项目验证集" \
  --output "outputs/DL课程项目验证集_predictions_conf03.json" \
  --model "models/best.pt" \
  --conf 0.3
```

The prediction JSON will be written to:

```text
outputs/DL课程项目验证集_predictions_conf03.json
```

