#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prediction_previews import (
    SelectedDetection,
    filter_detections_by_score,
    select_top_detections_per_class,
)

DEFAULT_COLOR_CYCLE = (
    "#D1495B",
    "#2E86AB",
    "#F18F01",
    "#4F772D",
    "#6C5CE7",
    "#C44536",
)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/local/share/fonts/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render test images with the top-confidence detection per class.",
    )
    parser.add_argument(
        "--model",
        default="exported_model/model.tflite",
        help="Path to the exported TFLite detector.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/prepared/test/images",
        help="Directory containing test images to render.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/test-previews",
        help="Directory where annotated preview images will be written.",
    )
    parser.add_argument(
        "--training-summary",
        default="exported_model/training_summary.json",
        help="Optional training summary used to preserve class order in the overlays.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Only detections with score greater than this value are rendered.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum raw detections to request from MediaPipe before per-class selection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of images to render.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=64,
        help="Font size used for detection labels.",
    )
    return parser.parse_args()


def load_class_order(training_summary_path: Path) -> list[str]:
    if not training_summary_path.is_file():
        return []
    with training_summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    classes = payload.get("classes")
    if not isinstance(classes, list):
        return []
    return [str(class_name) for class_name in classes]


def color_by_class(class_order: list[str], detected_classes: set[str]) -> dict[str, str]:
    ordered_classes = list(class_order)
    for class_name in sorted(detected_classes):
        if class_name not in ordered_classes:
            ordered_classes.append(class_name)

    return {
        class_name: DEFAULT_COLOR_CYCLE[index % len(DEFAULT_COLOR_CYCLE)]
        for index, class_name in enumerate(ordered_classes)
    }


def load_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    raise FileNotFoundError(
        "Could not load a TrueType font for preview labels. Rebuild the Docker image after installing fonts."
    )


def clamp_box(detection: SelectedDetection, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left = max(0, min(detection.origin_x, image_width - 1))
    top = max(0, min(detection.origin_y, image_height - 1))
    right = max(left + 1, min(detection.right, image_width))
    bottom = max(top + 1, min(detection.bottom, image_height))
    return left, top, right, bottom


def draw_detection(
    image: Image.Image,
    detection: SelectedDetection,
    color: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = clamp_box(detection, image.width, image.height)
    outline_width = max(6, argsafe_font_size(font) // 6)
    draw.rectangle((left, top, right, bottom), outline=color, width=outline_width)

    label = f"{detection.class_name} {detection.score:.2f}"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_padding_x = max(6, argsafe_font_size(font) // 4)
    text_padding_y = max(4, argsafe_font_size(font) // 6)
    text_left = left
    label_height = text_height + (2 * text_padding_y)
    if top - label_height - 8 >= 0:
        text_top = top - label_height - 8
    else:
        text_top = min(image.height - label_height - 8, top + 8)
    text_right = min(image.width, text_left + text_width + (2 * text_padding_x))
    text_bottom = min(image.height, text_top + label_height)

    draw.rectangle((text_left, text_top, text_right, text_bottom), fill=color)
    draw.text((text_left + text_padding_x, text_top + text_padding_y), label, fill="white", font=font)


def argsafe_font_size(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    return int(getattr(font, "size", 24))


def iter_image_paths(input_dir: Path, limit: int | None) -> list[Path]:
    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def detector_for(model_path: Path, args: argparse.Namespace):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        score_threshold=args.score_threshold,
        max_results=args.max_results,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    return mp, detector


def render_previews(args: argparse.Namespace) -> None:
    model_path = REPO_ROOT / args.model
    input_dir = REPO_ROOT / args.input_dir
    output_dir = REPO_ROOT / args.output_dir
    training_summary_path = REPO_ROOT / args.training_summary

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_order = load_class_order(training_summary_path)
    font = load_font(args.font_size)
    mp, detector = detector_for(model_path, args)

    image_paths = iter_image_paths(input_dir, args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    for image_path in image_paths:
        detection_image = mp.Image.create_from_file(str(image_path))
        result = detector.detect(detection_image)
        top_detections = filter_detections_by_score(
            select_top_detections_per_class(result.detections),
            min_score_exclusive=args.score_threshold,
        )

        rendered = Image.open(image_path).convert("RGB")
        colors = color_by_class(class_order, set(top_detections))
        ordered_class_names = class_order + sorted(class_name for class_name in top_detections if class_name not in class_order)
        for class_name in ordered_class_names:
            detection = top_detections.get(class_name)
            if detection is None:
                continue
            draw_detection(rendered, detection, colors[class_name], font)

        output_path = output_dir / image_path.name
        rendered.save(output_path)
        print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    render_previews(args)


if __name__ == "__main__":
    main()
