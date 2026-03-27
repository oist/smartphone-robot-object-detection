import argparse
import json
from pathlib import Path

from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization

from label_modes import (
    DEFAULT_LABEL_MODE,
    LABEL_MODE_CHOICES,
    expected_classes,
)
from training_artifacts import training_summary_payload, write_json_atomically

SUPPORTED_MODELS = {
    "mobilenet_v2": object_detector.SupportedModels.MOBILENET_V2,
    "mobilenet_v2_i320": object_detector.SupportedModels.MOBILENET_V2_I320,
    "mobilenet_multi_avg": object_detector.SupportedModels.MOBILENET_MULTI_AVG,
    "mobilenet_multi_avg_i384": object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the smartphone robot detector with MediaPipe Model Maker.",
    )
    parser.add_argument(
        "--train-data",
        default="data/prepared/train",
        help="Path to the COCO training split directory.",
    )
    parser.add_argument(
        "--validation-data",
        default="data/prepared/validation",
        help="Path to the COCO validation split directory.",
    )
    parser.add_argument(
        "--test-data",
        default="data/prepared/test",
        help="Optional path to the COCO test split directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="exported_model",
        help="Directory for exported model artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".mediapipe_cache",
        help="Directory used by MediaPipe for dataset caching.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="mobilenet_multi_avg_i384",
        help="MediaPipe supported model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--cosine-decay-epochs", type=int, default=None)
    parser.add_argument("--cosine-decay-alpha", type=float, default=1.0)
    parser.add_argument("--l2-weight-decay", type=float, default=3e-5)
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODE_CHOICES,
        default=DEFAULT_LABEL_MODE,
        help="Expected dataset label mode. 'robot-merged' expects categories ['puck', 'robot'].",
    )
    parser.add_argument(
        "--export-fp16",
        action="store_true",
        help="Also export a float16 quantized TFLite model for GPU-oriented use.",
    )
    parser.add_argument(
        "--run-qat",
        action="store_true",
        help="Run quantization-aware training and export an int8 TFLite model.",
    )
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--qat-batch-size", type=int, default=8)
    parser.add_argument("--qat-learning-rate", type=float, default=0.3)
    parser.add_argument("--qat-decay-steps", type=int, default=8)
    parser.add_argument("--qat-decay-rate", type=float, default=0.96)
    return parser.parse_args()


def load_categories(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = payload.get("categories", [])
    return [category["name"] for category in sorted(categories, key=lambda item: item["id"])]


def load_dataset(split_dir: Path, cache_dir: Path, label_mode: str) -> object_detector.Dataset:
    validate_coco_split(split_dir, label_mode)
    return object_detector.Dataset.from_coco_folder(str(split_dir), cache_dir=str(cache_dir))


def maybe_load_dataset(
    split_dir: Path | None, cache_dir: Path, label_mode: str
) -> object_detector.Dataset | None:
    if split_dir is None or not split_dir.exists():
        return None
    return load_dataset(split_dir, cache_dir, label_mode)


def validate_coco_split(split_dir: Path, label_mode: str) -> None:
    if not split_dir.exists():
        raise FileNotFoundError(f"COCO split not found: {split_dir}")

    labels_path = split_dir / "labels.json"
    images_dir = split_dir / "images"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels.json in {split_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory in {split_dir}")

    categories = load_categories(labels_path)
    expected_category_names = expected_classes(label_mode)
    if categories != expected_category_names:
        raise ValueError(
            f"{labels_path} categories {categories} do not match expected {expected_category_names}"
        )


def write_training_summary(
    output_dir: Path,
    args: argparse.Namespace,
    validation_loss: list[float] | tuple[float, ...],
    validation_metrics: dict,
    test_loss: list[float] | tuple[float, ...] | None,
    test_metrics: dict | None,
) -> None:
    summary = training_summary_payload(
        label_mode=args.label_mode,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_loss=validation_loss,
        validation_metrics=validation_metrics,
        test_loss=test_loss,
        test_metrics=test_metrics,
    )
    write_json_atomically(output_dir / "training_summary.json", summary)


def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_data)
    validation_dir = Path(args.validation_data)
    test_dir = Path(args.test_data)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset(train_dir, cache_dir, args.label_mode)
    validation_data = load_dataset(validation_dir, cache_dir, args.label_mode)
    test_data = maybe_load_dataset(test_dir, cache_dir, args.label_mode)

    options = object_detector.ObjectDetectorOptions(
        supported_model=SUPPORTED_MODELS[args.model],
        hparams=object_detector.HParams(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            cosine_decay_epochs=args.cosine_decay_epochs,
            cosine_decay_alpha=args.cosine_decay_alpha,
            export_dir=str(output_dir),
        ),
        model_options=object_detector.ModelOptions(
            l2_weight_decay=args.l2_weight_decay,
        ),
    )

    print("Loading dataset completed. Starting training...")
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    validation_loss, validation_metrics = model.evaluate(validation_data, batch_size=args.batch_size)
    print(f"Validation loss: {validation_loss}")
    print(f"Validation metrics: {validation_metrics}")

    if test_data is not None:
        test_loss, test_metrics = model.evaluate(test_data, batch_size=args.batch_size)
        print(f"Test loss: {test_loss}")
        print(f"Test metrics: {test_metrics}")
    else:
        test_loss = None
        test_metrics = None

    model.export_model(model_name="model.tflite")
    print(f"Exported float model to {output_dir / 'model.tflite'}")

    if args.export_fp16:
        fp16_config = quantization.QuantizationConfig.for_float16()
        model.export_model(
            model_name="model_fp16.tflite",
            quantization_config=fp16_config,
        )
        print(f"Exported float16 model to {output_dir / 'model_fp16.tflite'}")

    try:
        write_training_summary(
            output_dir=output_dir,
            args=args,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
            test_loss=test_loss,
            test_metrics=test_metrics,
        )
        print(f"Wrote training summary to {output_dir / 'training_summary.json'}")
    except Exception as exc:
        print(f"Warning: failed to write training summary: {exc}")

    if args.run_qat:
        qat_hparams = object_detector.QATHParams(
            learning_rate=args.qat_learning_rate,
            batch_size=args.qat_batch_size,
            epochs=args.qat_epochs,
            decay_steps=args.qat_decay_steps,
            decay_rate=args.qat_decay_rate,
        )
        model.quantization_aware_training(
            train_data=train_data,
            validation_data=validation_data,
            qat_hparams=qat_hparams,
        )
        qat_loss, qat_metrics = model.evaluate(validation_data, batch_size=args.qat_batch_size)
        print(f"QAT validation loss: {qat_loss}")
        print(f"QAT validation metrics: {qat_metrics}")
        model.export_model(model_name="model_int8_qat.tflite")
        print(f"Exported int8 QAT model to {output_dir / 'model_int8_qat.tflite'}")


if __name__ == "__main__":
    main()
