from __future__ import annotations

import json
from pathlib import Path

from label_modes import expected_classes, label_mode_display_name, label_mode_file_suffix


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def training_summary_payload(
    *,
    label_mode: str,
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_loss,
    validation_metrics,
    test_loss,
    test_metrics,
) -> dict:
    return {
        "label_mode": label_mode,
        "label_mode_display_name": label_mode_display_name(label_mode),
        "label_mode_file_suffix": label_mode_file_suffix(label_mode),
        "classes": expected_classes(label_mode),
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_loss": list(validation_loss),
        "validation_metrics": validation_metrics,
        "test_loss": list(test_loss) if test_loss is not None else None,
        "test_metrics": test_metrics,
    }


def write_json_atomically(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)
