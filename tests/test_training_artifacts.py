import json
import tempfile
import unittest
from pathlib import Path

from training_artifacts import to_jsonable, training_summary_payload, write_json_atomically


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class TrainingArtifactsTest(unittest.TestCase):
    def test_to_jsonable_converts_nested_scalar_like_values(self):
        payload = {
            "metrics": {
                "AP": FakeScalar(0.5),
                "nested": [FakeScalar(1.25)],
            }
        }
        converted = to_jsonable(payload)
        self.assertEqual(converted, {"metrics": {"AP": 0.5, "nested": [1.25]}})

    def test_write_json_atomically_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training_summary.json"
            summary = training_summary_payload(
                label_mode="robot-merged",
                model="mobilenet_multi_avg_i384",
                epochs=30,
                batch_size=8,
                learning_rate=0.3,
                validation_loss=[FakeScalar(0.1), FakeScalar(0.2)],
                validation_metrics={"AP": FakeScalar(0.55), "AP50": FakeScalar(0.82)},
                test_loss=[FakeScalar(0.3)],
                test_metrics={"AP": FakeScalar(0.45)},
            )

            write_json_atomically(output_path, summary)

            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["label_mode"], "robot-merged")
            self.assertEqual(loaded["classes"], ["puck", "robot"])
            self.assertEqual(loaded["validation_metrics"]["AP"], 0.55)
            self.assertFalse((Path(temp_dir) / "training_summary.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
