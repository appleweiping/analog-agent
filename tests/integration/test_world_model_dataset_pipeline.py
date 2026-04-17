from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WorldModelDatasetPipelineTests(unittest.TestCase):
    def test_build_dataset_and_train_world_model_scripts(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "world_model_dataset.json"
            training_path = tmp / "world_model_training.json"

            build = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_dataset.py",
                    "--tasks",
                    "ota2",
                    "--modes",
                    "full_system",
                    "--steps",
                    "1",
                    "--repeat-runs",
                    "1",
                    "--max-simulations",
                    "2",
                    "--max-candidates-per-step",
                    "2",
                    "--max-records-per-family",
                    "4",
                    "--output",
                    str(dataset_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(dataset_path.exists(), msg=build.stdout + build.stderr)
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            self.assertGreater(len(payload["records"]), 0)
            self.assertIn("two_stage_ota", payload["family_coverage"])

            train = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_world_model.py",
                    "--dataset",
                    str(dataset_path),
                    "--config",
                    "configs/world_model/tabular_surrogate.yaml",
                    "--output",
                    str(training_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(training_path.exists(), msg=train.stdout + train.stderr)
            training_payload = json.loads(training_path.read_text(encoding="utf-8"))
            self.assertEqual(training_payload["config"]["model_family"], "tabular_knn")
            self.assertGreater(training_payload["training_record_count"], 0)


if __name__ == "__main__":
    unittest.main()
