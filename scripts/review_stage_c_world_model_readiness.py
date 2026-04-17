"""Review Stage C world-model strengthening status in one structured report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.paper_evidence import WorldModelUtilitySummary
from libs.schema.world_model import MetricsPrediction, TrainedSurrogateCheckpoint
from libs.schema.world_model_dataset import SurrogateTrainingRun


def _has_fields(model, required: list[str]) -> bool:
    fields = getattr(model, "model_fields", {})
    return all(field in fields for field in required)


def build_status() -> dict[str, object]:
    training_reproducibility_ready = _has_fields(
        SurrogateTrainingRun,
        [
            "split_summary",
            "reproducibility",
            "coverage_summary",
            "confidence_alignment",
            "evaluation_examples",
        ],
    )
    trained_serving_boundary_ready = _has_fields(
        TrainedSurrogateCheckpoint,
        [
            "training_run_id",
            "dataset_signature",
            "training_signature",
            "target_metrics",
            "model_payload",
        ],
    ) and _has_fields(
        MetricsPrediction,
        [
            "surrogate_backend",
            "uncertainty_summary",
        ],
    )
    paper_evidence_extended = _has_fields(
        WorldModelUtilitySummary,
        [
            "prediction_gap_beats_no_world_model",
            "reliability_alignment_improves",
            "ranking_improves_efficiency",
            "calibration_improves_convergence",
            "calibration_updates_observable",
        ],
    )

    script_paths = {
        "dataset_export_script": REPO_ROOT / "scripts" / "build_dataset.py",
        "training_script": REPO_ROOT / "scripts" / "train_world_model.py",
        "world_model_evidence_script": REPO_ROOT / "scripts" / "generate_world_model_evidence.py",
        "vertical_slice_world_model_evidence_script": REPO_ROOT / "scripts" / "generate_vertical_slice_world_model_evidence.py",
    }
    scripts_ready = all(path.exists() for path in script_paths.values())

    stage_status = (
        "stage_c_world_model_complete"
        if all(
            [
                scripts_ready,
                training_reproducibility_ready,
                trained_serving_boundary_ready,
                paper_evidence_extended,
            ]
        )
        else "stage_c_world_model_incomplete"
    )

    return {
        "stage": "Stage C",
        "stage_status": stage_status,
        "training_reproducibility_ready": training_reproducibility_ready,
        "trained_serving_boundary_ready": trained_serving_boundary_ready,
        "paper_evidence_extended": paper_evidence_extended,
        "scripts_ready": scripts_ready,
        "expected_scripts": {key: str(path) for key, path in script_paths.items()},
        "ready_for_stage_d": stage_status == "stage_c_world_model_complete",
        "notes": [
            "Stage C is considered complete when reproducible training artifacts, train/eval outputs, explicit serving-boundary provenance, uncertainty objects, calibration/coverage summaries, and paper-facing evidence objects all exist in mainline code.",
            "This review script summarizes structural readiness and does not claim that every paper-facing figure has already been rerun on every benchmark in the current environment.",
        ],
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
