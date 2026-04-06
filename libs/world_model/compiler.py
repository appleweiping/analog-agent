"""Compile DesignTask objects into world-model bundles."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    CalibrationState,
    ConstraintReliabilitySummary,
    LossSummary,
    OODStatistics,
    ReplayStatistics,
    SampleCounts,
    ServiceMethodSpec,
    ServingContract,
    ThresholdSpec,
    TrainingState,
    TrustPolicy,
    UsableRegion,
    WorldModelBundle,
    WorldModelCompilationReport,
    WorldModelCompileResponse,
    WorldModelMetadata,
    WorldModelValidationStatus,
)
from libs.utils.hashing import stable_hash
from libs.world_model.design_task_adapter import (
    build_action_schema,
    build_prediction_heads,
    build_state_schema,
    collect_supported_metrics,
    infer_supported_families,
)
from libs.world_model.validation import validate_world_model_bundle


def _serving_contract() -> ServingContract:
    return ServingContract(
        predict_metrics=ServiceMethodSpec(input_schema="WorldState", output_schema="MetricsPrediction"),
        predict_feasibility=ServiceMethodSpec(input_schema="WorldState", output_schema="FeasibilityPrediction"),
        predict_transition=ServiceMethodSpec(input_schema="WorldState+DesignAction", output_schema="TransitionPrediction"),
        rollout=ServiceMethodSpec(input_schema="WorldState+DesignAction[]", output_schema="RolloutResponse"),
        rank_candidates=ServiceMethodSpec(input_schema="WorldState[]", output_schema="CandidateRanking"),
        estimate_simulation_value=ServiceMethodSpec(input_schema="WorldState", output_schema="SimulationValueEstimate"),
        calibrate_with_truth=ServiceMethodSpec(input_schema="WorldState+TruthCalibrationRecord", output_schema="CalibrationUpdateResponse"),
        validate_state=ServiceMethodSpec(input_schema="WorldState", output_schema="WorldModelValidationStatus"),
    )


def _trust_policy(task: DesignTask) -> TrustPolicy:
    cost = task.difficulty_profile.evaluation_cost
    screening = ThresholdSpec(max_ood_score=0.85, max_uncertainty_score=0.85, min_confidence=0.2)
    ranking = ThresholdSpec(max_ood_score=0.6 if cost == "expensive" else 0.7, max_uncertainty_score=0.55, min_confidence=0.45)
    rollout = ThresholdSpec(max_ood_score=0.35, max_uncertainty_score=0.35 if cost == "cheap" else 0.3, min_confidence=0.7)
    return TrustPolicy(
        screening_threshold=screening,
        ranking_threshold=ranking,
        rollout_threshold=rollout,
        must_escalate_conditions=[
            "stability_margin_near_boundary",
            "trust_head_high_risk",
            "constraint_group_conflict",
            "unknown_family",
            "template_mismatch",
        ],
        hard_block_conditions=[
            "invalid_state_schema",
            "unsupported_topology",
            "missing_critical_operating_point_without_safe_proxy",
        ],
    )


def compile_world_model_bundle(task: DesignTask) -> WorldModelCompileResponse:
    """Compile a DesignTask into a formal WorldModelBundle."""

    timestamp = datetime.now(timezone.utc).isoformat()
    supported_families = infer_supported_families(task)
    supported_metrics = collect_supported_metrics(task)
    task_signature = stable_hash(task.model_dump_json())
    bundle = WorldModelBundle(
        world_model_id=f"wm_{task_signature[:12]}",
        parent_task_id=task.task_id,
        supported_circuit_families=supported_families,
        supported_task_types=[task.task_type],
        state_schema=build_state_schema(),
        action_schema=build_action_schema(),
        prediction_heads=build_prediction_heads(supported_metrics),
        calibration_state=CalibrationState(
            calibration_version="1.0",
            last_calibrated_at=None,
            reference_simulator_signature="spice/placeholder",
            per_metric_error_summary=[],
            constraint_reliability_summary=[
                ConstraintReliabilitySummary(
                    constraint_group=group.name,
                    brier_score=0.0,
                    expected_calibration_error=0.0,
                    confidence=0.75,
                    sample_count=0,
                )
                for group in task.constraints.constraint_groups
            ],
            ood_statistics=OODStatistics(query_count=0, high_risk_count=0, high_risk_rate=0.0),
            local_patch_history=[],
            usable_regions=[
                UsableRegion(
                    circuit_family=family,
                    template_id=task.topology.template_id,
                    parameter_ranges={},
                    evaluation_intents=["quick_screening", "task_conditioned_prediction"],
                    readiness="screening",
                )
                for family in supported_families
            ],
        ),
        training_state=TrainingState(
            dataset_signature=f"task_bootstrap::{task.task_id}",
            sample_counts=SampleCounts(static_samples=1, transition_samples=1, failure_samples=1, multi_fidelity_samples=1),
            family_coverage=supported_families,
            fidelity_mix={"quick_screening": 0.4, "partial_simulation": 0.4, "full_ground_truth": 0.2},
            objective_mix=[term.metric for term in task.objective.terms],
            constraint_mix=[group.name for group in task.constraints.constraint_groups],
            loss_summary=LossSummary(metric_loss=0.0, feasibility_loss=0.0, transition_loss=0.0, uncertainty_loss=0.0, total_loss=0.0),
            best_checkpoint_ref=f"heuristic::{task.task_id}",
            update_policy="periodic_incremental",
            replay_statistics=ReplayStatistics(replay_sample_count=0, trajectory_sample_count=0, failure_fraction=0.0),
        ),
        serving_contract=_serving_contract(),
        trust_policy=_trust_policy(task),
        metadata=WorldModelMetadata(
            compile_timestamp=timestamp,
            source_task_signature=task_signature,
            implementation_version="heuristic-world-model-v1",
            assumptions=[
                "world model remains task-conditioned and does not redefine the optimization problem",
                "dynamic predictions are deterministic heuristic proxies until learned backends are attached",
                "trust policy gates rollout and simulator escalation explicitly",
            ],
            provenance=[
                "design_task_adapter",
                "state_builder",
                "feature_projection",
                "validation_engine",
            ],
        ),
        validation_status=WorldModelValidationStatus(
            is_valid=False,
            errors=[],
            warnings=[],
            unresolved_dependencies=[],
            repair_history=[],
            completeness_score=0.0,
        ),
    )
    validation = validate_world_model_bundle(bundle)
    compiled_bundle = bundle.model_copy(update={"validation_status": validation})
    status = "compiled" if validation.is_valid and not validation.warnings else "compiled_with_warnings"
    if validation.errors:
        status = "invalid"
    report = WorldModelCompilationReport(
        status=status,
        derived_fields=[
            "state_schema",
            "action_schema",
            "prediction_heads",
            "calibration_state",
            "training_state",
            "serving_contract",
            "trust_policy",
            "metadata",
            "validation_status",
        ],
        supported_metrics=supported_metrics,
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
        acceptance_summary={
            "supported_family_count": len(supported_families),
            "supported_metric_count": len(supported_metrics),
            "contract_method_count": 8,
            "completeness_score": validation.completeness_score,
        },
    )
    return WorldModelCompileResponse(
        status=status,
        world_model_bundle=None if status == "invalid" else compiled_bundle,
        report=report,
    )
