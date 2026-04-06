"""Robustness verification for the fifth layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import RobustnessCertificate, RobustnessPolicy
from libs.simulation.truth_model import compute_truth_metrics
from libs.utils.hashing import stable_hash


def _passes(task: DesignTask, metrics: dict[str, float]) -> tuple[bool, float]:
    margins: list[float] = []
    for constraint in task.constraints.hard_constraints:
        value = metrics.get(constraint.metric)
        if value is None:
            margins.append(-1.0)
            continue
        if constraint.relation == ">=":
            assert constraint.threshold is not None
            margins.append(value - float(constraint.threshold))
        elif constraint.relation == "<=":
            assert constraint.threshold is not None
            margins.append(float(constraint.threshold) - value)
        elif constraint.relation == "==":
            assert constraint.threshold is not None
            margins.append(constraint.tolerance - abs(value - float(constraint.threshold)))
        else:
            assert constraint.lower_threshold is not None and constraint.upper_threshold is not None
            margins.append(min(value - float(constraint.lower_threshold), float(constraint.upper_threshold) - value))
    worst = min(margins) if margins else 0.0
    return all(margin >= 0.0 for margin in margins), round(float(worst), 6)


def certify_robustness(
    task: DesignTask,
    candidate: CandidateRecord,
    robustness_policy: RobustnessPolicy,
    *,
    fidelity_level: str,
) -> RobustnessCertificate:
    """Evaluate robustness according to fidelity and policy."""

    if fidelity_level not in {"full_robustness_certification", "focused_validation"}:
        return RobustnessCertificate(
            certificate_id=f"rob_{stable_hash(candidate.candidate_id)[:12]}",
            certification_status="nominal_only",
            evaluated_conditions=["nominal"],
            pass_rate=1.0,
            weakest_condition="nominal",
            worst_case_margin=0.0,
            summary=["robustness_not_required_for_this_fidelity"],
        )

    corners = robustness_policy.required_corners or [candidate.world_state_snapshot.environment_state.corner]
    temperatures = robustness_policy.temperature_range_c or [candidate.world_state_snapshot.environment_state.temperature_c]
    loads = robustness_policy.load_conditions or [candidate.world_state_snapshot.environment_state.load_cap_f or 2e-12]
    conditions: list[str] = []
    pass_count = 0
    weakest_condition: str | None = None
    worst_margin: float | None = None

    for corner in corners:
        for temperature in temperatures:
            for load in loads:
                condition = f"{corner}|{temperature}|{load}"
                metrics = compute_truth_metrics(
                    task,
                    candidate,
                    analysis_type="pvt_sweep" if fidelity_level == "full_robustness_certification" else "ac",
                    corner=corner,
                    temperature_c=float(temperature),
                    load_cap_f=float(load),
                )
                passed, margin = _passes(task, metrics)
                conditions.append(condition)
                if passed:
                    pass_count += 1
                if worst_margin is None or margin < worst_margin:
                    worst_margin = margin
                    weakest_condition = condition

    total = len(conditions) or 1
    pass_rate = pass_count / total
    if fidelity_level == "full_robustness_certification":
        status = "robust_certified" if pass_rate == 1.0 else "robustness_failed"
    else:
        status = "partial_robust" if pass_rate >= 0.5 else "robustness_failed"
    return RobustnessCertificate(
        certificate_id=f"rob_{stable_hash(f'{candidate.candidate_id}|{fidelity_level}')[:12]}",
        certification_status=status,
        evaluated_conditions=conditions,
        pass_rate=round(pass_rate, 4),
        weakest_condition=weakest_condition,
        worst_case_margin=worst_margin,
        summary=[
            f"evaluated_condition_count={total}",
            f"pass_rate={round(pass_rate, 4)}",
            f"escalation_policy={','.join(robustness_policy.escalation_policy) or 'none'}",
        ],
    )
