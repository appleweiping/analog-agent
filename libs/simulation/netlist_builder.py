"""Netlist realization helpers for the fifth layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import (
    AnalysisStatement,
    IntegrityCheckResult,
    MeasurementHook,
    ModelBinding,
    NetlistInstance,
    ParameterBinding,
    SavePolicy,
    StimulusBinding,
    TemplateBinding,
)
from libs.utils.hashing import stable_hash


def _integrity_checks(task: DesignTask, candidate: CandidateRecord) -> list[IntegrityCheckResult]:
    checks = []
    bound_names = {parameter.variable_name for parameter in candidate.world_state_snapshot.parameter_state}
    for variable in task.design_space.variables:
        checks.append(
            IntegrityCheckResult(
                check_name=f"binding::{variable.name}",
                passed=variable.name in bound_names,
                detail="bound" if variable.name in bound_names else "missing",
            )
        )
    checks.append(
        IntegrityCheckResult(
            check_name="topology_binding",
            passed=task.topology.template_id is not None or task.topology.topology_mode != "fixed",
            detail=task.topology.template_id or task.topology.topology_mode,
        )
    )
    return checks


def realize_netlist_instance(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    backend: str,
    analyses: list[AnalysisStatement],
) -> NetlistInstance:
    """Realize a formal NetlistInstance for one candidate."""

    world_state = candidate.world_state_snapshot
    bindings = []
    for variable in task.design_space.variables:
        value = next((item.value for item in world_state.parameter_state if item.variable_name == variable.name), variable.default)
        bindings.append(
            ParameterBinding(
                variable_name=variable.name,
                netlist_target=f"param::{variable.name}",
                value_si=value,
                units=variable.units,
                source=variable.source,
            )
        )
    model_binding = ModelBinding(
        process_node=task.parent_spec_id,
        corner=world_state.environment_state.corner,
        temperature_c=world_state.environment_state.temperature_c,
        supply_voltage_v=world_state.environment_state.supply_voltage_v,
        backend_model_ref=f"{backend}::{task.circuit_family}::{world_state.environment_state.corner}",
    )
    stimulus = [
        StimulusBinding(source_name="vdd", stimulus_type="supply", parameters={"value": world_state.environment_state.supply_voltage_v or 1.2}),
        StimulusBinding(source_name="input_common_mode", stimulus_type="bias", parameters={"value": 0.6}),
    ]
    rendered_lines = [
        f"* realized candidate {candidate.candidate_id}",
        f"* backend {backend}",
        f"* family {task.circuit_family}",
        f".param {' '.join(f'{binding.variable_name}={binding.value_si}' for binding in bindings)}",
    ]
    for analysis in analyses:
        rendered_lines.append(f"* analysis {analysis.analysis_type} order={analysis.order}")
    rendered_lines.append(".end")
    hooks = [MeasurementHook(metric=metric.metric, from_analysis=metric.from_analysis, method=metric.method) for metric in task.evaluation_plan.metric_extractors]
    checks = _integrity_checks(task, candidate)
    render_status = "ready" if all(check.passed for check in checks) else "invalid"
    signature = stable_hash(f"{task.task_id}|{candidate.candidate_id}|{backend}|{world_state.environment_state.corner}")
    return NetlistInstance(
        netlist_id=f"net_{signature[:12]}",
        template_binding=TemplateBinding(
            template_id=task.topology.template_id,
            template_version=task.topology.template_version,
            topology_mode=task.topology.topology_mode,
            circuit_family=task.circuit_family,
        ),
        parameter_binding=bindings,
        model_binding=model_binding,
        stimulus_binding=stimulus,
        analysis_statements=analyses,
        save_policy=SavePolicy(
            save_node_voltages=[port.name for port in task.topology.ports],
            save_branch_currents=[instance.name for instance in task.topology.instances_schema[:3]],
            save_waveforms=True,
            save_operating_point=True,
        ),
        measurement_hooks=hooks,
        integrity_checks=checks,
        render_status=render_status,
        rendered_netlist="\n".join(rendered_lines),
    )
