"""Netlist realization helpers for the fifth layer."""

from __future__ import annotations

from pathlib import Path
from string import Template

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

TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


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


def _parameter_map(candidate: CandidateRecord) -> dict[str, float]:
    values: dict[str, float] = {}
    for parameter in candidate.world_state_snapshot.parameter_state:
        if isinstance(parameter.value, (int, float)):
            values[parameter.variable_name] = float(parameter.value)
    return values


def _demonstrator_ota2_bindings(task: DesignTask, candidate: CandidateRecord) -> tuple[list[ParameterBinding], ModelBinding, list[StimulusBinding], str]:
    values = _parameter_map(candidate)
    env = candidate.world_state_snapshot.environment_state
    w_in = values.get("w_in", 8e-6)
    l_in = max(values.get("l_in", 1e-6), 1e-9)
    w_tail = values.get("w_tail", 6e-6)
    l_tail = max(values.get("l_tail", 1e-6), 1e-9)
    ibias = max(values.get("ibias", 5e-5), 1e-7)
    cc = max(values.get("cc", 1e-12), 5e-14)
    load_cap_f = float(env.load_cap_f or 2e-12)
    supply_v = float(env.supply_voltage_v or 1.2)
    vin_cm = round(supply_v / 2.0, 6)
    vin_step_high = round(min(supply_v * 0.7, vin_cm + 0.08), 6)

    width_ratio_in = max(w_in / l_in, 0.5)
    width_ratio_tail = max(w_tail / l_tail, 0.5)
    gm1 = 0.08 * (ibias * width_ratio_in) ** 0.5
    gm2 = 0.085 * (ibias * width_ratio_tail) ** 0.5
    ro1 = max(2.8e6 * (l_in / max(w_in, 1e-9)) * (5e-5 / ibias) ** 0.5, 8e4)
    ro2 = max(4.2e6 * (l_tail / max(w_tail, 1e-9)) * (5e-5 / ibias) ** 0.45, 1.5e5)
    cp1 = max(0.12 * cc, 5e-14)
    effective_secondary_cap = max(0.05 * cc + 0.2 * load_cap_f, 1e-15)
    p2_hint_hz = gm2 / (2.0 * 3.141592653589793 * effective_secondary_cap)

    bindings = [
        ParameterBinding(variable_name="w_in", netlist_target="param::w_in", value_si=w_in, units="m", source="user_override"),
        ParameterBinding(variable_name="l_in", netlist_target="param::l_in", value_si=l_in, units="m", source="user_override"),
        ParameterBinding(variable_name="w_tail", netlist_target="param::w_tail", value_si=w_tail, units="m", source="user_override"),
        ParameterBinding(variable_name="l_tail", netlist_target="param::l_tail", value_si=l_tail, units="m", source="user_override"),
        ParameterBinding(variable_name="ibias", netlist_target="param::ibias", value_si=ibias, units="A", source="user_override"),
        ParameterBinding(variable_name="cc", netlist_target="param::cc", value_si=cc, units="F", source="user_override"),
        ParameterBinding(variable_name="cload", netlist_target="param::cload", value_si=load_cap_f, units="F", source="system_inferred"),
        ParameterBinding(variable_name="gm1", netlist_target="param::gm1", value_si=gm1, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="gm2", netlist_target="param::gm2", value_si=gm2, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="ro1", netlist_target="param::ro1", value_si=ro1, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="ro2", netlist_target="param::ro2", value_si=ro2, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="cp1", netlist_target="param::cp1", value_si=cp1, units="F", source="system_inferred"),
        ParameterBinding(variable_name="p2_hint_hz", netlist_target="hint::p2_hz", value_si=p2_hint_hz, units="Hz", source="system_inferred"),
        ParameterBinding(variable_name="vin_step_high", netlist_target="param::vin_step_high", value_si=vin_step_high, units="V", source="system_inferred"),
    ]
    model_binding = ModelBinding(
        process_node=task.parent_spec_id,
        corner=env.corner,
        temperature_c=env.temperature_c,
        supply_voltage_v=supply_v,
        backend_model_ref="builtin_demo_ota2_small_signal_v1",
    )
    stimulus = [
        StimulusBinding(source_name="VDD", stimulus_type="supply", parameters={"value": supply_v}),
        StimulusBinding(source_name="VINP", stimulus_type="ac_input", parameters={"dc_value": vin_cm, "ac_amplitude": 1.0}),
        StimulusBinding(source_name="VINN", stimulus_type="bias", parameters={"value": vin_cm}),
    ]
    template = Template((TEMPLATE_ROOT / "ota2_nominal_op_ac.spice.tpl").read_text(encoding="utf-8"))
    rendered = template.safe_substitute(
        truth_mode="demonstrator_truth",
        template_id=task.topology.template_id or "ota2_nominal_op_ac",
        p2_hint_hz=f"{p2_hint_hz:.6e}",
        vdd=f"{supply_v:.6e}",
        vin_cm=f"{vin_cm:.6e}",
        vin_step_high=f"{vin_step_high:.6e}",
        ibias=f"{ibias:.6e}",
        cc=f"{cc:.6e}",
        cload=f"{load_cap_f:.6e}",
        gm1=f"{gm1:.6e}",
        gm2=f"{gm2:.6e}",
        ro1=f"{ro1:.6e}",
        ro2=f"{ro2:.6e}",
        cp1=f"{cp1:.6e}",
    )
    return bindings, model_binding, stimulus, rendered


def realize_netlist_instance(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    backend: str,
    analyses: list[AnalysisStatement],
) -> NetlistInstance:
    """Realize a formal NetlistInstance for one candidate."""

    world_state = candidate.world_state_snapshot
    if backend == "ngspice" and task.circuit_family == "two_stage_ota" and task.topology.topology_mode == "fixed":
        bindings, model_binding, stimulus, rendered_netlist = _demonstrator_ota2_bindings(task, candidate)
    else:
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
        rendered_netlist = "\n".join(rendered_lines)
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
        rendered_netlist=rendered_netlist,
    )
