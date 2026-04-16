"""Netlist realization helpers for the fifth layer."""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import (
    AnalysisStatement,
    IntegrityCheckResult,
    MeasurementHook,
    ModelBinding,
    ModelSource,
    ModelValidityLevel,
    NetlistInstance,
    ParameterBinding,
    SavePolicy,
    StimulusBinding,
    TemplateBinding,
)
from libs.utils.hashing import stable_hash
from libs.vertical_slices.bandgap_spec import bandgap_v1_netlist_template_path
from libs.vertical_slices.folded_cascode_spec import folded_cascode_v1_netlist_template_path
from libs.vertical_slices.ldo_spec import ldo_v1_netlist_template_path
from libs.vertical_slices.ota2_spec import ota2_v1_netlist_template_path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "simulator"


def _load_backend_model_config(backend: str) -> dict[str, str | int | bool]:
    config_path = CONFIG_ROOT / f"{backend}.yaml"
    config: dict[str, str | int | bool] = {}
    if not config_path.exists():
        return config
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        lowered = value.lower()
        if lowered in {"true", "false"}:
            config[key.strip()] = lowered == "true"
        elif value.isdigit():
            config[key.strip()] = int(value)
        else:
            config[key.strip()] = value
    return config


def _split_csv_names(value: str | int | bool | None) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_env_value(names: list[str]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _model_binding_from_config(
    *,
    backend: str,
    process_node: str,
    corner: str,
    temperature_c: float,
    supply_voltage_v: float | None,
    default_builtin_ref: str,
    overrides: dict[str, float | int | str | bool] | None = None,
) -> ModelBinding:
    config = _load_backend_model_config(backend)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value not in {None, ""}})
    configured_truth_mode = str(config.get("configured_truth_mode", "disabled")).strip() or "disabled"
    configured_truth_contract = str(config.get("configured_truth_contract", "unconfigured")).strip() or "unconfigured"
    configured_truth_source = str(config.get("configured_truth_model_source", "external_model_card_or_pdk_root")).strip() or "external_model_card_or_pdk_root"
    model_type = str(config.get("model_type", "builtin")).strip() or "builtin"
    external_path = _first_env_value(_split_csv_names(config.get("external_model_card_env_vars"))) or str(config.get("external_model_card_path", "")).strip()
    pdk_root = _first_env_value(_split_csv_names(config.get("pdk_root_env_vars"))) or str(config.get("pdk_root", "")).strip()
    builtin_ref = str(config.get("default_builtin_model", default_builtin_ref)).strip() or default_builtin_ref
    external_requested = model_type == "external" or configured_truth_mode != "disabled"
    if external_requested and external_path:
        source = ModelSource(source_type="path", locator=external_path)
        validity = ModelValidityLevel(
            truth_level="configured_truth",
            detail="external_model_card_configured",
            industrial_confidence=0.9,
        )
        binding_confidence = 0.95
        backend_model_ref = external_path
    elif external_requested and configured_truth_source in {"external_model_card_or_pdk_root", "external_pdk_root"} and pdk_root:
        source = ModelSource(source_type="path", locator=pdk_root)
        validity = ModelValidityLevel(
            truth_level="configured_truth",
            detail=f"external_pdk_root_candidate:{configured_truth_contract}",
            industrial_confidence=0.55,
        )
        binding_confidence = 0.58
        backend_model_ref = f"{pdk_root}::pdk_root"
    elif external_requested:
        missing_source = "missing_configured_truth_source"
        if configured_truth_source == "external_pdk_root" or (
            configured_truth_source == "external_model_card_or_pdk_root" and not external_path and not pdk_root
        ):
            missing_source = "missing_external_model_card_or_pdk_root"
        elif not external_path:
            missing_source = "missing_external_model_card"
        source = ModelSource(source_type="path", locator=missing_source)
        validity = ModelValidityLevel(
            truth_level="configured_truth",
            detail=f"configured_truth_requested_but_source_missing:{configured_truth_contract}",
            industrial_confidence=0.25,
        )
        binding_confidence = 0.2
        backend_model_ref = missing_source
    else:
        source = ModelSource(source_type="registry", locator=builtin_ref, registry_key=builtin_ref)
        validity = ModelValidityLevel(
            truth_level="demonstrator_truth",
            detail="builtin_demonstrator_model",
            industrial_confidence=0.35,
        )
        binding_confidence = 0.62
        backend_model_ref = builtin_ref
    return ModelBinding(
        model_type="external" if external_requested else "builtin",
        model_source=source,
        process_node=process_node,
        corner=corner,
        temperature_c=temperature_c,
        supply_voltage_v=supply_voltage_v,
        backend_model_ref=backend_model_ref,
        binding_confidence=binding_confidence,
        validity_level=validity,
    )


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
    model_binding = _model_binding_from_config(
        backend="ngspice",
        process_node=task.parent_spec_id,
        corner=env.corner,
        temperature_c=env.temperature_c,
        supply_voltage_v=supply_v,
        default_builtin_ref="builtin_demo_ota2_small_signal_v1",
    )
    stimulus = [
        StimulusBinding(source_name="VDD", stimulus_type="supply", parameters={"value": supply_v}),
        StimulusBinding(source_name="VINP", stimulus_type="ac_input", parameters={"dc_value": vin_cm, "ac_amplitude": 1.0}),
        StimulusBinding(source_name="VINN", stimulus_type="bias", parameters={"value": vin_cm}),
    ]
    template = Template(ota2_v1_netlist_template_path().read_text(encoding="utf-8"))
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


def _demonstrator_folded_cascode_bindings(task: DesignTask, candidate: CandidateRecord) -> tuple[list[ParameterBinding], ModelBinding, list[StimulusBinding], str]:
    values = _parameter_map(candidate)
    env = candidate.world_state_snapshot.environment_state
    w_in = values.get("w_in", 10e-6)
    l_in = max(values.get("l_in", 1.5e-6), 1e-9)
    w_cas = values.get("w_cas", 8e-6)
    l_cas = max(values.get("l_cas", 1.5e-6), 1e-9)
    ibias = max(values.get("ibias", 8e-5), 1e-7)
    cc = max(values.get("cc", 0.5e-12), 5e-14)
    load_cap_f = float(env.load_cap_f or 2.5e-12)
    supply_v = float(env.supply_voltage_v or 1.2)
    vin_cm = round(supply_v / 2.0, 6)
    vin_step_high = round(min(supply_v * 0.7, vin_cm + 0.075), 6)

    width_ratio_in = max(w_in / l_in, 0.5)
    width_ratio_cas = max(w_cas / l_cas, 0.5)
    gm_in = 0.09 * (ibias * width_ratio_in) ** 0.5
    gm_fold = 0.11 * (ibias * width_ratio_cas) ** 0.5
    ro_in = max(3.6e6 * (l_in / max(w_in, 1e-9)) * (8e-5 / ibias) ** 0.45, 1.0e5)
    ro_fold = max(6.4e6 * (l_cas / max(w_cas, 1e-9)) * (8e-5 / ibias) ** 0.42, 2.0e5)
    c_fold = max(0.1 * cc, 2e-14)
    effective_secondary_cap = max(0.06 * cc + 0.18 * load_cap_f, 1e-15)
    p2_hint_hz = gm_fold / (2.0 * 3.141592653589793 * effective_secondary_cap)

    bindings = [
        ParameterBinding(variable_name="w_in", netlist_target="param::w_in", value_si=w_in, units="m", source="user_override"),
        ParameterBinding(variable_name="l_in", netlist_target="param::l_in", value_si=l_in, units="m", source="user_override"),
        ParameterBinding(variable_name="w_cas", netlist_target="param::w_cas", value_si=w_cas, units="m", source="user_override"),
        ParameterBinding(variable_name="l_cas", netlist_target="param::l_cas", value_si=l_cas, units="m", source="user_override"),
        ParameterBinding(variable_name="ibias", netlist_target="param::ibias", value_si=ibias, units="A", source="user_override"),
        ParameterBinding(variable_name="cc", netlist_target="param::cc", value_si=cc, units="F", source="user_override"),
        ParameterBinding(variable_name="cload", netlist_target="param::cload", value_si=load_cap_f, units="F", source="system_inferred"),
        ParameterBinding(variable_name="gm_in", netlist_target="param::gm_in", value_si=gm_in, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="gm_fold", netlist_target="param::gm_fold", value_si=gm_fold, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="ro_in", netlist_target="param::ro_in", value_si=ro_in, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="ro_fold", netlist_target="param::ro_fold", value_si=ro_fold, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="c_fold", netlist_target="param::c_fold", value_si=c_fold, units="F", source="system_inferred"),
        ParameterBinding(variable_name="p2_hint_hz", netlist_target="hint::p2_hz", value_si=p2_hint_hz, units="Hz", source="system_inferred"),
        ParameterBinding(variable_name="vin_step_high", netlist_target="param::vin_step_high", value_si=vin_step_high, units="V", source="system_inferred"),
    ]
    model_binding = _model_binding_from_config(
        backend="ngspice",
        process_node=task.parent_spec_id,
        corner=env.corner,
        temperature_c=env.temperature_c,
        supply_voltage_v=supply_v,
        default_builtin_ref="builtin_demo_folded_cascode_small_signal_v1",
    )
    stimulus = [
        StimulusBinding(source_name="VDD", stimulus_type="supply", parameters={"value": supply_v}),
        StimulusBinding(source_name="VINP", stimulus_type="ac_input", parameters={"dc_value": vin_cm, "ac_amplitude": 1.0}),
        StimulusBinding(source_name="VINN", stimulus_type="bias", parameters={"value": vin_cm}),
    ]
    template = Template(folded_cascode_v1_netlist_template_path().read_text(encoding="utf-8"))
    rendered = template.safe_substitute(
        truth_mode="demonstrator_truth",
        template_id=task.topology.template_id or "folded_cascode_nominal_op_ac",
        p2_hint_hz=f"{p2_hint_hz:.6e}",
        vdd=f"{supply_v:.6e}",
        vin_cm=f"{vin_cm:.6e}",
        vin_step_high=f"{vin_step_high:.6e}",
        ibias=f"{ibias:.6e}",
        cc=f"{cc:.6e}",
        cload=f"{load_cap_f:.6e}",
        gm_in=f"{gm_in:.6e}",
        gm_fold=f"{gm_fold:.6e}",
        ro_in=f"{ro_in:.6e}",
        ro_fold=f"{ro_fold:.6e}",
        c_fold=f"{c_fold:.6e}",
    )
    return bindings, model_binding, stimulus, rendered


def _demonstrator_ldo_bindings(task: DesignTask, candidate: CandidateRecord) -> tuple[list[ParameterBinding], ModelBinding, list[StimulusBinding], str]:
    values = _parameter_map(candidate)
    env = candidate.world_state_snapshot.environment_state
    w_pass = values.get("w_pass", 200e-6)
    l_pass = max(values.get("l_pass", 0.5e-6), 1e-9)
    w_err = values.get("w_err", 8e-6)
    l_err = max(values.get("l_err", 1e-6), 1e-9)
    ibias = max(values.get("ibias", 1.0e-4), 1e-7)
    c_comp = max(values.get("c_comp", 5e-12), 1e-14)
    load_cap_f = float(env.load_cap_f or 5e-12)
    output_load_ohm = float(env.output_load_ohm or 1200.0)
    supply_v = float(env.supply_voltage_v or 1.2)

    width_ratio_pass = max(w_pass / l_pass, 1.0)
    width_ratio_err = max(w_err / l_err, 0.5)
    gm_pass = 0.025 * (ibias * width_ratio_pass) ** 0.5
    gm_err = 0.007 * (ibias * width_ratio_err) ** 0.5
    ro_err = max(3.5e5 * (l_err / max(w_err, 1e-9)) * (1e-4 / ibias) ** 0.4, 6e4)
    c_eff = max(load_cap_f + 0.3 * c_comp, 1e-15)
    p2_hint_hz = gm_pass / (2.0 * 3.141592653589793 * c_eff)
    vref_dc = round(min(0.7, supply_v * 0.5), 6)
    vref_step_high = round(min(vref_dc + 0.03, supply_v * 0.65), 6)
    rfb_bot = 60e3
    rfb_top = rfb_bot * max((1.0 / max(vref_dc, 1e-3)) - 1.0, 0.4)
    quiescent_current = max(1.8 * ibias, 1.2e-4)

    bindings = [
        ParameterBinding(variable_name="w_pass", netlist_target="param::w_pass", value_si=w_pass, units="m", source="user_override"),
        ParameterBinding(variable_name="l_pass", netlist_target="param::l_pass", value_si=l_pass, units="m", source="user_override"),
        ParameterBinding(variable_name="w_err", netlist_target="param::w_err", value_si=w_err, units="m", source="user_override"),
        ParameterBinding(variable_name="l_err", netlist_target="param::l_err", value_si=l_err, units="m", source="user_override"),
        ParameterBinding(variable_name="ibias", netlist_target="param::ibias", value_si=ibias, units="A", source="user_override"),
        ParameterBinding(variable_name="c_comp", netlist_target="param::c_comp", value_si=c_comp, units="F", source="user_override"),
        ParameterBinding(variable_name="cload", netlist_target="param::cload", value_si=load_cap_f, units="F", source="system_inferred"),
        ParameterBinding(variable_name="rload", netlist_target="param::rload", value_si=output_load_ohm, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="gm_pass", netlist_target="param::gm_pass", value_si=gm_pass, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="gm_err", netlist_target="param::gm_err", value_si=gm_err, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="ro_err", netlist_target="param::ro_err", value_si=ro_err, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="rfb_top", netlist_target="param::rfb_top", value_si=rfb_top, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="rfb_bot", netlist_target="param::rfb_bot", value_si=rfb_bot, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="p2_hint_hz", netlist_target="hint::p2_hz", value_si=p2_hint_hz, units="Hz", source="system_inferred"),
        ParameterBinding(variable_name="vref_step_high", netlist_target="param::vref_step_high", value_si=vref_step_high, units="V", source="system_inferred"),
        ParameterBinding(variable_name="quiescent_current", netlist_target="param::iquiescent", value_si=quiescent_current, units="A", source="system_inferred"),
    ]
    model_binding = _model_binding_from_config(
        backend="ngspice",
        process_node=task.parent_spec_id,
        corner=env.corner,
        temperature_c=env.temperature_c,
        supply_voltage_v=supply_v,
        default_builtin_ref="builtin_demo_ldo_loop_v1",
    )
    stimulus = [
        StimulusBinding(source_name="VDD", stimulus_type="supply", parameters={"value": supply_v}),
        StimulusBinding(source_name="VREF", stimulus_type="ac_input", parameters={"dc_value": vref_dc, "ac_amplitude": 1.0}),
    ]
    template = Template(ldo_v1_netlist_template_path().read_text(encoding="utf-8"))
    rendered = template.safe_substitute(
        truth_mode="demonstrator_truth",
        template_id=task.topology.template_id or "ldo_nominal_op_ac",
        p2_hint_hz=f"{p2_hint_hz:.6e}",
        vdd=f"{supply_v:.6e}",
        vref_dc=f"{vref_dc:.6e}",
        vref_step_high=f"{vref_step_high:.6e}",
        gm_err=f"{gm_err:.6e}",
        ro_err=f"{ro_err:.6e}",
        gm_pass=f"{gm_pass:.6e}",
        c_comp=f"{c_comp:.6e}",
        cload=f"{load_cap_f:.6e}",
        rload=f"{output_load_ohm:.6e}",
        rfb_top=f"{rfb_top:.6e}",
        rfb_bot=f"{rfb_bot:.6e}",
        quiescent_current=f"{quiescent_current:.6e}",
    )
    return bindings, model_binding, stimulus, rendered


def _demonstrator_bandgap_bindings(task: DesignTask, candidate: CandidateRecord) -> tuple[list[ParameterBinding], ModelBinding, list[StimulusBinding], str]:
    values = _parameter_map(candidate)
    env = candidate.world_state_snapshot.environment_state
    area_ratio = max(int(round(values.get("area_ratio", 8.0))), 1)
    r1 = max(values.get("r1", 12e3), 1e3)
    r2 = max(values.get("r2", 36e3), r1)
    w_core = values.get("w_core", 4e-6)
    l_core = max(values.get("l_core", 1e-6), 1e-9)
    ibias = max(values.get("ibias", 5e-6), 1e-8)
    supply_v = float(env.supply_voltage_v or 1.2)
    vdd_step_high = round(min(supply_v + 0.05, supply_v * 1.08), 6)

    width_ratio_core = max(w_core / l_core, 0.5)
    gm_core = 0.004 * (ibias * width_ratio_core) ** 0.5
    ro_core = max(1.5e6 * (l_core / max(w_core, 1e-9)) * (5e-6 / ibias) ** 0.35, 1.2e5)
    c_ref = max(0.08e-12 * area_ratio + 0.02e-12 * (r2 / max(r1, 1.0)), 5e-14)
    balance_target = 0.4 * area_ratio
    balance_error = abs((r2 / max(r1, 1.0)) - balance_target) / max(balance_target, 1e-6)
    tempco_hint = 18.0 + 18.0 * balance_error + 3.0 * max(0.0, 1.2 - width_ratio_core)

    bindings = [
        ParameterBinding(variable_name="area_ratio", netlist_target="param::area_ratio", value_si=area_ratio, units="ratio", source="user_override"),
        ParameterBinding(variable_name="r1", netlist_target="param::r1", value_si=r1, units="ohm", source="user_override"),
        ParameterBinding(variable_name="r2", netlist_target="param::r2", value_si=r2, units="ohm", source="user_override"),
        ParameterBinding(variable_name="w_core", netlist_target="param::w_core", value_si=w_core, units="m", source="user_override"),
        ParameterBinding(variable_name="l_core", netlist_target="param::l_core", value_si=l_core, units="m", source="user_override"),
        ParameterBinding(variable_name="ibias", netlist_target="param::ibias", value_si=ibias, units="A", source="user_override"),
        ParameterBinding(variable_name="gm_core", netlist_target="param::gm_core", value_si=gm_core, units="A/V", source="system_inferred"),
        ParameterBinding(variable_name="ro_core", netlist_target="param::ro_core", value_si=ro_core, units="ohm", source="system_inferred"),
        ParameterBinding(variable_name="c_ref", netlist_target="param::c_ref", value_si=c_ref, units="F", source="system_inferred"),
        ParameterBinding(variable_name="iref", netlist_target="param::iref", value_si=max(2.2 * ibias, 1e-6), units="A", source="system_inferred"),
        ParameterBinding(variable_name="vdd_step_high", netlist_target="param::vdd_step_high", value_si=vdd_step_high, units="V", source="system_inferred"),
        ParameterBinding(variable_name="tempco_hint_ppm_per_c", netlist_target="hint::tempco_ppm_per_c", value_si=tempco_hint, units="ppm_per_c", source="system_inferred"),
    ]
    model_binding = _model_binding_from_config(
        backend="ngspice",
        process_node=task.parent_spec_id,
        corner=env.corner,
        temperature_c=env.temperature_c,
        supply_voltage_v=supply_v,
        default_builtin_ref="builtin_demo_bandgap_reference_v1",
    )
    stimulus = [
        StimulusBinding(source_name="VDD", stimulus_type="supply", parameters={"value": supply_v, "step_high": vdd_step_high}),
    ]
    template = Template(bandgap_v1_netlist_template_path().read_text(encoding="utf-8"))
    rendered = template.safe_substitute(
        truth_mode="demonstrator_truth",
        template_id=task.topology.template_id or "bandgap_demonstrator_truth",
        tempco_hint_ppm_per_c=f"{tempco_hint:.6e}",
        vdd=f"{supply_v:.6e}",
        vdd_step_high=f"{vdd_step_high:.6e}",
        area_ratio=str(area_ratio),
        r1=f"{r1:.6e}",
        r2=f"{r2:.6e}",
        gm_core=f"{gm_core:.6e}",
        ro_core=f"{ro_core:.6e}",
        c_ref=f"{c_ref:.6e}",
        iref=f"{max(2.2 * ibias, 1e-6):.6e}",
    )
    return bindings, model_binding, stimulus, rendered


def realize_netlist_instance(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    backend: str,
    analyses: list[AnalysisStatement],
    model_binding_overrides: dict[str, float | int | str | bool] | None = None,
) -> NetlistInstance:
    """Realize a formal NetlistInstance for one candidate."""

    world_state = candidate.world_state_snapshot
    if backend == "ngspice" and task.circuit_family == "two_stage_ota" and task.topology.topology_mode == "fixed":
        bindings, model_binding, stimulus, rendered_netlist = _demonstrator_ota2_bindings(task, candidate)
        if model_binding_overrides:
            model_binding = _model_binding_from_config(
                backend="ngspice",
                process_node=task.parent_spec_id,
                corner=world_state.environment_state.corner,
                temperature_c=world_state.environment_state.temperature_c,
                supply_voltage_v=world_state.environment_state.supply_voltage_v,
                default_builtin_ref=model_binding.backend_model_ref,
                overrides=model_binding_overrides,
            )
    elif backend == "ngspice" and task.circuit_family == "folded_cascode_ota" and task.topology.topology_mode == "fixed":
        bindings, model_binding, stimulus, rendered_netlist = _demonstrator_folded_cascode_bindings(task, candidate)
        if model_binding_overrides:
            model_binding = _model_binding_from_config(
                backend="ngspice",
                process_node=task.parent_spec_id,
                corner=world_state.environment_state.corner,
                temperature_c=world_state.environment_state.temperature_c,
                supply_voltage_v=world_state.environment_state.supply_voltage_v,
                default_builtin_ref=model_binding.backend_model_ref,
                overrides=model_binding_overrides,
            )
    elif backend == "ngspice" and task.circuit_family == "ldo" and task.topology.topology_mode == "fixed":
        bindings, model_binding, stimulus, rendered_netlist = _demonstrator_ldo_bindings(task, candidate)
        if model_binding_overrides:
            model_binding = _model_binding_from_config(
                backend="ngspice",
                process_node=task.parent_spec_id,
                corner=world_state.environment_state.corner,
                temperature_c=world_state.environment_state.temperature_c,
                supply_voltage_v=world_state.environment_state.supply_voltage_v,
                default_builtin_ref=model_binding.backend_model_ref,
                overrides=model_binding_overrides,
            )
    elif backend == "ngspice" and task.circuit_family == "bandgap" and task.topology.topology_mode == "fixed":
        bindings, model_binding, stimulus, rendered_netlist = _demonstrator_bandgap_bindings(task, candidate)
        if model_binding_overrides:
            model_binding = _model_binding_from_config(
                backend="ngspice",
                process_node=task.parent_spec_id,
                corner=world_state.environment_state.corner,
                temperature_c=world_state.environment_state.temperature_c,
                supply_voltage_v=world_state.environment_state.supply_voltage_v,
                default_builtin_ref=model_binding.backend_model_ref,
                overrides=model_binding_overrides,
            )
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
        model_binding = _model_binding_from_config(
            backend=backend,
            process_node=task.parent_spec_id,
            corner=world_state.environment_state.corner,
            temperature_c=world_state.environment_state.temperature_c,
            supply_voltage_v=world_state.environment_state.supply_voltage_v,
            default_builtin_ref=f"{backend}::{task.circuit_family}::{world_state.environment_state.corner}",
            overrides=model_binding_overrides,
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
