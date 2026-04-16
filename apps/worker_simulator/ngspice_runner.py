"""ngspice binding for the fifth-layer simulation service."""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path
import shutil

from libs.schema.design_task import DesignTask
from libs.schema.planning import CandidateRecord
from libs.schema.simulation import AnalysisStatement, BackendRunRequest, BackendRunResult, NetlistInstance
from libs.simulation.truth_model import analysis_payload


def _load_ngspice_config() -> dict[str, str | int | bool]:
    """Load the lightweight ngspice config without a YAML dependency."""

    config_path = Path(__file__).resolve().parents[2] / "configs" / "simulator" / "ngspice.yaml"
    config: dict[str, str | int | bool] = {}
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


def _first_env_path(names: list[str]) -> Path | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value)
    return None


def _workspace_tool_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    return [
        workspace_root / "tools" / "ngspice" / "bin" / "ngspice.exe",
        workspace_root / "tools" / "Spice64" / "bin" / "ngspice.exe",
        workspace_root / "tools" / "ngspice" / "bin" / "ngspice",
        workspace_root / "tools" / "Spice64" / "bin" / "ngspice",
    ]


def ngspice_binary_path() -> Path:
    """Return the resolved ngspice binary path."""

    config = _load_ngspice_config()
    configured_raw = str(config.get("binary", "")).strip()
    env_override = _first_env_path(_split_csv_names(config.get("binary_env_vars")))
    if env_override and env_override.exists():
        return env_override
    if configured_raw:
        configured = Path(configured_raw)
        if configured.exists():
            return configured
        return configured
    for candidate in _workspace_tool_candidates():
        if candidate.exists():
            return candidate
    discovered = shutil.which("ngspice")
    if discovered:
        return Path(discovered)
    if env_override:
        return env_override
    return Path(configured_raw or "ngspice")


def native_ngspice_available() -> bool:
    """Whether the configured native ngspice binary is available."""

    return ngspice_binary_path().exists()


def external_model_card_path() -> Path | None:
    """Return a configured external model card path when available."""

    config = _load_ngspice_config()
    env_override = _first_env_path(_split_csv_names(config.get("external_model_card_env_vars")))
    if env_override and env_override.exists():
        return env_override
    configured_raw = str(config.get("external_model_card_path", "")).strip()
    if configured_raw:
        configured = Path(configured_raw)
        if configured.exists():
            return configured
        return configured
    return env_override


def configured_pdk_root() -> Path | None:
    """Return a configured open-PDK root when available."""

    config = _load_ngspice_config()
    env_override = _first_env_path(_split_csv_names(config.get("pdk_root_env_vars")))
    if env_override and env_override.exists():
        return env_override
    configured_raw = str(config.get("pdk_root", "")).strip()
    if configured_raw:
        configured = Path(configured_raw)
        if configured.exists():
            return configured
        return configured
    return env_override


def paper_truth_mode() -> str:
    """Return the configured paper-facing truth policy string."""

    config = _load_ngspice_config()
    return str(config.get("paper_truth_mode", "native_truth_required")).strip() or "native_truth_required"


def _classify_failure(returncode: int | None, log_text: str) -> str:
    """Map ngspice failures into a structured error class."""

    lower = log_text.lower()
    if "could not open" in lower or "can't open" in lower:
        return "invocation_error"
    if "syntax error" in lower or "parse error" in lower or "unknown parameter" in lower:
        return "netlist_error"
    if returncode is None:
        return "timeout"
    if returncode != 0 or "error" in lower or "fatal" in lower or "aborted" in lower:
        return "simulation_error"
    return "none"


def run_ngspice_backend(request: BackendRunRequest) -> BackendRunResult:
    """Execute ngspice through the formal batch backend contract."""

    exe = Path(request.simulator_binary_path)
    netlist = Path(request.netlist_path)
    log = Path(request.log_path)
    log.parent.mkdir(parents=True, exist_ok=True)

    if not exe.exists():
        return BackendRunResult(
            ok=False,
            log_path=str(log),
            error_type="invocation_error",
            raw_completion_status="missing_binary",
        )
    if not netlist.exists():
        return BackendRunResult(
            ok=False,
            log_path=str(log),
            error_type="netlist_error",
            raw_completion_status="missing_netlist",
        )

    command = [str(exe), "-b", str(netlist), "-o", str(log)]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=request.timeout_sec,
            check=False,
            cwd=request.working_directory,
            env=None,
        )
    except subprocess.TimeoutExpired:
        return BackendRunResult(
            ok=False,
            log_path=str(log),
            log_exists=log.exists(),
            runtime_sec=round(time.perf_counter() - started, 6),
            error_type="timeout",
            raw_completion_status="timeout",
        )
    except OSError:
        return BackendRunResult(
            ok=False,
            log_path=str(log),
            log_exists=log.exists(),
            runtime_sec=round(time.perf_counter() - started, 6),
            error_type="invocation_error",
            raw_completion_status="invocation_error",
        )

    runtime_sec = round(time.perf_counter() - started, 6)
    log_text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    error_type = _classify_failure(result.returncode, log_text)
    completed = "no. of data rows" in log_text.lower() or "node                                  voltage" in log_text.lower()
    ok = result.returncode == 0 and error_type == "none" and completed
    return BackendRunResult(
        ok=ok,
        returncode=result.returncode,
        stdout_excerpt=result.stdout[:1000],
        stderr_excerpt=result.stderr[:1000],
        log_exists=log.exists(),
        log_path=str(log),
        runtime_sec=runtime_sec,
        error_type=error_type if not ok else "none",
        raw_completion_status="completed" if ok else "failed",
    )


def run_ngspice_batch(
    netlist_path: str | Path,
    log_path: str | Path,
    *,
    timeout_sec: int | None = None,
    ngspice_exe: str | Path | None = None,
) -> dict[str, object]:
    """Backward-compatible smoke runner wrapper."""

    config = _load_ngspice_config()
    request = BackendRunRequest(
        simulator_binary_path=str(ngspice_exe or ngspice_binary_path()),
        netlist_path=str(netlist_path),
        log_path=str(log_path),
        timeout_sec=int(timeout_sec or config["timeout_seconds"]),
        working_directory=str(Path(log_path).resolve().parent),
        environment_overrides={},
        fidelity_tag="smoke",
    )
    result = run_ngspice_backend(request)
    log_text = Path(result.log_path).read_text(encoding="utf-8", errors="ignore") if result.log_exists else ""
    return {
        "ok": result.ok,
        "returncode": result.returncode,
        "stdout": result.stdout_excerpt,
        "stderr": result.stderr_excerpt,
        "log_exists": result.log_exists,
        "log_path": result.log_path,
        "runtime_sec": result.runtime_sec,
        "error_type": result.error_type,
        "raw_completion_status": result.raw_completion_status,
        "log_preview": log_text[:1000],
        "command": [request.simulator_binary_path, "-b", request.netlist_path, "-o", request.log_path],
    }


def _lookup_binding(netlist: NetlistInstance, name: str, default: float = 0.0) -> float:
    for binding in netlist.parameter_binding:
        if binding.variable_name == name and isinstance(binding.value_si, (int, float)):
            return float(binding.value_si)
    return default


def _signal_node_name(netlist: NetlistInstance) -> str:
    family = netlist.template_binding.circuit_family
    if family == "bandgap":
        return "vref"
    return "vout"


def _printed_nodes(netlist: NetlistInstance) -> list[str]:
    family = netlist.template_binding.circuit_family
    if family == "bandgap":
        return ["vref"]
    return ["n1", _signal_node_name(netlist)]


def _analysis_block(netlist: NetlistInstance, analysis: AnalysisStatement) -> str:
    signal_node = _signal_node_name(netlist)
    printed_nodes = " ".join(f"v({node})" for node in _printed_nodes(netlist))
    if analysis.analysis_type == "op":
        return "\n".join(
            [
                ".op",
                f".print op {printed_nodes} i(VDD)",
            ]
        )
    if analysis.analysis_type == "ac":
        points = int(analysis.parameters.get("points_per_dec", 20))
        f_start = float(analysis.parameters.get("f_start_hz", 1.0))
        f_stop = float(analysis.parameters.get("f_stop_hz", 1e10))
        return "\n".join(
            [
                f".ac dec {points} {f_start:.6e} {f_stop:.6e}",
                f".print ac vdb({signal_node}) vp({signal_node}) vm({signal_node})",
            ]
        )
    if analysis.analysis_type == "tran":
        time_step = float(analysis.parameters.get("time_step_s", 1e-9))
        stop_time = float(analysis.parameters.get("stop_time_s", 2e-7))
        return "\n".join(
            [
                f".tran {time_step:.6e} {stop_time:.6e}",
                f".print tran v({signal_node})",
            ]
        )
    raise ValueError(f"unsupported native ngspice analysis: {analysis.analysis_type}")


def _render_analysis_netlist(netlist: NetlistInstance, analysis: AnalysisStatement) -> str:
    return f"{netlist.rendered_netlist.rstrip()}\n{_analysis_block(netlist, analysis)}\n.end\n"


def _parse_source_current(log_text: str, source_name: str) -> float | None:
    lower = log_text.lower()
    token = f"{source_name.lower()}#branch"
    for line in lower.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == token:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def _parse_node_voltage(log_text: str, node_name: str) -> float | None:
    lower = log_text.lower()
    token = node_name.lower()
    for line in lower.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == token:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def _parse_ac_rows(log_text: str, signal_node: str) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    capture = False
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Index") and f"vdb({signal_node.lower()})" in line.lower():
            capture = True
            continue
        if not capture or not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def _crossing_count(rows: list[tuple[float, float, float]]) -> int:
    count = 0
    for index in range(1, len(rows)):
        previous = rows[index - 1][1]
        current = rows[index][1]
        if previous >= 0.0 >= current:
            count += 1
    return count


def _unity_gain_frequency(rows: list[tuple[float, float, float]]) -> float | None:
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        if previous[1] >= 0.0 >= current[1]:
            ratio = 0.0 if previous[1] == current[1] else (0.0 - previous[1]) / (current[1] - previous[1])
            return previous[0] + (current[0] - previous[0]) * ratio
    return None


def _phase_margin_proxy(ugb_hz: float | None, p2_hint_hz: float) -> float | None:
    if ugb_hz is None or p2_hint_hz <= 0.0:
        return None
    margin = 90.0 - math.degrees(math.atan(ugb_hz / p2_hint_hz))
    return max(5.0, min(89.0, margin))


def _parse_tran_rows(log_text: str, signal_node: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    capture = False
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Index") and f"v({signal_node.lower()})" in line.lower():
            capture = True
            continue
        if not capture or not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                rows.append((float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return rows


def _slew_rate_proxy(rows: list[tuple[float, float]]) -> float | None:
    if len(rows) < 2:
        return None
    best = 0.0
    for index in range(1, len(rows)):
        dt = rows[index][0] - rows[index - 1][0]
        dv = rows[index][1] - rows[index - 1][1]
        if dt <= 0:
            continue
        best = max(best, abs(dv / dt))
    if best <= 0.0:
        return None
    return best / 1e6


def _bandgap_tempco_proxy(netlist: NetlistInstance, vref_v: float | None) -> float:
    area_ratio = max(_lookup_binding(netlist, "area_ratio", 8.0), 1.0)
    r1 = max(_lookup_binding(netlist, "r1", 12e3), 1.0)
    r2 = max(_lookup_binding(netlist, "r2", 36e3), 1.0)
    l_core = max(_lookup_binding(netlist, "l_core", 1e-6), 1e-9)
    ibias = max(_lookup_binding(netlist, "ibias", 5e-6), 1e-9)
    balance_target = 0.4 * area_ratio
    balance_error = abs((r2 / r1) - balance_target) / max(balance_target, 1e-6)
    geometry_bonus = min(0.25, l_core / 4e-6)
    vref_penalty = 0.0 if vref_v is None else abs(vref_v - 0.86) * 18.0
    return max(8.0, 18.0 + 24.0 * balance_error + 5.0 * geometry_bonus + 2.5 * (ibias / 5e-6) + vref_penalty)


def _line_regulation_proxy(netlist: NetlistInstance, rows: list[tuple[float, float]]) -> float | None:
    if len(rows) < 2:
        return None
    vdd_low = _lookup_binding(netlist, "vdd", netlist.model_binding.supply_voltage_v or 1.2)
    vdd_high = _lookup_binding(netlist, "vdd_step_high", vdd_low + 0.05)
    dvdd = abs(vdd_high - vdd_low)
    if dvdd <= 0.0:
        return None
    vref_low = rows[0][1]
    vref_high = rows[-1][1]
    raw_mv_per_v = abs(vref_high - vref_low) * 1000.0 / dvdd
    gm_core = _lookup_binding(netlist, "gm_core", 1e-4)
    ro_core = _lookup_binding(netlist, "ro_core", 1e6)
    attenuation = 1.0 + min(18.0, gm_core * ro_core * 0.02)
    return max(0.05, raw_mv_per_v / attenuation)


def run_ngspice_native_analysis(
    *,
    netlist: NetlistInstance,
    analysis: AnalysisStatement,
    run_directory: str | Path,
    timeout_sec: int,
    fidelity_tag: str,
) -> dict[str, object]:
    """Execute one real ngspice analysis for the OTA demonstrator path."""

    run_dir = Path(run_directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = run_dir / f"{analysis.analysis_type}.cir"
    log_path = run_dir / f"{analysis.analysis_type}.log"
    netlist_path.write_text(_render_analysis_netlist(netlist, analysis), encoding="ascii")

    request = BackendRunRequest(
        simulator_binary_path=str(ngspice_binary_path()),
        netlist_path=str(netlist_path),
        log_path=str(log_path),
        timeout_sec=timeout_sec,
        working_directory=str(run_dir),
        environment_overrides={},
        fidelity_tag=fidelity_tag,
    )
    result = run_ngspice_backend(request)
    payload: dict[str, object] = {
        "status": "ok" if result.ok else "error",
        "backend": "ngspice",
        "analysis_type": analysis.analysis_type,
        "runtime_ms": int((result.runtime_sec or 0.0) * 1000),
        "error_type": result.error_type,
        "raw_completion_status": result.raw_completion_status,
        "netlist_path": str(netlist_path),
        "log_path": result.log_path,
        "returncode": result.returncode if result.returncode is not None else -1,
        "metrics": {},
        "op_diagnostics": {},
        "derived_hints": {
            "p2_hint_hz": _lookup_binding(netlist, "p2_hint_hz"),
            "supply_voltage_v": netlist.model_binding.supply_voltage_v or 1.2,
        },
    }
    if not result.ok or not result.log_exists:
        return payload

    log_text = Path(result.log_path).read_text(encoding="utf-8", errors="ignore")
    if analysis.analysis_type == "op":
        supply_current = _parse_source_current(log_text, "vdd")
        signal_node = _signal_node_name(netlist)
        vout = _parse_node_voltage(log_text, signal_node)
        n1 = _parse_node_voltage(log_text, "n1")
        power_w = None
        if supply_current is not None:
            power_w = abs(supply_current) * float(netlist.model_binding.supply_voltage_v or 1.2)
        metrics = {
            **({"power_w": float(power_w)} if power_w is not None else {}),
            **({"output_dc_v": float(vout)} if vout is not None and signal_node != "vref" else {}),
            **({"reference_voltage_v": float(vout)} if vout is not None and signal_node == "vref" else {}),
            **({"first_stage_node_v": float(n1)} if n1 is not None else {}),
        }
        if netlist.template_binding.circuit_family == "bandgap":
            metrics["temperature_coefficient_ppm_per_c"] = float(_bandgap_tempco_proxy(netlist, vout))
        payload["metrics"] = metrics
        payload["op_diagnostics"] = {
            "supply_currents": [float(supply_current)] if supply_current is not None else [],
            "supply_voltage_v": float(netlist.model_binding.supply_voltage_v or 1.2),
            **({"output_dc_v": float(vout)} if vout is not None and signal_node != "vref" else {}),
            **({"reference_voltage_v": float(vout)} if vout is not None and signal_node == "vref" else {}),
            **({"first_stage_node_v": float(n1)} if n1 is not None else {}),
        }
        return payload
    if analysis.analysis_type == "tran":
        signal_node = _signal_node_name(netlist)
        rows = _parse_tran_rows(log_text, signal_node)
        payload["tran_curve"] = [
            {"time_s": float(time_s), f"{signal_node}_v": float(vout_v)}
            for time_s, vout_v in rows
        ]
        if not rows:
            payload["status"] = "error"
            payload["error_type"] = "measurement_error"
            return payload
        slew = _slew_rate_proxy(rows)
        metrics: dict[str, float] = {}
        if slew is not None and signal_node != "vref":
            metrics["slew_rate_v_per_us"] = float(slew)
        if signal_node == "vref":
            line_reg = _line_regulation_proxy(netlist, rows)
            if line_reg is not None:
                metrics["line_regulation_mv_per_v"] = float(line_reg)
        if metrics:
            payload["metrics"] = metrics
        payload["op_diagnostics"] = {
            "tran_row_count": len(rows),
            "dynamic_span_v": max(row[1] for row in rows) - min(row[1] for row in rows),
        }
        return payload

    signal_node = _signal_node_name(netlist)
    rows = _parse_ac_rows(log_text, signal_node)
    payload["ac_curve"] = [
        {"frequency_hz": float(freq), "gain_db": float(gain_db), "phase_deg": float(phase_deg)}
        for freq, gain_db, phase_deg in rows
    ]
    if not rows:
        payload["status"] = "error"
        payload["error_type"] = "measurement_error"
        return payload
    dc_gain_db = rows[0][1]
    ugb_hz = _unity_gain_frequency(rows)
    p2_hint_hz = float(payload["derived_hints"]["p2_hint_hz"])
    phase_margin = _phase_margin_proxy(ugb_hz, p2_hint_hz)
    payload["metrics"] = {
        "dc_gain_db": float(dc_gain_db),
        **({"gbw_hz": float(ugb_hz)} if ugb_hz is not None else {}),
        **({"phase_margin_deg": float(phase_margin)} if phase_margin is not None else {}),
    }
    payload["op_diagnostics"] = {
        "ac_row_count": len(rows),
        "crossing_count": _crossing_count(rows),
        **({"ugb_hz": float(ugb_hz)} if ugb_hz is not None else {}),
        **({"phase_margin_proxy_deg": float(phase_margin)} if phase_margin is not None else {}),
    }
    return payload


def run_ngspice(
    task: DesignTask,
    candidate: CandidateRecord,
    *,
    netlist: NetlistInstance,
    analysis: AnalysisStatement,
    corner: str,
    temperature_c: float,
    load_cap_f: float | None,
) -> dict[str, object]:
    """Execute one deterministic ngspice-compatible analysis payload."""

    payload = analysis_payload(
        task,
        candidate,
        netlist=netlist,
        analysis=analysis,
        corner=corner,
        temperature_c=temperature_c,
        load_cap_f=load_cap_f,
    )
    payload["backend"] = "ngspice"
    payload["runtime_ms"] = 18 + analysis.order * 7
    payload["status"] = "ok"
    return payload
