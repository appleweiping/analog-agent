"""Raw simulator output parsing helpers."""

from __future__ import annotations


def parse_raw_output(raw_output: dict[str, object]) -> dict[str, object]:
    """Normalize backend payloads into a parser-neutral structure."""

    return {
        "status": raw_output.get("status", "unknown"),
        "backend": raw_output.get("backend", "unknown"),
        "analysis_type": raw_output.get("analysis_type", "unknown"),
        "metrics": dict(raw_output.get("metrics", {})),
        "op_diagnostics": dict(raw_output.get("op_diagnostics", {})),
        "runtime_ms": int(raw_output.get("runtime_ms", 0)),
        "corner": raw_output.get("corner"),
        "temperature_c": raw_output.get("temperature_c"),
        "load_cap_f": raw_output.get("load_cap_f"),
    }
