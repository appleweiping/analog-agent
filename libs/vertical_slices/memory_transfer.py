"""Cross-task memory transfer evidence entry points for frozen vertical slices."""

from __future__ import annotations

from pathlib import Path

from libs.eval.memory_evidence import (
    build_memory_transfer_evidence_bundle,
    run_cross_task_memory_transfer_suite,
)
from libs.schema.memory_evidence import MemoryTransferEvidenceBundle, MemoryTransferSuiteResult
from libs.vertical_slices.bandgap_spec import build_bandgap_v1_design_task
from libs.vertical_slices.folded_cascode_spec import build_folded_cascode_v1_design_task
from libs.vertical_slices.ldo_spec import build_ldo_v1_design_task
from libs.vertical_slices.ota2_spec import build_ota2_v1_design_task


def run_ota_to_folded_cascode_memory_transfer_suite(
    *,
    source_episodes: int = 3,
    target_episodes: int = 3,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryTransferSuiteResult:
    return run_cross_task_memory_transfer_suite(
        source_task_slug="ota2-v1",
        source_task_builder=build_ota2_v1_design_task,
        target_task_slug="folded_cascode-v1",
        target_task_builder=build_folded_cascode_v1_design_task,
        transfer_kind="same_family",
        source_episodes=source_episodes,
        target_episodes=target_episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
    )


def run_ota_to_ldo_memory_transfer_suite(
    *,
    source_episodes: int = 3,
    target_episodes: int = 3,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryTransferSuiteResult:
    return run_cross_task_memory_transfer_suite(
        source_task_slug="ota2-v1",
        source_task_builder=build_ota2_v1_design_task,
        target_task_slug="ldo-v1",
        target_task_builder=build_ldo_v1_design_task,
        transfer_kind="cross_family",
        source_episodes=source_episodes,
        target_episodes=target_episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
    )


def run_folded_cascode_to_ota_memory_transfer_suite(
    *,
    source_episodes: int = 3,
    target_episodes: int = 3,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryTransferSuiteResult:
    return run_cross_task_memory_transfer_suite(
        source_task_slug="folded_cascode-v1",
        source_task_builder=build_folded_cascode_v1_design_task,
        target_task_slug="ota2-v1",
        target_task_builder=build_ota2_v1_design_task,
        transfer_kind="same_family",
        source_episodes=source_episodes,
        target_episodes=target_episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
    )


def run_ota_to_bandgap_memory_transfer_suite(
    *,
    source_episodes: int = 3,
    target_episodes: int = 3,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryTransferSuiteResult:
    return run_cross_task_memory_transfer_suite(
        source_task_slug="ota2-v1",
        source_task_builder=build_ota2_v1_design_task,
        target_task_slug="bandgap-v1",
        target_task_builder=build_bandgap_v1_design_task,
        transfer_kind="cross_family",
        source_episodes=source_episodes,
        target_episodes=target_episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
    )


def run_memory_transfer_evidence(
    *,
    source_task_slug: str,
    target_task_slug: str,
    suite: MemoryTransferSuiteResult,
    output_root: str | Path,
) -> MemoryTransferEvidenceBundle:
    root = Path(output_root)
    return build_memory_transfer_evidence_bundle(
        suite,
        figures_dir=root / "memory_transfer_figs",
        tables_dir=root / "memory_transfer_tables",
        json_output_path=root / "memory_transfer_evidence_bundle.json",
    )
