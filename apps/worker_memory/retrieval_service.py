"""Worker-facing retrieval adapter."""

from __future__ import annotations

from libs.memory.service import MemoryService
from libs.schema.design_task import DesignTask
from libs.schema.memory import MemoryBundle, RetrievalResult


def retrieve(bundle: MemoryBundle, design_task: DesignTask) -> RetrievalResult:
    """Retrieve task-conditioned memory from the formal sixth-layer service."""

    return MemoryService(bundle).retrieve_relevant_memory(design_task)
