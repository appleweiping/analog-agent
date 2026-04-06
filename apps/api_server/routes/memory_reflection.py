"""Memory and reflection layer API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.schema.design_task import DesignTask
from libs.schema.memory import IngestionResponse, MemoryCompileResponse, MemoryBundle, RetrievalResult
from libs.schema.planning import SearchState
from libs.schema.simulation import VerificationResult

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryContextRequest(BaseModel):
    """Request payload for compiling or querying memory."""

    model_config = ConfigDict(extra="forbid")

    memory_bundle: MemoryBundle | None = None


class MemoryIngestionRequest(BaseModel):
    """Request payload for trajectory ingestion."""

    model_config = ConfigDict(extra="forbid")

    memory_bundle: MemoryBundle
    design_task: DesignTask
    search_state: SearchState
    verification_result: VerificationResult | None = None


class MemoryRetrievalRequest(BaseModel):
    """Request payload for task-conditioned memory retrieval."""

    model_config = ConfigDict(extra="forbid")

    memory_bundle: MemoryBundle
    design_task: DesignTask


@router.post("/compile", response_model=MemoryCompileResponse)
def compile_bundle(_: MemoryContextRequest) -> MemoryCompileResponse:
    """Compile an empty formal memory bundle."""

    return compile_memory_bundle()


@router.post("/ingest", response_model=IngestionResponse)
def ingest(request: MemoryIngestionRequest) -> IngestionResponse:
    """Ingest one episode into the sixth-layer bundle."""

    return MemoryService(request.memory_bundle).ingest_episode(
        request.design_task,
        request.search_state,
        request.verification_result,
    )


@router.post("/retrieve", response_model=RetrievalResult)
def retrieve_memory(request: MemoryRetrievalRequest) -> RetrievalResult:
    """Retrieve task-relevant memory and advisory feedback."""

    return MemoryService(request.memory_bundle).retrieve_relevant_memory(request.design_task)
