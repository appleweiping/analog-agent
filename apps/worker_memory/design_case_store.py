"""Worker-facing pattern/reflection store adapter."""

from __future__ import annotations

from libs.schema.memory import MemoryBundle, PatternMemoryRecord, ReflectionReport

class DesignCaseStore:
    """Persist structured pattern and reflection records."""

    def write(
        self,
        bundle: MemoryBundle,
        *,
        patterns: list[PatternMemoryRecord],
        reflection: ReflectionReport | None,
    ) -> MemoryBundle:
        reflection_records = [*bundle.reflection_records, reflection] if reflection is not None else list(bundle.reflection_records)
        return bundle.model_copy(
            update={
                "pattern_records": patterns,
                "reflection_records": reflection_records,
                "indexing_state": bundle.indexing_state.model_copy(
                    update={
                        "pattern_count": len(patterns),
                        "reflection_count": len(reflection_records),
                    }
                ),
            }
        )
