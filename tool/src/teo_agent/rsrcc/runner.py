"""Constrained tool executor with trace and budget enforcement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import time
from typing import Any, Mapping

from .schemas import ToolResponse
from .tools import RsrccTools


TOOL_NAMES = (
    "load_rgb_pair",
    "inspect_rgb_pair",
    "check_pixel_alignment",
    "correct_pixel_alignment",
    "detect_multiscale_change",
    "parse_scene_entities",
    "extract_linear_infrastructure",
    "parse_paved_and_parking_areas",
    "detect_small_aux_objects",
    "match_entities_across_time",
    "compare_entity_attributes",
    "build_scene_graph",
    "resolve_spatial_query",
    "infer_change_event",
    "summarize_roi_change",
    "score_answer_options",
    "assess_global_scene_change",
    "render_evidence",
    "verify_evidence_consistency",
    "validate_output_schema",
)


@dataclass(frozen=True)
class ToolTraceStep:
    step_id: str
    tool: str
    arguments: Mapping[str, Any]
    artifacts: Mapping[str, str]
    observation: Mapping[str, Any]
    elapsed_ms: float


class ToolExecutor:
    """Allowlisted executor; observations never include stored raster payloads."""

    def __init__(self, tools: RsrccTools, *, max_calls: int = 64) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.tools = tools
        self.max_calls = max_calls
        self._trace: list[ToolTraceStep] = []

    @property
    def tool_names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    @property
    def trace(self) -> tuple[ToolTraceStep, ...]:
        return tuple(self._trace)

    def trace_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self._trace]

    def call(self, tool: str, arguments: Mapping[str, Any]) -> ToolResponse:
        if tool not in TOOL_NAMES:
            raise ValueError(f"tool is not allowlisted: {tool}")
        if len(self._trace) >= self.max_calls:
            raise RuntimeError("tool-call budget exhausted")
        method = getattr(self.tools, tool)
        signature = inspect.signature(method)
        try:
            signature.bind(**arguments)
        except TypeError as exc:
            raise ValueError(f"invalid arguments for {tool}: {exc}") from exc
        started = time.perf_counter()
        response = method(**dict(arguments))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(response, ToolResponse):
            raise TypeError(f"tool {tool} returned {type(response).__name__}, expected ToolResponse")
        for artifact_id in response.artifacts.values():
            if artifact_id not in self.tools.store:
                raise RuntimeError(f"tool {tool} returned an unknown artifact ID")
        step = ToolTraceStep(
            step_id=f"s{len(self._trace) + 1}",
            tool=tool,
            arguments=dict(arguments),
            artifacts=dict(response.artifacts),
            observation=dict(response.observation),
            elapsed_ms=elapsed_ms,
        )
        self._trace.append(step)
        return response

