"""Stable schemas and controlled vocabularies for RSRCC tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


EVENT_VOCAB = frozenset(
    {
        "constructed",
        "appeared",
        "installed",
        "removed",
        "demolished",
        "destroyed",
        "cleared",
        "expanded",
        "enlarged",
        "extended",
        "widened",
        "narrowed",
        "replaced",
        "converted",
        "paved",
        "unpaved",
        "developed",
        "grown",
        "planted",
        "vegetation_increase",
        "vegetation_loss",
        "water_increase",
        "water_decrease",
        "repainted",
        "roof_color_changed",
        "surface_covered",
        "construction_completed",
        "object_added",
        "object_removed",
        "count_increased",
        "count_decreased",
        "layout_changed",
        "obstructed",
        "unchanged",
        "local_change",
        "global_change",
    }
)


ONTOLOGY_ALIASES: dict[str, tuple[str, ...]] = {
    "building": (
        "building",
        "house",
        "structure",
        "shed",
        "warehouse",
        "commercial building",
        "apartment",
        "mobile home",
        "greenhouse",
        "gazebo",
        "roof",
        "foundation",
        "debris",
    ),
    "road": (
        "road",
        "street",
        "sidewalk",
        "driveway",
        "trail",
        "path",
        "bike lane",
        "railway",
        "bridge",
        "intersection",
        "cul-de-sac",
        "roundabout",
        "crosswalk",
    ),
    "parking_lot": (
        "parking lot",
        "parking stalls",
        "pavement",
        "paved pad",
        "impervious surface",
    ),
    "vegetation": (
        "tree",
        "tree cluster",
        "vegetation",
        "forest",
        "woodland",
        "grass",
        "lawn",
        "shrub",
        "garden",
        "landscaped island",
    ),
    "water": (
        "water",
        "pond",
        "lake",
        "stream",
        "canal",
        "swimming pool",
        "tennis court",
        "athletic field",
        "stadium",
        "playground",
    ),
    "small_object": (
        "vehicle",
        "car",
        "truck",
        "boat",
        "container",
        "fence",
        "wall",
        "solar panel",
        "construction equipment",
    ),
    "terrain": (
        "field",
        "agricultural land",
        "open land",
        "vacant lot",
        "dirt",
        "bare ground",
        "terrain",
    ),
}


LINEAR_LABELS = frozenset(
    {
        "road",
        "street",
        "sidewalk",
        "driveway",
        "trail",
        "path",
        "bike_lane",
        "railway",
        "bridge",
        "crosswalk",
    }
)
PAVED_LABELS = frozenset(
    {"parking_lot", "parking", "pavement", "paved_pad", "impervious_surface"}
)
SMALL_OBJECT_LABELS = frozenset(
    {
        "vehicle",
        "car",
        "truck",
        "boat",
        "container",
        "fence",
        "wall",
        "solar_panel",
        "construction_equipment",
    }
)


@dataclass(frozen=True)
class ToolResponse:
    """What the executor returns to the controller."""

    artifacts: Mapping[str, str]
    observation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuerySpec:
    labels: tuple[str, ...] = ()
    region: str | None = None
    relations: tuple[Mapping[str, str], ...] = ()
    ordinal: str | None = None

    @classmethod
    def from_value(cls, value: "QuerySpec | Mapping[str, Any]") -> "QuerySpec":
        if isinstance(value, cls):
            return value
        labels = value.get("labels", ())
        if isinstance(labels, str):
            labels = (labels,)
        return cls(
            labels=tuple(str(label) for label in labels),
            region=value.get("region"),
            relations=tuple(value.get("relations", ())),
            ordinal=value.get("ordinal"),
        )


@dataclass(frozen=True)
class ParsedRsrccTask:
    question: str
    options: Mapping[str, str]
    answer_format: str
    gold_answer: str

    def agent_view(self) -> dict[str, Any]:
        """Return fields safe to expose to the agent; gold never leaks."""

        result: dict[str, Any] = {
            "question": self.question,
            "answer_format": self.answer_format,
        }
        if self.options:
            result["options"] = dict(self.options)
        return result

