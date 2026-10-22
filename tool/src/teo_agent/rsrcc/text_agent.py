"""A text-only controller that plans and executes RSRCC capability tools.

The controller never reads image arrays, masks, or rendered evidence pixels.
Its state is limited to the natural-language task, tool schemas, artifact IDs,
and JSON-compatible observations returned by :class:`ToolExecutor`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from .runner import ToolExecutor
from .schemas import EVENT_VOCAB


_LINEAR_TERMS = (
    "road",
    "street",
    "sidewalk",
    "driveway",
    "trail",
    "path",
    "railway",
    "bridge",
    "intersection",
    "cul-de-sac",
    "roundabout",
    "crosswalk",
)
_PAVED_TERMS = ("parking", "paved", "pavement", "concrete", "asphalt")
_SMALL_OBJECT_TERMS = (
    "vehicle",
    "car",
    "truck",
    "boat",
    "container",
    "fence",
    "wall",
    "solar panel",
    "construction equipment",
)
_GLOBAL_TERMS = (
    "observable discrepancy",
    "scene consistency",
    "visual parity",
    "global alteration",
    "temporal relationship",
    "temporal snapshots",
    "temporal coherence",
    "overall composition",
    "overall structural",
    "overall scene",
    "two captures",
    "two provided images",
)


@dataclass(frozen=True)
class TextAgentPlan:
    labels: tuple[str, ...]
    region: str | None
    ordinal: str | None
    global_question: bool
    use_linear_parser: bool
    use_paved_parser: bool
    use_small_object_parser: bool
    hypothesis_events: tuple[str, ...]
    planning_notes: tuple[str, ...]


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _infer_region(question: str) -> str | None:
    normalized = question.lower().replace("-", " ")
    groups = (
        ("northwest", ("northwest", "northwestern", "top left", "upper left")),
        ("northeast", ("northeast", "northeastern", "top right", "upper right")),
        ("southwest", ("southwest", "southwestern", "bottom left", "lower left")),
        ("southeast", ("southeast", "southeastern", "bottom right", "lower right")),
        ("north", ("northern", "north side", "top center", "top portion")),
        ("south", ("southern", "south side", "bottom center", "bottom portion")),
        ("east", ("eastern", "east side", "right side", "right portion")),
        ("west", ("western", "west side", "left side", "left portion")),
        ("center", ("central", "center", "middle")),
    )
    for region, phrases in groups:
        if any(phrase in normalized for phrase in phrases):
            return region
    return None


def _infer_label(question: str, global_question: bool) -> tuple[str, ...]:
    if global_question:
        return ()
    normalized = question.lower()
    candidates = (
        ("water", ("swimming pool", "pool", "pond", "lake", "water", "canal", "stream")),
        (
            "vegetation",
            (
                "tree canopy",
                "tree cluster",
                "wooded area",
                "woodland",
                "tree",
                "trees",
                "vegetation",
                "forest",
                "grass",
                "lawn",
                "shrub",
                "garden",
            ),
        ),
        ("vehicle", ("vehicle", "car", "truck", "boat")),
        ("parking_lot", ("parking lot", "parking area", "parking")),
        (
            "road",
            _LINEAR_TERMS
            + (
                "pedestrian infrastructure",
                "railroad tracks",
                "railway tracks",
                "bike lane",
            ),
        ),
        (
            "building",
            (
                "building",
                "buildings",
                "house",
                "houses",
                "residence",
                "residential development",
                "residential structure",
                "warehouse",
                "roof",
                "structure",
            ),
        ),
        (
            "terrain",
            (
                "open area",
                "open land",
                "wooded area",
                "vacant plot",
                "undeveloped plot",
                "undeveloped area",
                "bare ground",
                "clearing",
                "plot",
                "lot",
                "field",
                "terrain",
                "undeveloped land",
                "site",
                "property",
            ),
        ),
    )
    # Prefer the object mentioned closest to the question's change predicate.
    occurrences: list[tuple[int, int, str]] = []
    for label, phrases in candidates:
        for phrase in phrases:
            pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
            for match in re.finditer(pattern, normalized):
                occurrences.append((match.start(), match.end(), label))
    if not occurrences:
        return ()
    # Binary/passive questions put the target in the grammatical subject and
    # often mention one or more anchor objects after the predicate. Restricting
    # the choice to that subject prevents "tree ... near a road" from becoming
    # a road query and "pool ... behind a residence" from becoming a building
    # query.
    passive = re.search(
        r"\b(?:has|have|had|is|are|was|were)\s+"
        r"(?P<subject>[^?]{1,220}?)\s+"
        r"(?:been\s+)?(?:newly\s+)?"
        r"(?:constructed|built|established|installed|appeared|removed|demolished|"
        r"destroyed|cleared|expanded|enlarged|extended|widened|narrowed|paved|"
        r"unpaved|planted|grown|increased|decreased|filled|repainted|painted|"
        r"replaced|converted|developed|obstructed)\b",
        normalized,
    )
    if passive:
        start, end = passive.span("subject")
        subject_occurrences = [item for item in occurrences if start <= item[0] < end]
        if subject_occurrences:
            return (min(subject_occurrences, key=lambda item: (item[0], -(item[1] - item[0])))[2],)
    # Main-clause target markers are stronger than later anchor objects such
    # as "between two buildings" or "adjacent to the road".
    target_markers = tuple(
        re.finditer(
            r"(?:occurred to|happened to|regarding|concerning|observed in|seen regarding|involving)",
            normalized,
        )
    )
    if target_markers:
        marker_end = target_markers[-1].end()
        following = [item for item in occurrences if marker_end <= item[0] <= marker_end + 140]
        if following:
            return (min(following, key=lambda item: (item[0], -(item[1] - item[0])))[2],)
    # Locative main clauses can name the target area before introducing an
    # anchor ("in the wooded area north of the road"). Do not select an object
    # that appears only after an anchor preposition.
    locative_markers = tuple(re.finditer(r"(?:occurred|happened)\s+(?:in|at|on)\b", normalized))
    if locative_markers:
        marker_end = locative_markers[-1].end()
        following = sorted(
            (item for item in occurrences if marker_end <= item[0] <= marker_end + 140),
            key=lambda item: (item[0], -(item[1] - item[0])),
        )
        if following:
            prefix = normalized[marker_end : following[0][0]]
            if not re.search(
                r"\b(?:north|south|east|west)\s+of\b|\badjacent\s+to\b|\bnear\b|\bbetween\b|\bwhere\b",
                prefix,
            ):
                return (following[0][2],)
    event_positions = [
        match.start()
        for match in re.finditer(
            r"construct|built|establish|install|demolish|destroy|remove|clear|expand|enlarg|extend|widen|narrow|pav|plant|grow|appear|fill|paint|increase|decrease",
            normalized,
        )
    ]
    if event_positions:
        return (
            min(
                occurrences,
                key=lambda item: min(abs(item[0] - event) for event in event_positions),
            )[2],
        )
    return (max(occurrences, key=lambda item: item[0])[2],)


def _infer_ordinal(question: str) -> str | None:
    normalized = question.lower()
    for ordinal, phrases in (
        ("leftmost", ("leftmost", "far left")),
        ("rightmost", ("rightmost", "far right")),
        ("topmost", ("topmost", "furthest north")),
        ("bottommost", ("bottommost", "furthest south")),
        ("largest", ("largest", "biggest")),
        ("smallest", ("smallest",)),
    ):
        if any(phrase in normalized for phrase in phrases):
            return ordinal
    return None


def _infer_hypothesis_events(question: str, label: str | None) -> tuple[str, ...]:
    normalized = question.lower()
    if any(phrase in normalized for phrase in ("remained unchanged", "no change occurred")):
        return ("unchanged",)
    if _contains_any(normalized, _GLOBAL_TERMS) or "any change" in normalized:
        return tuple(sorted(EVENT_VOCAB - {"unchanged"}))
    rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("demolish", "torn down", "destroy"), ("demolished", "destroyed", "removed")),
        (("removed", "disappear", "cleared"), ("removed", "cleared", "vegetation_loss", "water_decrease", "object_removed")),
        (("expanded", "enlarged", "addition"), ("expanded", "enlarged", "grown", "water_increase")),
        (("extended",), ("extended", "expanded")),
        (("widened",), ("widened", "expanded")),
        (("narrowed",), ("narrowed",)),
        (("paved", "concrete driveway", "asphalt"), ("paved", "constructed", "developed")),
        (("planted",), ("planted", "appeared", "vegetation_increase")),
        (("grown", "larger"), ("grown", "vegetation_increase", "expanded")),
        (("roof color", "repaint"), ("roof_color_changed", "repainted")),
        (("constructed", "newly built", "new building", "new house"), ("constructed", "appeared", "developed")),
        (("appeared", "new tree", "new pool"), ("appeared", "planted", "object_added", "constructed")),
    )
    for phrases, events in rules:
        if any(phrase in normalized for phrase in phrases):
            return events
    if label == "building":
        return ("constructed", "demolished", "expanded", "roof_color_changed", "local_change")
    if label == "vegetation":
        return ("planted", "grown", "vegetation_increase", "vegetation_loss", "local_change")
    if label == "water":
        return ("appeared", "water_increase", "water_decrease", "local_change")
    return tuple(sorted(EVENT_VOCAB - {"unchanged"}))


class RuleBasedTextAgent:
    """Deterministic stand-in for a text LLM tool-use controller."""

    controller_name = "rule-based-text-llm-simulator-v1"

    def plan(self, agent_input: Mapping[str, Any]) -> TextAgentPlan:
        question = str(agent_input["question"])
        options = agent_input.get("options", {})
        combined = " ".join([question, *map(str, options.values())]).lower()
        normalized_question = question.lower()
        global_question = _contains_any(normalized_question, _GLOBAL_TERMS)
        labels = _infer_label(question, global_question)
        notes: list[str] = []
        temporal_between = re.search(
            r"between[^?.]{0,40}\b(?:images|captures|frames|photographs?|photos?|pictures?|"
            r"snapshots?|time periods|temporal states)\b",
            normalized_question,
        )
        if "between" in normalized_question and temporal_between is None:
            notes.append("ternary_between_relation_not_represented")
        if re.search(
            r"\b(second|third|fourth)\b(?!\s+(?:image|capture|frame|photograph|photo|"
            r"picture|snapshot|time|temporal))",
            normalized_question,
        ):
            notes.append("general_ordinal_not_represented")
        hypothesis = _infer_hypothesis_events(question, labels[0] if labels else None)
        return TextAgentPlan(
            labels=labels,
            region=_infer_region(question),
            ordinal=_infer_ordinal(question),
            global_question=global_question,
            use_linear_parser=_contains_any(combined, _LINEAR_TERMS),
            use_paved_parser=_contains_any(combined, _PAVED_TERMS),
            use_small_object_parser=_contains_any(combined, _SMALL_OBJECT_TERMS),
            hypothesis_events=hypothesis,
            planning_notes=tuple(notes),
        )

    def run(
        self,
        executor: ToolExecutor,
        *,
        img_1: str,
        img_2: str,
        agent_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute one task using only text plus structured observations."""

        plan = self.plan(agent_input)
        tool_gaps: list[str] = []
        backend_gaps: list[str] = []
        warnings: list[str] = list(plan.planning_notes)
        artifacts: dict[str, str] = {}

        loaded = executor.call("load_rgb_pair", {"img_1": img_1, "img_2": img_2})
        pair_id = loaded.artifacts["rgb_pair_id"]
        artifacts.update(loaded.artifacts)
        inspected = executor.call("inspect_rgb_pair", {"rgb_pair_id": pair_id})
        artifacts.update(inspected.artifacts)
        if not inspected.observation["valid"]:
            raise RuntimeError("input pair failed RGB inspection")

        alignment = executor.call("check_pixel_alignment", {"rgb_pair_id": pair_id})
        artifacts.update(alignment.artifacts)
        active_pair_id = pair_id
        if not alignment.observation["passed"]:
            if alignment.observation["correctable"]:
                corrected = executor.call(
                    "correct_pixel_alignment",
                    {
                        "rgb_pair_id": pair_id,
                        "alignment_report_id": alignment.artifacts["alignment_report_id"],
                    },
                )
                active_pair_id = corrected.artifacts["aligned_rgb_pair_id"]
                artifacts.update(corrected.artifacts)
            else:
                if alignment.observation["reliable_correction"]:
                    tool_gaps.append("robust_large_alignment_correction")
                else:
                    backend_gaps.append("stable_tie_point_alignment_backend")
                    warnings.append("alignment_inconclusive_no_stable_tie_points")
                warnings.append("continued_with_uncorrected_pair")

        change = executor.call(
            "detect_multiscale_change",
            {"rgb_pair_id": active_pair_id, "threshold": 0.12, "min_region_area": 16},
        )
        artifacts.update(change.artifacts)
        if not change.observation["backend"]["validated_for_rsrcc"]:
            backend_gaps.append("rsrcc_validated_change_backend")

        before = executor.call(
            "parse_scene_entities", {"rgb_pair_id": active_pair_id, "time": "before"}
        )
        after = executor.call(
            "parse_scene_entities", {"rgb_pair_id": active_pair_id, "time": "after"}
        )
        artifacts.update(before.artifacts)
        # Preserve time-specific aliases because output artifact names collide.
        artifacts.update({f"before_{key}": value for key, value in before.artifacts.items()})
        artifacts.update({f"after_{key}": value for key, value in after.artifacts.items()})
        if not after.observation["backend"]["validated_for_rsrcc"]:
            backend_gaps.append("rsrcc_validated_entity_backend")

        linear_ids: list[str] = []
        small_ids: list[str] = []
        if plan.use_linear_parser:
            linear = executor.call(
                "extract_linear_infrastructure",
                {"entity_instances_id": after.artifacts["entity_instances_id"]},
            )
            artifacts.update(linear.artifacts)
            linear_ids.append(linear.artifacts["linear_feature_layer_id"])
            if linear.observation["feature_count"] == 0:
                backend_gaps.append("rsrcc_linear_infrastructure_backend")
        if plan.use_paved_parser:
            paved = executor.call(
                "parse_paved_and_parking_areas",
                {"entity_instances_id": after.artifacts["entity_instances_id"]},
            )
            artifacts.update(paved.artifacts)
            if paved.observation["area_count"] == 0:
                backend_gaps.append("rsrcc_paved_parking_backend")
        if plan.use_small_object_parser:
            small = executor.call(
                "detect_small_aux_objects",
                {"entity_instances_id": after.artifacts["entity_instances_id"]},
            )
            artifacts.update(small.artifacts)
            small_ids.append(small.artifacts["small_object_layer_id"])
            if not small.observation["counts"]:
                backend_gaps.append("rsrcc_small_object_backend")

        matching = executor.call(
            "match_entities_across_time",
            {
                "before_instances_id": before.artifacts["entity_instances_id"],
                "after_instances_id": after.artifacts["entity_instances_id"],
            },
        )
        artifacts.update(matching.artifacts)
        attributes = executor.call(
            "compare_entity_attributes",
            {
                "temporal_entity_matches_id": matching.artifacts[
                    "temporal_entity_matches_id"
                ],
                "rgb_pair_id": active_pair_id,
            },
        )
        artifacts.update(attributes.artifacts)

        graph = executor.call(
            "build_scene_graph",
            {
                "entity_instances_ids": [after.artifacts["entity_instances_id"]],
                "linear_feature_layer_ids": linear_ids,
                "small_object_layer_ids": small_ids,
            },
        )
        artifacts.update(graph.artifacts)
        query = executor.call(
            "resolve_spatial_query",
            {
                "query": {
                    "labels": list(plan.labels),
                    "region": plan.region,
                    "ordinal": plan.ordinal,
                },
                "scene_graph_id": graph.artifacts["scene_graph_id"],
            },
        )
        artifacts.update(query.artifacts)
        if plan.labels and not query.observation["target_ids"]:
            backend_gaps.append(f"grounding_for_{plan.labels[0]}")

        global_report = None
        if plan.global_question:
            global_report = executor.call(
                "assess_global_scene_change",
                {"change_map_id": change.artifacts["change_map_id"]},
            )
            artifacts.update(global_report.artifacts)

        event_arguments: dict[str, Any] = {
            "query_roi_id": query.artifacts["query_roi_id"],
            "temporal_entity_matches_id": matching.artifacts["temporal_entity_matches_id"],
            "attribute_delta_table_id": attributes.artifacts["attribute_delta_table_id"],
            "change_map_id": change.artifacts["change_map_id"],
        }
        if global_report is not None:
            event_arguments["global_change_report_id"] = global_report.artifacts[
                "global_change_report_id"
            ]
        event = executor.call("infer_change_event", event_arguments)
        artifacts.update(event.artifacts)
        statistics = executor.call(
            "summarize_roi_change",
            {
                "query_roi_id": query.artifacts["query_roi_id"],
                "change_map_id": change.artifacts["change_map_id"],
                "before_instances_id": before.artifacts["entity_instances_id"],
                "after_instances_id": after.artifacts["entity_instances_id"],
            },
        )
        artifacts.update(statistics.artifacts)

        answer_format = str(agent_input["answer_format"])
        option_scores = None
        if answer_format == "multiple_choice":
            option_scores = executor.call(
                "score_answer_options",
                {
                    "options": dict(agent_input["options"]),
                    "event_evidence_id": event.artifacts["event_evidence_id"],
                },
            )
            artifacts.update(option_scores.artifacts)
            scores = option_scores.observation["scores"]
            details = option_scores.observation["details"]
            ranked_scores = sorted(scores.values(), reverse=True)
            recognized_options = sum(
                bool(item["support"] or item["conflicts"]) for item in details.values()
            )
            if (
                len(set(scores.values())) == 1
                or max(scores.values()) <= 0
                or recognized_options < 2
                or ranked_scores[0] - ranked_scores[1] < 0.5
            ):
                tool_gaps.append("semantic_option_constraint_parser")
            candidate_answer = str(option_scores.observation["best_option"])
        elif answer_format == "binary":
            observed_events = {item["event"] for item in event.observation["events"]}
            candidate_answer = (
                "yes" if observed_events.intersection(plan.hypothesis_events) else "no"
            )
        else:
            raise ValueError(f"unsupported answer format: {answer_format}")

        render = executor.call(
            "render_evidence",
            {
                "rgb_pair_id": active_pair_id,
                "change_map_id": change.artifacts["change_map_id"],
                "query_roi_id": query.artifacts["query_roi_id"],
                "entity_instances_id": after.artifacts["entity_instances_id"],
                "change_statistics_id": statistics.artifacts["change_statistics_id"],
            },
        )
        artifacts.update(render.artifacts)
        verification_arguments: dict[str, Any] = {
            "candidate_answer": candidate_answer,
            "answer_format": answer_format,
            "event_evidence_id": event.artifacts["event_evidence_id"],
            "query_roi_id": query.artifacts["query_roi_id"],
        }
        if answer_format == "binary":
            verification_arguments["hypothesis_events"] = list(plan.hypothesis_events)
        else:
            verification_arguments["option_scores_id"] = option_scores.artifacts[
                "option_scores_id"
            ]
        verification = executor.call("verify_evidence_consistency", verification_arguments)
        artifacts.update(verification.artifacts)
        schema = executor.call(
            "validate_output_schema",
            {"candidate_answer": candidate_answer, "output_schema": answer_format},
        )
        artifacts.update(schema.artifacts)

        if "ternary_between_relation_not_represented" in plan.planning_notes:
            tool_gaps.append("between_relation_spatial_query")
        if "general_ordinal_not_represented" in plan.planning_notes:
            tool_gaps.append("general_ordinal_spatial_query")
        return {
            "controller": self.controller_name,
            "agent_input": dict(agent_input),
            "plan": asdict(plan),
            "final_answer": candidate_answer,
            "verification": dict(verification.observation),
            "schema_validation": dict(schema.observation),
            "tool_gaps": sorted(set(tool_gaps)),
            "backend_gaps": sorted(set(backend_gaps)),
            "warnings": sorted(set(warnings)),
            "artifacts": artifacts,
            "evidence_render_id": render.artifacts["evidence_render_id"],
        }
