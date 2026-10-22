from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from teo_agent.rsrcc import (
    ArtifactStore,
    PaletteEntityBackend,
    QuerySpec,
    RuleBasedTextAgent,
    RsrccTools,
    TOOL_NAMES,
    ToolExecutor,
    parse_rsrcc_annotation,
)


PALETTE = {
    "building": (200, 200, 200),
    "road": (100, 100, 100),
    "parking_lot": (140, 140, 140),
    "vehicle": (255, 0, 0),
    "vegetation": (0, 180, 0),
    "water": (0, 0, 220),
}


class RsrccToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        after = np.zeros_like(before)
        before[20:32, 8:20] = PALETTE["building"]
        after[18:34, 8:26] = PALETTE["building"]
        before[42:48, 2:62] = PALETTE["road"]
        after[42:48, 2:62] = PALETTE["road"]
        before[4:16, 36:58] = PALETTE["parking_lot"]
        after[4:16, 36:58] = PALETTE["parking_lot"]
        before[7:11, 40:44] = PALETTE["vehicle"]
        after[7:11, 40:44] = PALETTE["vehicle"]
        after[7:11, 49:53] = PALETTE["vehicle"]
        before[50:58, 6:14] = PALETTE["vegetation"]
        after[50:60, 6:16] = PALETTE["vegetation"]
        after[50:56, 22:28] = PALETTE["vegetation"]
        before[3:11, 3:11] = PALETTE["water"]
        after[3:11, 3:11] = PALETTE["water"]
        self.before_path = root / "before.png"
        self.after_path = root / "after.png"
        Image.fromarray(before).save(self.before_path)
        Image.fromarray(after).save(self.after_path)

        self.store = ArtifactStore()
        self.tools = RsrccTools(
            self.store,
            entity_backend=PaletteEntityBackend(PALETTE, tolerance=0, min_area=4),
        )
        self.pair = self.tools.load_rgb_pair(self.before_path, self.after_path).artifacts[
            "rgb_pair_id"
        ]
        self.change = self.tools.detect_multiscale_change(
            self.pair, threshold=0.05, min_region_area=2
        )
        self.before_scene = self.tools.parse_scene_entities(self.pair, time="before")
        self.after_scene = self.tools.parse_scene_entities(self.pair, time="after")

    def _matching(self):
        return self.tools.match_entities_across_time(
            self.before_scene.artifacts["entity_instances_id"],
            self.after_scene.artifacts["entity_instances_id"],
            max_centroid_distance_px=12,
        )

    def _building_roi(self):
        graph = self.tools.build_scene_graph(
            [self.after_scene.artifacts["entity_instances_id"]]
        )
        return self.tools.resolve_spatial_query(
            QuerySpec(labels=("building",), region="west"),
            graph.artifacts["scene_graph_id"],
        )

    def _event(self):
        matching = self._matching()
        attributes = self.tools.compare_entity_attributes(
            matching.artifacts["temporal_entity_matches_id"], self.pair
        )
        roi = self._building_roi()
        return self.tools.infer_change_event(
            roi.artifacts["query_roi_id"],
            matching.artifacts["temporal_entity_matches_id"],
            attribute_delta_table_id=attributes.artifacts["attribute_delta_table_id"],
            change_map_id=self.change.artifacts["change_map_id"],
        ), roi

    def test_load_rgb_pair(self):
        artifact = self.store.get(self.pair, expected_kind="RGBPair")
        self.assertEqual(artifact.data["before"].shape, (64, 64, 3))
        self.assertFalse(artifact.data["before"].flags.writeable)

    def test_inspect_rgb_pair(self):
        response = self.tools.inspect_rgb_pair(self.pair)
        self.assertTrue(response.observation["valid"])
        self.assertIn("metadata_report_id", response.artifacts)

    def test_check_pixel_alignment(self):
        rng = np.random.default_rng(7)
        first = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        second[3:, 4:] = first[:-3, :-4]
        root = Path(self.temp_dir.name)
        first_path, second_path = root / "align1.png", root / "align2.png"
        Image.fromarray(first).save(first_path)
        Image.fromarray(second).save(second_path)
        pair = self.tools.load_rgb_pair(first_path, second_path).artifacts["rgb_pair_id"]
        response = self.tools.check_pixel_alignment(pair, tolerance_px=1)
        shift = self.store.get(response.artifacts["alignment_report_id"]).data[
            "shift_to_apply_px"
        ]
        self.assertEqual((shift["dx"], shift["dy"]), (-4, -3))
        self.assertIn("scale_to_apply", self.store.get(response.artifacts["alignment_report_id"]).data)
        self.assertFalse(response.observation["passed"])
        self.assertTrue(response.observation["reliable_correction"])
        self.assertTrue(response.observation["correctable"])

    def test_alignment_rejects_unrelated_scene_as_correction(self):
        rng = np.random.default_rng(71)
        first = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        second = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        root = Path(self.temp_dir.name)
        first_path, second_path = root / "unrelated1.png", root / "unrelated2.png"
        Image.fromarray(first).save(first_path)
        Image.fromarray(second).save(second_path)
        pair = self.tools.load_rgb_pair(first_path, second_path).artifacts["rgb_pair_id"]
        response = self.tools.check_pixel_alignment(pair)
        self.assertFalse(response.observation["reliable_correction"])
        self.assertFalse(response.observation["correctable"])

    def test_correct_pixel_alignment(self):
        rng = np.random.default_rng(9)
        first = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        second[2:, 3:] = first[:-2, :-3]
        root = Path(self.temp_dir.name)
        first_path, second_path = root / "correct1.png", root / "correct2.png"
        Image.fromarray(first).save(first_path)
        Image.fromarray(second).save(second_path)
        pair = self.tools.load_rgb_pair(first_path, second_path).artifacts["rgb_pair_id"]
        report = self.tools.check_pixel_alignment(pair).artifacts["alignment_report_id"]
        response = self.tools.correct_pixel_alignment(pair, report)
        aligned = self.store.get(response.artifacts["aligned_rgb_pair_id"]).data
        before_error = np.abs(first[3:-3, 4:-4].astype(int) - second[3:-3, 4:-4]).mean()
        after_error = np.abs(
            aligned["before"][3:-3, 4:-4].astype(int)
            - aligned["after"][3:-3, 4:-4].astype(int)
        ).mean()
        self.assertLess(after_error, before_error)

    def test_detect_multiscale_change(self):
        self.assertGreater(self.change.observation["changed_pixel_ratio"], 0)
        self.assertGreater(self.change.observation["region_count"], 0)
        self.assertFalse(self.change.observation["backend"]["validated_for_rsrcc"])

    def test_parse_scene_entities(self):
        counts = self.after_scene.observation["counts"]
        self.assertEqual(counts["building"], 1)
        self.assertEqual(counts["vehicle"], 2)
        self.assertFalse(self.after_scene.observation["backend"]["validated_for_rsrcc"])

    def test_extract_linear_infrastructure(self):
        response = self.tools.extract_linear_infrastructure(
            self.after_scene.artifacts["entity_instances_id"]
        )
        self.assertEqual(response.observation["feature_count"], 1)
        self.assertGreater(response.observation["total_length_px"], 0)

    def test_parse_paved_and_parking_areas(self):
        response = self.tools.parse_paved_and_parking_areas(
            self.after_scene.artifacts["entity_instances_id"]
        )
        self.assertEqual(response.observation["area_count"], 1)
        layout = self.store.get(response.artifacts["parking_layout_id"]).data
        self.assertIsNone(layout["estimated_stall_count"])

    def test_detect_small_aux_objects(self):
        response = self.tools.detect_small_aux_objects(
            self.after_scene.artifacts["entity_instances_id"]
        )
        self.assertEqual(response.observation["counts"], {"vehicle": 2})

    def test_match_entities_across_time(self):
        response = self._matching()
        self.assertGreaterEqual(response.observation["matched_count"], 5)
        self.assertGreaterEqual(response.observation["added_count"], 2)

    def test_compare_entity_attributes(self):
        matching = self._matching()
        response = self.tools.compare_entity_attributes(
            matching.artifacts["temporal_entity_matches_id"], self.pair
        )
        table = self.store.get(response.artifacts["attribute_delta_table_id"]).data
        building = next(row for row in table["deltas"] if row["label"] == "building")
        self.assertGreater(building["area_ratio"], 1.5)

    def test_build_scene_graph(self):
        response = self.tools.build_scene_graph(
            [self.after_scene.artifacts["entity_instances_id"]]
        )
        graph = self.store.get(response.artifacts["scene_graph_id"]).data
        self.assertGreater(len(graph["nodes"]), 5)
        self.assertIn("north_of", {edge["relation"] for edge in graph["edges"]})

    def test_resolve_spatial_query(self):
        response = self._building_roi()
        self.assertEqual(len(response.observation["target_ids"]), 1)
        self.assertIsNotNone(response.observation["bbox"])

    def test_infer_change_event(self):
        response, _ = self._event()
        self.assertEqual(response.observation["primary_event"], "expanded")

    def test_summarize_roi_change(self):
        _, roi = self._event()
        response = self.tools.summarize_roi_change(
            roi.artifacts["query_roi_id"],
            self.change.artifacts["change_map_id"],
            before_instances_id=self.before_scene.artifacts["entity_instances_id"],
            after_instances_id=self.after_scene.artifacts["entity_instances_id"],
        )
        self.assertGreater(response.observation["changed_pixel_count"], 0)
        self.assertNotIn("square_metres", response.observation)

    def test_score_answer_options(self):
        event, _ = self._event()
        response = self.tools.score_answer_options(
            {
                "A": "The building was demolished.",
                "B": "The building was expanded with an addition.",
                "C": "There was no change.",
                "D": "Vegetation was planted.",
            },
            event.artifacts["event_evidence_id"],
        )
        self.assertEqual(response.observation["best_option"], "B")

    def test_assess_global_scene_change(self):
        response = self.tools.assess_global_scene_change(
            self.change.artifacts["change_map_id"], global_min_ratio=0.95
        )
        self.assertEqual(response.observation["state"], "local_change")

    def test_render_evidence(self):
        _, roi = self._event()
        response = self.tools.render_evidence(
            self.pair,
            change_map_id=self.change.artifacts["change_map_id"],
            query_roi_id=roi.artifacts["query_roi_id"],
        )
        artifact = self.store.get(response.artifacts["evidence_render_id"])
        self.assertTrue(artifact.data.startswith(b"\x89PNG"))
        self.assertEqual(response.observation["width"], 192)

    def test_verify_evidence_consistency(self):
        event, roi = self._event()
        response = self.tools.verify_evidence_consistency(
            "yes",
            "binary",
            event.artifacts["event_evidence_id"],
            query_roi_id=roi.artifacts["query_roi_id"],
        )
        self.assertTrue(response.observation["consistent"])

    def test_validate_output_schema(self):
        valid = self.tools.validate_output_schema("A", "multiple_choice")
        invalid = self.tools.validate_output_schema("Yes", "binary")
        self.assertTrue(valid.observation["valid"])
        self.assertFalse(invalid.observation["valid"])

    def test_annotation_parser_hides_gold(self):
        parsed = parse_rsrcc_annotation(
            "**Question:** What changed?\\n\\n**A)** New building.\\n"
            "**B)** No change.\\n**C)** New tree.\\n**D)** New road.\\n\\n**Answer:** A"
        )
        self.assertEqual(parsed.gold_answer, "A")
        self.assertNotIn("gold_answer", parsed.agent_view())
        self.assertNotIn("Answer", str(parsed.agent_view()))

    def test_text_agent_planner_prefers_changed_target_over_anchor(self):
        agent = RuleBasedTextAgent()
        construction = agent.plan(
            {
                "question": (
                    "Has a new house been constructed in the area that was previously "
                    "covered by vegetation?"
                ),
                "options": {},
                "answer_format": "binary",
            }
        )
        between = agent.plan(
            {
                "question": (
                    "What change was observed in the parking lot between the two buildings?"
                ),
                "options": {},
                "answer_format": "binary",
            }
        )
        temporal = agent.plan(
            {
                "question": "What changed between the two images?",
                "options": {},
                "answer_format": "binary",
            }
        )
        passive_tree = agent.plan(
            {
                "question": (
                    "Has the small tree located in the natural terrain been removed to "
                    "allow for the paving of a new cul-de-sac?"
                ),
                "options": {},
                "answer_format": "binary",
            }
        )
        passive_pool = agent.plan(
            {
                "question": (
                    "Has a swimming pool been constructed behind a newly built residence?"
                ),
                "options": {},
                "answer_format": "binary",
            }
        )
        temporal_photographs = agent.plan(
            {
                "question": (
                    "What changed north of the road between the first and second photographs?"
                ),
                "options": {},
                "answer_format": "multiple_choice",
            }
        )
        pool_near_street = agent.plan(
            {
                "question": "Has a new swimming pool been installed beside the street?",
                "options": {},
                "answer_format": "binary",
            }
        )
        self.assertEqual(construction.labels, ("building",))
        self.assertEqual(between.labels, ("parking_lot",))
        self.assertIn("ternary_between_relation_not_represented", between.planning_notes)
        self.assertNotIn("ternary_between_relation_not_represented", temporal.planning_notes)
        self.assertEqual(passive_tree.labels, ("vegetation",))
        self.assertEqual(passive_pool.labels, ("water",))
        self.assertEqual(pool_near_street.labels, ("water",))
        self.assertNotIn(
            "ternary_between_relation_not_represented",
            temporal_photographs.planning_notes,
        )
        self.assertNotIn(
            "general_ordinal_not_represented",
            temporal_photographs.planning_notes,
        )

    def test_executor_allowlist_trace_and_budget(self):
        executor = ToolExecutor(self.tools, max_calls=1)
        response = executor.call("inspect_rgb_pair", {"rgb_pair_id": self.pair})
        self.assertIn("metadata_report_id", response.artifacts)
        self.assertEqual(executor.trace[0].tool, "inspect_rgb_pair")
        self.assertEqual(set(TOOL_NAMES), set(executor.tool_names))
        with self.assertRaises(RuntimeError):
            executor.call("inspect_rgb_pair", {"rgb_pair_id": self.pair})

    def test_executor_end_to_end_evidence_trace(self):
        executor = ToolExecutor(self.tools, max_calls=32)
        inspect = executor.call("inspect_rgb_pair", {"rgb_pair_id": self.pair})
        change = executor.call(
            "detect_multiscale_change",
            {"rgb_pair_id": self.pair, "threshold": 0.05, "min_region_area": 2},
        )
        before = executor.call(
            "parse_scene_entities", {"rgb_pair_id": self.pair, "time": "before"}
        )
        after = executor.call(
            "parse_scene_entities", {"rgb_pair_id": self.pair, "time": "after"}
        )
        linear = executor.call(
            "extract_linear_infrastructure",
            {"scene_layer_id": after.artifacts["scene_layer_id"]},
        )
        small = executor.call(
            "detect_small_aux_objects",
            {"rgb_pair_id": self.pair, "time": "after"},
        )
        matches = executor.call(
            "match_entities_across_time",
            {
                "before_instances_id": before.artifacts["entity_instances_id"],
                "after_instances_id": after.artifacts["entity_instances_id"],
                "max_centroid_distance_px": 12,
            },
        )
        attributes = executor.call(
            "compare_entity_attributes",
            {
                "temporal_entity_matches_id": matches.artifacts[
                    "temporal_entity_matches_id"
                ],
                "rgb_pair_id": self.pair,
            },
        )
        graph = executor.call(
            "build_scene_graph",
            {
                "entity_instances_ids": [after.artifacts["entity_instances_id"]],
                "linear_feature_layer_ids": [linear.artifacts["linear_feature_layer_id"]],
                "small_object_layer_ids": [small.artifacts["small_object_layer_id"]],
            },
        )
        roi = executor.call(
            "resolve_spatial_query",
            {
                "query": {"labels": ["building"], "region": "west"},
                "scene_graph_id": graph.artifacts["scene_graph_id"],
            },
        )
        event = executor.call(
            "infer_change_event",
            {
                "query_roi_id": roi.artifacts["query_roi_id"],
                "temporal_entity_matches_id": matches.artifacts[
                    "temporal_entity_matches_id"
                ],
                "attribute_delta_table_id": attributes.artifacts[
                    "attribute_delta_table_id"
                ],
                "change_map_id": change.artifacts["change_map_id"],
            },
        )
        statistics = executor.call(
            "summarize_roi_change",
            {
                "query_roi_id": roi.artifacts["query_roi_id"],
                "change_map_id": change.artifacts["change_map_id"],
            },
        )
        options = executor.call(
            "score_answer_options",
            {
                "options": {
                    "A": "The building was demolished.",
                    "B": "The building was expanded with an addition.",
                    "C": "No change occurred.",
                    "D": "A tree was planted.",
                },
                "event_evidence_id": event.artifacts["event_evidence_id"],
            },
        )
        render = executor.call(
            "render_evidence",
            {
                "rgb_pair_id": self.pair,
                "change_map_id": change.artifacts["change_map_id"],
                "query_roi_id": roi.artifacts["query_roi_id"],
                "entity_instances_id": after.artifacts["entity_instances_id"],
                "change_statistics_id": statistics.artifacts["change_statistics_id"],
            },
        )
        verify = executor.call(
            "verify_evidence_consistency",
            {
                "candidate_answer": options.observation["best_option"],
                "answer_format": "multiple_choice",
                "event_evidence_id": event.artifacts["event_evidence_id"],
                "option_scores_id": options.artifacts["option_scores_id"],
                "query_roi_id": roi.artifacts["query_roi_id"],
            },
        )
        schema = executor.call(
            "validate_output_schema",
            {
                "candidate_answer": options.observation["best_option"],
                "output_schema": "multiple_choice",
            },
        )
        self.assertTrue(inspect.observation["valid"])
        self.assertTrue(verify.observation["consistent"])
        self.assertTrue(schema.observation["valid"])
        self.assertIn("evidence_render_id", render.artifacts)
        self.assertEqual(len(executor.trace), 16)
        for step in executor.trace:
            for value in step.observation.values():
                self.assertNotIsInstance(value, np.ndarray)

    def test_every_registered_tool_has_a_basic_test(self):
        test_names = {name for name in dir(type(self)) if name.startswith("test_")}
        missing = {name for name in TOOL_NAMES if f"test_{name}" not in test_names}
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()
