from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import unittest

from teo_agent.rsrcc import parse_rsrcc_annotation


class RsrccDatasetContractTest(unittest.TestCase):
    def test_all_local_test_annotations_parse_without_gold_leakage(self):
        metadata = Path(__file__).parents[1] / "RSRCC" / "test" / "metadata.csv"
        if not metadata.is_file():
            self.skipTest("local RSRCC test metadata is not installed")
        formats: Counter[str] = Counter()
        with metadata.open(newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = parse_rsrcc_annotation(row["text"])
                formats[parsed.answer_format] += 1
                agent_view = parsed.agent_view()
                self.assertNotIn("gold_answer", agent_view)
                self.assertNotIn("**Answer", str(agent_view))
        self.assertEqual(sum(formats.values()), 21_999)
        self.assertEqual(set(formats), {"binary", "multiple_choice"})


if __name__ == "__main__":
    unittest.main()
