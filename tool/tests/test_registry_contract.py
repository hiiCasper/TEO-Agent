from __future__ import annotations

from pathlib import Path
import re
import unittest

from teo_agent.rsrcc import EVENT_VOCAB, RsrccTools, TOOL_NAMES


class RegistryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry_path = Path(__file__).parents[1] / "rsrcc-tool-registry.md"
        cls.registry = cls.registry_path.read_text(encoding="utf-8")

    def test_documented_tools_equal_executable_allowlist(self):
        tool_section = self.registry.split("## 3. 工具注册表", 1)[1].split(
            "## 4. 受控事件词表", 1
        )[0]
        documented = tuple(re.findall(r"^\| `([a-z][a-z0-9_]*)` \|", tool_section, re.MULTILINE))
        self.assertEqual(documented, TOOL_NAMES)
        for name in documented:
            self.assertTrue(callable(getattr(RsrccTools, name, None)), name)

    def test_documented_events_equal_code_vocabulary(self):
        event_section = self.registry.split("## 4. 受控事件词表", 1)[1].split(
            "## 5. 执行闭环", 1
        )[0]
        documented: set[str] = set()
        for line in event_section.splitlines():
            if line.startswith("|"):
                documented.update(re.findall(r"`([a-z][a-z0-9_]*)`", line))
        self.assertEqual(documented, set(EVENT_VOCAB))


if __name__ == "__main__":
    unittest.main()
