import unittest

from aibaton.runner import AgentRes


class TestSelect(unittest.TestCase):
    def test_select(self):
        res = AgentRes(
            text="before <option>abc</option> after",
            events=[],
            status="success",
            usage=None,
            artifacts=None,
            provider="codex",
            model=None,
            elapsed_ms=1,
        )
        self.assertEqual(res.select(), "abc")

    def test_select_missing(self):
        res = AgentRes(
            text="no option",
            events=[],
            status="success",
            usage=None,
            artifacts=None,
            provider="codex",
            model=None,
            elapsed_ms=1,
        )
        self.assertEqual(res.select(), "")


if __name__ == "__main__":
    unittest.main()
