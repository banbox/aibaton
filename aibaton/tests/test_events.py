import unittest

from aibaton.events import extract_text


class TestEvents(unittest.TestCase):
    def test_extract_text_simple(self):
        self.assertEqual(extract_text({"text": "hello"}), "hello")
        self.assertEqual(extract_text({"message": "hi"}), "hi")
        self.assertEqual(extract_text({"content": "yo"}), "yo")

    def test_extract_text_anthropic_delta(self):
        raw = {"type": "content_block_delta", "delta": {"text": "x"}}
        self.assertEqual(extract_text(raw), "x")

    def test_extract_text_codex_item(self):
        raw = {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "agent_message", "text": "hello"},
        }
        self.assertEqual(extract_text(raw), "hello")

    def test_extract_text_codex_reasoning_ignored(self):
        raw = {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "reasoning", "text": "secret"},
        }
        self.assertIsNone(extract_text(raw))


if __name__ == "__main__":
    unittest.main()
