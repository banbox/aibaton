from typing import Any, Dict, Optional

from .utils import now_ms


def normalize_event(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    event_type = raw.get("type") or raw.get("event") or "message"
    return {
        "type": event_type,
        "ts": raw.get("ts") or now_ms(),
        "payload": raw,
        "source": source,
    }


def _extract_content_text(content: Any) -> Optional[str]:
    if not isinstance(content, list):
        return None
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        sub_content = item.get("content")
        if isinstance(sub_content, str):
            parts.append(sub_content)
    return "".join(parts) if parts else None


def extract_text(raw: Dict[str, Any]) -> Optional[str]:
    # Codex item events (agent_message/reasoning) nest text under "item".
    item = raw.get("item")
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type in ("agent_message", "assistant_message", "message"):
            text = item.get("text")
            if isinstance(text, str):
                return text
            extracted = _extract_content_text(item.get("content"))
            if extracted:
                return extracted

    # Common shapes across providers
    if "text" in raw and isinstance(raw["text"], str):
        return raw["text"]
    if "message" in raw and isinstance(raw["message"], str):
        return raw["message"]
    if "content" in raw:
        if isinstance(raw["content"], str):
            return raw["content"]
        extracted = _extract_content_text(raw["content"])
        if extracted:
            return extracted

    if isinstance(raw.get("message"), dict):
        msg = raw.get("message")
        if isinstance(msg.get("text"), str):
            return msg.get("text")
        extracted = _extract_content_text(msg.get("content"))
        if extracted:
            return extracted

    # Some providers (including codex) stream deltas with nested "delta" objects.
    if isinstance(raw.get("delta"), dict) and isinstance(raw["delta"].get("text"), str):
        return raw["delta"].get("text")

    # Anthropic stream-json shapes
    if raw.get("type") == "content_block_delta":
        delta = raw.get("delta", {})
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta.get("text")

    # Some providers nest text in "delta"
    if isinstance(raw.get("delta"), str):
        return raw.get("delta")
    if isinstance(raw.get("text_delta"), str):
        return raw.get("text_delta")

    response = raw.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                extracted = _extract_content_text(item.get("content"))
                if extracted:
                    parts.append(extracted)
            if parts:
                return "".join(parts)

    return None
