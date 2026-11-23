"""Stable visible-message IDs and safe removal of text from typed agent history."""

from copy import deepcopy
from dataclasses import replace
from html import escape
import json
from uuid import uuid4

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart


def message_ref(kind):
    return {"id": uuid4().hex, "kind": kind, "message_index": None, "part_index": None}


def existing_refs(transcript, session):
    if len(session.ui_messages) == len(transcript):
        return deepcopy(session.ui_messages)
    return [message_ref("user" if row["role"] == "user" else "assistant" if isinstance(row["content"], str) else "media")
            for row in transcript]


def bind_text_refs(refs, user_ref, answer_ref, previous_count, session, prompt):
    """Keep protocol indices stable: deletions replace text, never remove tool pairs."""
    for index in range(previous_count, len(session.messages)):
        message = session.messages[index]
        if isinstance(message, ModelRequest):
            for part_index, part in enumerate(message.parts):
                if isinstance(part, UserPromptPart) and part.content == prompt:
                    user_ref.update(message_index=index, part_index=part_index)
        elif isinstance(message, ModelResponse) and not any(isinstance(part, ToolCallPart) for part in message.parts):
            if any(isinstance(part, TextPart) for part in message.parts):
                answer_ref.update(message_index=index)
    return replace(session, ui_messages=refs)


def action_manifest(transcript, session):
    refs = session.ui_messages if len(session.ui_messages) == len(transcript or []) else []
    ids = [ref["id"] for ref in refs]
    return '<div data-message-ids="' + escape(json.dumps(ids), quote=True) + '"></div>'


def remove_message(payload, transcript, session):
    """Delete exactly one bubble, preserving later messages and scientific evidence."""
    try:
        request = json.loads(payload)
        identifier = request["message_id"]
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(identifier, str) or len(session.ui_messages) != len(transcript or []):
        return None
    index = next((index for index, ref in enumerate(session.ui_messages) if ref["id"] == identifier), None)
    if index is None:
        return None  # A stale click must not delete the next message at the same position.
    updated = deepcopy(session)
    ref = updated.ui_messages.pop(index)
    messages = list(transcript)
    messages.pop(index)
    protocol_index = ref.get("message_index")
    if protocol_index is not None and 0 <= protocol_index < len(updated.messages):
        message = updated.messages[protocol_index]
        if ref["kind"] == "user" and isinstance(message, ModelRequest):
            part_index = ref.get("part_index")
            if part_index is not None and 0 <= part_index < len(message.parts) and isinstance(message.parts[part_index], UserPromptPart):
                message.parts[part_index] = replace(message.parts[part_index], content="[User message removed from this chat.]")
        elif ref["kind"] == "assistant" and isinstance(message, ModelResponse):
            message.parts = [part for part in message.parts if isinstance(part, ToolCallPart)] + [TextPart("[Assistant message removed from this chat.]")]
    return messages, updated
