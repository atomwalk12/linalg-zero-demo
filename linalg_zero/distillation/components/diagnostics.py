from __future__ import annotations

from typing import Any

from linalg_zero.distillation.components.models import DIAG_PREFIX, ModelType
from linalg_zero.grpo.verifiers.xml_parser import XMLParser


class Diagnostics:
    def __init__(self, model_type: ModelType) -> None:
        self.config = model_type.get_model_parameters()
        self.diagnostics: list[str] = []
        self.append_assistant = self.config.append_policy()

    # -------- Hint identification / filtering --------
    def is_diagnostic_user_message(self, msg: dict[str, Any]) -> bool:
        return msg.get("role") == "user" and str(msg.get("content", "")).lstrip().startswith(f"{DIAG_PREFIX} ")

    def remove_hint_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove diagnostic user hints and their preceding malformed assistant messages.

        Keeps the first user query and drops any subsequent diagnostic user messages
        along with the malformed assistant messages that triggered them.
        Returns the cleaned message list.
        """
        user_idx = next((i for i, msg in enumerate(messages) if msg.get("role") == "user"), None)
        assert user_idx in (0, 1), "User message not found"

        indices_to_skip = set()
        for i, msg in enumerate(messages):
            if self.is_diagnostic_user_message(msg) and i != user_idx:
                indices_to_skip.add(i)
                if self.append_assistant:
                    assert messages[i - 1]["role"] == "assistant", "Preceding message must be assistant"
                    indices_to_skip.add(i - 1)

        cleaned: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i not in indices_to_skip:
                cleaned.append(msg)

        return cleaned

    # -------- Hint creation / application --------
    def build_hint(self, reason: str) -> str:
        return (
            f"{DIAG_PREFIX} Format error: {reason}.\n"
            "Required format: <think>reasoning</think>, then either\n"
            '<tool_call>{"name":"<function_name>","arguments":{"<param>":"<value>"}}</tool_call>\n'
            "or <answer>result</answer>."
        )

    def analyze_and_build_hint(
        self,
        *,
        context: list[dict[str, Any]],
        message: str,
        tool_names: list[str],
    ) -> str | None:
        """Return a conservative hint only for provably true issues.

        Uses XMLParser structural checks and a small, safe mapping.
        """
        parser = XMLParser()
        seeded = parser.ensure_think_prefix(message) or message
        analysis = parser.analyze_message_in_context(context, message=seeded, tool_names=tool_names)

        reason = parser.get_analysis_failure_reason(analysis, tool_names=tool_names)
        if not reason:
            return None
        return self.build_hint(reason)

    def apply_hint(self, conversation: list[dict[str, Any]], hint: str, *, max_hints: int | None = 1) -> None:
        # Remove older diagnostic user messages if we enforce a cap
        if max_hints is not None and max_hints >= 0:
            existing = [m for m in conversation if self.is_diagnostic_user_message(m)]

            # Keep only the last (max_hints - 1) existing to make room for the new one.
            # Avoid Python's [-0:] == [:] effect when max_hints == 1.
            keep_count = max_hints - 1 if max_hints > 0 else 0
            to_keep = {id(m) for m in existing[-keep_count:]} if keep_count > 0 else set()

            # Remove diagnostic messages and their accompanying assistant messages
            indices_to_remove = set()
            for i, m in enumerate(conversation):
                if self.is_diagnostic_user_message(m) and id(m) not in to_keep:
                    indices_to_remove.add(i)
                    # Also remove the preceding malformed assistant message if it exists
                    if self.append_assistant:
                        assert conversation[i - 1]["role"] == "assistant"
                        indices_to_remove.add(i - 1)

            conversation[:] = [m for i, m in enumerate(conversation) if i not in indices_to_remove]
        else:
            # If cap is None or negative, remove none (unbounded)
            pass
        conversation.append({"role": "user", "content": hint})
