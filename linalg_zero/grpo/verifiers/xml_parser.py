import json
import re
from typing import Any

from linalg_zero.distillation.components.models import DIAG_PREFIX


class XMLParser:
    # Checks exact format: <think>...</think> followed by <answer>...</answer>
    think_then_answer_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<answer>\s*([\s\S]*?)\s*<\/answer>$"
    )
    # Checks exact format: <think>...</think> followed by <tool_call>...</tool_call>
    think_then_tool_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<tool_call>\s*([\s\S]*?)\s*<\/tool_call>$"
    )

    def get_messages(self, completion: list[dict], role: str) -> list[dict]:
        """Helper function to extract messages of a specific type from a completion."""
        return [msg for msg in completion if msg["role"] == role]

    def _has_tool_calls(self, context: list[dict]) -> bool:
        """Check if the context contains any tool calls."""
        return len(self.get_messages(context, role="tool")) > 0

    def get_last_message(self, messages: list[dict], role: str) -> str | None:
        role_messages = self.get_messages(messages, role)
        if role_messages:
            result = role_messages[-1]["content"]
            assert isinstance(result, str)
            return result
        return None

    def extract_last_answer(self, message: str) -> str | None:
        """Extract answer content from <answer> tags.

        Primary path: properly closed <answer>...</answer> (last occurrence).
        Fallback: if an opening <answer> exists but closing is missing (e.g.,
        stop sequences), return content from after <answer> to end of message.
        """
        contents = self.extract_tag_contents(message, "answer", last_only=True)
        return contents[0] if contents else None

    def _check_format(self, message: str, regex: str, expected_groups: int) -> bool:
        """Check if message matches the expected format with correct number of groups."""
        last_think_pos = message.rfind("<think>")
        if last_think_pos == -1:
            return False

        # Find the last <think> token in the string, then verify format till the end
        last_think_token = message[last_think_pos:]

        match = re.search(regex, last_think_token, re.DOTALL)

        # We look for a match and assert that the number of matched groups is correct
        return match is not None and len(match.groups()) == expected_groups

    def _is_valid_think_then_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<answer>...</answer>'."""

        return self._check_format(message, self.think_then_answer_regex, expected_groups=2)

    def _is_valid_think_then_tool(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<tool_call>...</tool_call>'."""

        return self._check_format(message, self.think_then_tool_regex, expected_groups=2)

    def _is_valid_think_then_tool_or_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by exactly one of '<tool_call>...</tool_call>' or '<answer>...</answer>'."""

        valid_tool = self._check_format(message, self.think_then_tool_regex, expected_groups=2)
        valid_answer = self._check_format(message, self.think_then_answer_regex, expected_groups=2)
        return valid_tool or valid_answer

    def ensure_think_prefix(self, message: str | None) -> str | None:
        """Ensure the message starts with a single '<think>' prefix without duplicating it."""
        if message is None:
            return None
        if message.lstrip().startswith("<think>"):
            return message
        return "<think>" + message

    def extract_tag_contents(
        self,
        message: str,
        tag: str,
        *,
        last_only: bool = False,
    ) -> list[str]:
        """
        Extract contents enclosed by a specific XML-like tag.

        - If last_only is True, returns at most one element: the content between the
          last occurrence of <tag> and its subsequent </tag>, if properly closed.
        - If last_only is False, returns all non-overlapping, properly closed tag
          contents in document order.
        - Whitespace around extracted content is stripped.
        """
        assert tag in ["tool_call", "answer", "think"]
        if not message:
            return []

        # Lazy search for all tagged blocks in the text
        pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL)
        matches = [m.group(1).strip() for m in pattern.finditer(message)]

        if last_only:
            return matches[-1:] if matches else []

        return matches

    def _extract_last_tool_call(self, message: str) -> str | None:
        """Extract <tool_call>...</tool_call> block contents."""
        contents = self.extract_tag_contents(message, "tool_call", last_only=True)
        return contents[0] if contents else None

    def extract_last_thought(self, message: str) -> str | None:
        """Extract thought content from properly formed <think></think> tags.

        Supports normalization when the message begins with an auto-seeded
        "<think>" and the model also emitted its own "<think>", resulting in
        leading "<think><think>". In such case, we still return the content of
        the last properly closed think block.
        """
        contents = self.extract_tag_contents(message, "think", last_only=True)
        return contents[0] if contents else None

    def analyze_message(
        self,
        message: str,
        *,
        tool_names: list[str] | None = None,
    ) -> dict:
        """
        Parse a single assistant message and return diagnostics + extracted fields.

        Returns a dictionary with keys:
        - has_think, has_tool_call, has_answer: bool
        - valid_format: bool (think then tool_call|answer)
        - is_valid_think_then_answer: bool (think then answer)
        - think_count, tool_call_count, answer_count: int
        - unopened: {think, tool_call, answer}: bool
        - unclosed: {think, tool_call, answer}: bool
        - thought: str | None
        - answer: str | None
        - tool: {
            json_valid: bool,
            name: str | "",
            arguments: dict | {},
            name_known: bool | None,
            }
        """

        diagnostics = XMLDiagnostics(self)

        thought = self.extract_last_thought(message)
        answer = self.extract_last_answer(message)
        tool_block = self._extract_last_tool_call(message)

        result: dict = {}
        result["thought"] = thought
        result["answer"] = answer
        result["has_think"] = thought is not None
        result["has_tool_call"] = tool_block is not None
        result["has_answer"] = answer is not None

        # Counts of properly closed blocks (uniqueness diagnostics)
        result["think_count"] = len(self.extract_tag_contents(message, "think"))
        result["tool_call_count"] = len(self.extract_tag_contents(message, "tool_call"))
        result["answer_count"] = len(self.extract_tag_contents(message, "answer"))

        # Format validity
        result["is_valid_think_then_tool_or_answer"] = self._is_valid_think_then_tool_or_answer(message)
        result["is_valid_think_then_answer"] = self._is_valid_think_then_answer(message)
        result["has_content_outside_tags"] = self.has_content_outside_tags(message)

        # Structural diagnostics
        result["unopened"] = {
            "think": diagnostics._has_unopened_tag(message, "think"),
            "tool_call": diagnostics._has_unopened_tag(message, "tool_call"),
            "answer": diagnostics._has_unopened_tag(message, "answer"),
        }
        result["unclosed"] = {
            "think": diagnostics._has_unclosed_tag(message, "think"),
            "tool_call": diagnostics._has_unclosed_tag(message, "tool_call"),
            "answer": diagnostics._has_unclosed_tag(message, "answer"),
        }
        # Optional checks for stray content/code fences
        # result["stray_content"] = diagnostics._has_stray_content_outside_allowed(message)
        # result["code_fences_in_last_tool"] = diagnostics._has_code_fences_in_last_tool(message)

        # Tool parsing
        tool_info: dict = {"json_valid": False, "name": "", "arguments": {}, "name_known": None}
        if tool_block is not None:
            data = _safe_json_loads(tool_block)
            if (
                isinstance(data, dict)
                and isinstance(data.get("name"), str)
                and isinstance(data.get("arguments"), dict | str)
            ):

                def parse_args() -> tuple[dict | None, bool]:
                    arguments = data["arguments"]
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (ValueError, SyntaxError):
                            return None, False
                        else:
                            return arguments, True
                    else:
                        return arguments, True

                arguments, json_valid = parse_args()
                tool_info["json_valid"] = json_valid
                tool_info["name"] = data["name"]
                tool_info["arguments"] = arguments
                if tool_names is not None:
                    tool_info["name_known"] = data["name"] in tool_names
            else:
                tool_info["json_valid"] = False

        result["tool"] = tool_info

        return result

    def has_content_outside_tags(self, message: str) -> bool:
        """
        Check if there's any non-whitespace content outside of
        <think>...</think>, <tool_call>...</tool_call>, and <answer>...</answer> tags.
        """
        # Remove all allowed tag blocks
        cleaned = message
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<answer>.*?</answer>", "", cleaned, flags=re.DOTALL)

        # Check if there's any non-whitespace content left
        return bool(cleaned.strip())

    def analyze_message_in_context(
        self,
        context: list[dict],
        message: str,
        *,
        tool_names: list[str] | None = None,
    ) -> dict:
        """
        Like analyze_message, but also evaluates conversation-level policy:
        - answer_policy_valid: if an <answer> is present in message, the immediately
        previous message in the conversation MUST be a tool response (a system message
        whose content contains <tool_response> ... </tool_response>).
        Adds fields:
        - answer_allowed: bool (there is a prior adjacent tool_response)
        - answer_policy_valid: bool (no answer, or answer_allowed)
        """
        result = self.analyze_message(message, tool_names=tool_names)
        result["answer_policy_valid"] = result["is_valid_think_then_answer"] and self.is_answer_policy_valid(
            context, message
        )
        return result

    def is_answer_policy_valid(self, context: list[dict], message: str) -> bool:
        answer = self.extract_last_answer(message)
        if not answer:
            return True
        skipped_current_assistant = False
        for prev in reversed(context):
            role = prev["role"]
            content = str(prev["content"])

            # If the current assistant message is already included verbatim in context,
            # skip it once so we only check prior messages for adjacency
            if not skipped_current_assistant and role == "assistant" and content == message:
                skipped_current_assistant = True
                continue
            if role == "user" and content.lstrip().startswith(f"{DIAG_PREFIX} "):
                continue
            return bool(role == "tool")
        return False

    def get_analysis_failure_reason(self, analysis: dict, tool_names: list[str]) -> str:  # noqa: C901
        """
        Get the failure reason for a given analysis. The analysis follows:
        Syntax → Structure → Content → Format → Fallback
        """
        # Uniqueness errors
        if analysis["think_count"] > 1:
            return "multiple <think> blocks (nested/repeated)"

        if analysis["tool_call_count"] > 1:
            return "multiple <tool_call> blocks (nested/repeated)"

        if analysis["answer_count"] > 1:
            return "multiple <answer> blocks (nested/repeated)"

        if not analysis["has_think"] and not analysis["has_tool_call"] and not analysis["has_answer"]:
            return "no <think>/<tool_call>/<answer> blocks"

        # Core requirements
        if analysis["has_tool_call"] and analysis["has_answer"]:
            return "both tool call and answer present"

        if not analysis["has_tool_call"] and not analysis["has_answer"]:
            return "no tool call or answer"

        # Structural errors
        if not analysis["has_think"]:
            return "missing <think>"

        # Validate tool internal structure
        if analysis["has_tool_call"]:
            tool = analysis["tool"]
            if not tool["json_valid"]:
                return "invalid tool JSON"
            if not isinstance(tool["arguments"], dict):
                return "invalid tool arguments"
            name = tool["name"]
            if tool_names and name not in tool_names:
                return "unknown tool name"

        # Check for content outside allowed tags
        if analysis["has_content_outside_tags"]:
            return "content outside allowed tags"

        # Overall format validation
        if not analysis["is_valid_think_then_tool_or_answer"]:
            return "invalid format"

        # Context dependent errors
        if analysis["has_answer"] and not bool(analysis["answer_policy_valid"]):
            return "answer without tool response"

        return "unspecified issue"


class XMLDiagnostics:
    """Diagnostic helpers for the analysis of malformed generations.

    Separated from XMLParser to keep core parsing/validation lean.
    """

    def __init__(self, parser: XMLParser):
        self.parser = parser

    def _has_unclosed_tag(self, message: str, tag: str) -> bool:
        assert tag in ["tool_call", "answer", "think"]
        if not message:
            return False
        open_token = f"<{tag}>"
        close_token = f"</{tag}>"
        if open_token not in message:
            return False
        last_open = message.rfind(open_token)
        after_open = message[last_open + len(open_token) :]
        return close_token not in after_open

    def _has_code_fences_in_last_tool(self, message: str) -> bool:
        block = self.parser.extract_tag_contents(message, "tool_call", last_only=True)
        if not block:
            return False
        return "```" in block[0]

    def _has_stray_content_outside_allowed(self, message: str) -> bool:
        if "<think>" not in message:
            return False
        has_tool = "<tool_call>" in message and "</tool_call>" in message
        has_answer = "<answer>" in message and "</answer>" in message
        if not (has_tool or has_answer):
            return False
        return not (self.parser._is_valid_think_then_tool(message) or self.parser._is_valid_think_then_answer(message))

    def _has_unopened_tag(self, message: str, tag: str) -> bool:
        """Return True if a closing </tag> appears without any prior opening <tag>."""
        assert tag in ["tool_call", "answer", "think"]
        if not message:
            return False
        open_token = f"<{tag}>"
        close_token = f"</{tag}>"
        if close_token not in message:
            return False
        first_close = message.find(close_token)
        first_open = message.find(open_token)
        return first_open == -1 or first_close < first_open


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else None
    except Exception:
        return None
