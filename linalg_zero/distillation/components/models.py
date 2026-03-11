import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from distilabel.errors import DistilabelUserError
from distilabel.typing import ChatType

from linalg_zero.distillation.data import ThoughtSchema
from linalg_zero.shared.system_prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)

if TYPE_CHECKING:
    pass


DIAG_PREFIX = "[diag]"


class ModelType(str, Enum):
    DEFAULT = "default"

    def get_model_parameters(self) -> "ModelParameters":
        return DefaultConfig()


class ModelParameters(ABC):
    @abstractmethod
    def set_recommended_defaults(self, generation_kwargs: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
        """Inject recommended generation defaults for the model.

        deterministic=True should configure sampling deterministically (e.g., temperature=0, top_p=1).
        """
        raise NotImplementedError

    @abstractmethod
    def format_assistant_message(self, message: ThoughtSchema) -> dict[str, Any] | None:
        """Return an OpenAI-compatible assistant message for the given parsed output."""
        raise NotImplementedError

    @abstractmethod
    def append_policy(self) -> bool:
        """Return whether to append the assistant message to the conversation."""
        pass


class DefaultConfig(ModelParameters):
    def set_recommended_defaults(self, generation_kwargs: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
        if deterministic:
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["top_p"] = 0.95
        else:
            # Recommended non-deterministic defaults for high-quality generations
            # Aligns with Qwen best practices while remaining backend-agnostic
            generation_kwargs.setdefault("temperature", 0.6)
            generation_kwargs.setdefault("top_p", 0.95)

            # Some backends (e.g., vLLM) accept additional sampling params
            extra_body = generation_kwargs.setdefault("extra_body", {})
            extra_body.setdefault("top_k", 20)
            extra_body.setdefault("min_p", 0)
        return generation_kwargs

    def append_policy(self) -> bool:
        """Return whether to append the assistant message to the conversation."""
        return False

    def format_assistant_message(self, message: ThoughtSchema) -> dict[str, Any] | None:
        if message.completed:
            if message.final_answer is None:
                raise DistilabelUserError("final_answer cannot be None when completed=True")
            return {
                "role": "assistant",
                "content": f"{THINK_OPEN}{message.thought}{THINK_CLOSE}\n\n{ANSWER_OPEN}{message.final_answer}{ANSWER_CLOSE}",
            }
        if message.tool_call is not None:
            return {
                "role": "assistant",
                "content": THINK_OPEN + message.thought + THINK_CLOSE,
                "tool_calls": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": message.tool_call.name,
                            "arguments": json.dumps(message.tool_call.arguments),
                        },
                    }
                ],
            }
        return None

    def create_tool_message(self, conversation: list[ChatType], message: dict[str, Any]) -> dict[str, Any]:
        # NOTE: Find the last assistant message with tool calls. This only works for single-turn tool calls,
        # if we transition to multiple calls per turn, must match by name or position.
        tool_call_id = None
        for msg in reversed(conversation):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_id = msg.get("tool_calls", [{}])[0].get("id")
                break

        if tool_call_id is None:
            raise DistilabelUserError("No assistant message with tool_calls found for tool response")

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": message["function_name"],
            "content": message["execution_result"],
        }
