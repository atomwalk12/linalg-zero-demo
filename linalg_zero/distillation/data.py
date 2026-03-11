from typing import Any

from pydantic import BaseModel, Field


class FunctionInvocationInfo(BaseModel):
    name: str = Field(..., description="The name of the linear algebra function to call.")
    arguments: dict[str, Any] = Field(..., description="The arguments for the linear algebra function call.")


class AssistantMessage(BaseModel):
    """The structured output for multi-turn tool calling conversations."""

    thinking: str = Field(..., description="The reasoning process for selecting linear algebra tools.")
    tool_calls: list[FunctionInvocationInfo] = Field(
        ..., description="List of linear algebra function calls to solve the problem."
    )


class ThoughtSchema(BaseModel):
    thought: str = Field(
        ..., description="Step-by-step reasoning process for selecting tools or providing the final answer."
    )
    tool_call: FunctionInvocationInfo | None = Field(
        None, description="Tool to use for the next step, or None if problem is solved."
    )
    final_answer: str | None = Field(
        None,
        description="Enter ONLY the final result (e.g., a single number, a vector, or a matrix) after all necessary tool calls have been executed and the problem is completely solved.",
    )
    completed: bool = Field(..., description="Whether the problem is solved.")
