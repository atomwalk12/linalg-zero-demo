"""Diagnostic metrics tracker for tool calling evaluation."""

from __future__ import annotations

import json as _json
from typing import Any

from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.lib import get_lib_fn_names
from linalg_zero.shared.types import LibTypes


class DiagnosticTracker:
    """Tracks messages and samples for evaluation logging."""

    def __init__(self) -> None:
        # Store all messages and samples for Weave logging
        self.all_messages: list[list[dict[str, Any]]] = []
        self.all_samples: list[dict[str, Any]] = []
        self.all_strict_formats: list[float] = []
        self.all_partial_formats: list[float] = []
        self.all_generated_answers: list[str | LibTypes | None] = []

    def update(self, state: EvaluationState) -> None:
        """Update tracker from an evaluation state."""
        self.all_strict_formats.append(state.strict_format_match)
        self.all_partial_formats.append(state.partial_format_score)
        self.all_generated_answers.append(state.generated_answer)
        # Store messages and sample from this evaluation
        self.all_messages.append(state.messages)
        if state.sample is not None:
            self.all_samples.append(state.sample)

    def get_history(self) -> tuple[list[list[dict[str, Any]]], dict[str, int | float], dict[str, float]]:
        """Return all messages, metadata and loss metrics for Weave logging best model selection."""
        metadata = self._compute_metadata()
        loss_metrics = self.calculate_loss_metrics()
        return self.all_messages, {**metadata, **loss_metrics}, loss_metrics

    def _compute_metadata(self) -> dict[str, int]:
        """Compute metadata statistics from messages and samples."""
        parser = XMLParser()
        tool_names = get_lib_fn_names()

        total_samples = len(self.all_samples)
        total_expected_tool_calls = 0
        total_actual_tool_calls = 0
        total_expected_answers = total_samples  # One answer per sample
        total_actual_answers = 0
        total_correct_answers = 0

        for messages, sample in zip(self.all_messages, self.all_samples, strict=True):
            # Calculate expected tool calls from ground truth
            if "stepwise_ground_truths" in sample:
                ground_truths = _json.loads(sample["stepwise_ground_truths"])
                total_expected_tool_calls += len(ground_truths)

            # Track the last answer found in this conversation
            last_answer = None

            # Count actual tool calls and answers from assistant messages
            # Skip first two messages (system and user)
            for message in messages[2:]:
                if message.get("role") != "assistant":
                    continue

                content = message.get("content", "")
                analysis = parser.analyze_message_in_context(
                    messages[: messages.index(message) + 1], message=content, tool_names=tool_names
                )

                # Count tool calls
                if analysis.get("tool") and analysis["tool"].get("json_valid"):
                    total_actual_tool_calls += 1

                # Count answers and track the last one
                if analysis.get("has_answer"):
                    total_actual_answers += 1
                    last_answer = analysis.get("answer")

            # Check if the last answer is correct
            if last_answer is not None and "ground_truth" in sample:
                try:
                    ground_truth = sample["ground_truth"]
                    parsed_gt = parse_string(ground_truth)
                    parsed_answer = parse_string(last_answer)

                    if verify_answers(parsed_gt, parsed_answer):
                        total_correct_answers += 1
                except Exception:  # noqa: S110
                    pass

        return {
            "total_samples": total_samples,
            "total_expected_tool_calls": total_expected_tool_calls,
            "total_actual_tool_calls": total_actual_tool_calls,
            "total_expected_answers": total_expected_answers,
            "total_actual_answers": total_actual_answers,
            "total_correct_answers": total_correct_answers,
        }

    def calculate_loss_metrics(self) -> dict[str, float]:
        metrics = self._compute_metadata()
        expected_tool_calls = metrics["total_expected_tool_calls"]
        expected_answers = metrics["total_expected_answers"]
        total_samples = metrics["total_samples"]
        total_actions = expected_tool_calls + total_samples
        format_overall_accuracy = (
            (metrics["total_actual_tool_calls"] + metrics["total_actual_answers"]) / total_actions
            if total_actions > 0
            else 0.0
        )
        format_tool_call_accuracy = (
            metrics["total_actual_tool_calls"] / expected_tool_calls if expected_tool_calls > 0 else 0.0
        )
        answer_attempt_accuracy = metrics["total_actual_answers"] / expected_answers if expected_answers > 0 else 0.0
        answer_accuracy = metrics["total_correct_answers"] / total_samples if total_samples > 0 else 0.0
        return {
            "format_accuracy": format_overall_accuracy,
            "format_tool_call_accuracy": format_tool_call_accuracy,
            "format_answer_accuracy": answer_attempt_accuracy,
            "answer_accuracy": answer_accuracy,
        }

    def get_progress_info(self) -> dict[str, str]:
        """Return current progress info for progress bar (3 key metrics)."""
        total_samples = len(self.all_samples)
        if total_samples == 0:
            return {
                "strict": "0.000",
                "partial": "0.000",
                "correct": "0.000",
            }

        partial_format = sum(self.all_partial_formats) / total_samples
        strict_format = sum(self.all_strict_formats) / total_samples
        answers_generated = sum(1 for ans in self.all_generated_answers if ans is not None)
        answer_rate = answers_generated / total_samples

        return {
            "strict": f"{strict_format:.3f}",
            "partial": f"{partial_format:.3f}",
            "correct": f"{answer_rate:.3f}",
        }
