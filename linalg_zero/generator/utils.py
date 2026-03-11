import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from sympy.core.random import seed

from linalg_zero.generator.models import Question
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


def load_entropy_settings(path: str) -> dict[str, Any]:
    """
    Load entropy analysis results from JSON file.
    """
    results_file = Path(path)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Entropy analysis results not found at {results_file}. "
            "Run the entropy analysis first using linalg_zero.generator.analysis.analyse"
        )

    with results_file.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    return settings


def print_entropy_settings(settings: dict[str, Any]) -> None:
    """Print the loaded entropy settings in a readable format."""

    logger.info("=" * 60)
    logger.info("RECOMMENDED ENTROPY SETTINGS")
    logger.info("=" * 60)

    for problem_type, config in settings.items():
        combination = config["combination"]
        score = config["score"]
        logger.info(f"{problem_type}: {combination} (score: {score:.2f})")

    logger.info("=" * 60)


def _verify_step_result(step: dict[str, Any], lib: dict[str, Any]) -> None:
    """Verify a single step's result against library function output."""
    step_id = step["step_id"]

    if "result" not in step:
        raise ValueError(f"Step {step_id} has no result - implementation is bugged")

    result_value = parse_string(step["result"])
    if result_value is None:
        raise ValueError(f"Step {step_id}: invalid result - implementation is bugged")

    fn_type = step["tool"]
    lib_fn = lib[fn_type]
    input_data = json.loads(step["verification"]["input"])
    fn_result = lib_fn(**input_data)

    if not verify_answers(result_value, fn_result):
        raise ValueError(f"Step mismatch - step - {json.dumps(step)} - lib_fn - {fn_type}")


def _verify_step_dependencies(step: dict[str, Any], question_stepwise: list[dict[str, Any]]) -> None:
    """Verify step dependencies against referenced steps."""
    step_id = step.get("step_id", "unknown")
    dependent_on = step["verification"].get("dependent_on", None)

    if dependent_on is None:
        return

    if not isinstance(dependent_on, dict):
        raise TypeError(f"Step {step_id}: dependent_on must be a dict, got {type(dependent_on)}")

    # Verify each input_* field against its corresponding referenced step's result
    for input_name, input_value in step["verification"].items():
        if input_name.startswith("input_"):
            expected_step_index = dependent_on[input_name]

            # Validate the reference step exists
            if not isinstance(expected_step_index, int):
                raise TypeError(
                    f"Step {step_id}: dependency index for '{input_name}' must be an integer, got {type(expected_step_index)}"
                )

            if expected_step_index < 0 or expected_step_index >= len(question_stepwise):
                raise ValueError(
                    f"Step {step_id}: dependent_on index {expected_step_index} for '{input_name}' out of bounds "
                    f"(stepwise has {len(question_stepwise)} steps)"
                )

            referenced_step = question_stepwise[expected_step_index]
            referenced_result = parse_string(referenced_step["result"])

            if referenced_result is None:
                raise ValueError(f"Step {step_id}: referenced step {expected_step_index} has invalid result")

            field_value = json.loads(input_value)
            if not verify_answers(field_value, referenced_result) or field_value != referenced_result:
                raise ValueError(
                    f"Step {step_id}: dependency verification failed - "
                    f"{input_name} ({field_value}) does not match referenced step {expected_step_index} result ({referenced_result})"
                )


def _verify_golden_answer(question: Question, question_index: int) -> None:
    """Verify the golden answer matches the final stepwise result."""
    if not question.golden or "final_answer" not in question.golden:
        raise ValueError(f"Question {question_index} has no golden final answer - implementation is bugged")

    golden_value = parse_string(question.golden["final_answer"])
    answer_value = parse_string(question.stepwise[-1]["result"])

    if golden_value is None:
        raise ValueError(f"Question {question_index}: invalid golden answer - implementation is bugged")
    if answer_value is None:
        raise ValueError(f"Question {question_index}: invalid formatted answer - implementation is bugged")

    if not verify_answers(golden_value, answer_value):
        raise ValueError(
            f"Question {question_index}: Golden answer mismatch - implementation is bugged. "
            f"Golden={golden_value}, Answer={answer_value}"
        )


def verify_dataset(dataset: list[Question]) -> dict[str, Any]:
    """
    Verify a dataset of questions by checking constituent ground truths and target values.
    """
    # NOTE: this function is temporary

    verification_results = {
        "total_questions": len(dataset),
        "verified_questions": 0,
        "stepwise_verifications": 0,
        "golden_verifications": 0,
    }
    lib = get_lib()

    for i, question in enumerate(dataset):
        if len(question.stepwise) == 0:
            raise ValueError(f"Question {i} has no stepwise results - implementation is bugged")

        # Verify stepwise results
        for step in question.stepwise:
            _verify_step_result(step, lib)
            _verify_step_dependencies(step, question.stepwise)
            verification_results["stepwise_verifications"] += 1

        # Verify golden answer
        _verify_golden_answer(question, i)
        verification_results["golden_verifications"] += 1
        verification_results["verified_questions"] += 1

    logger.info(
        "Dataset verification complete: All %d questions verified successfully (%d stepwise checks, %d golden checks)",
        verification_results["total_questions"],
        verification_results["stepwise_verifications"],
        verification_results["golden_verifications"],
    )

    return verification_results


def check_constraints(dataset: list[Question], config: dict[str, Any], statistics: dict[str, Any]) -> None:
    """Check the constraints for the given config."""

    for problem_type, stats in statistics.get("per_problem_type", {}).items():
        if problem_type in config:
            actual_min = stats.get("min")
            actual_max = stats.get("max")
            expected_min = config[problem_type]["metadata"].get("target_min_value")
            expected_max = config[problem_type]["metadata"].get("target_max_value")

            if actual_min is not None and expected_min is not None and actual_min < expected_min:
                raise ValueError(f"{problem_type}: min {actual_min} < expected {expected_min}")
            if actual_max is not None and expected_max is not None and actual_max > expected_max:
                raise ValueError(f"{problem_type}: max {actual_max} > expected {expected_max}")


def set_seed(seed_val: int = 42) -> None:
    """Set the seed for the deterministic generation."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    seed(seed_val)


def print_dataset(questions: list[Question], include_invalid: bool = False) -> None:  # pragma: no cover
    """Display a formatted dataset of questions."""

    questions_to_print = questions if include_invalid else [q for q in questions if q.is_valid]

    if not questions_to_print:
        logger.info("No questions to display.")
        return

    logger.info("=" * 30)
    logger.info("GENERATED DATASET")
    logger.info("=" * 30)

    # Questions
    for i, question in enumerate(questions_to_print, 1):
        status = " [INVALID]" if not question.is_valid else ""
        logger.info("Question %d:%s", i, status)
        logger.info("Q: %s", question.question)
        logger.info("A: %s", ast.literal_eval(question.answer))
        logger.info("")

    # Metadata
    topics = {q.topic for q in questions_to_print}
    problem_types = {q.problem_type for q in questions_to_print}
    difficulties = {q.difficulty for q in questions_to_print}
    entropy_values = [q.entropy_used for q in questions_to_print]
    tool_calls = [q.tool_calls_required for q in questions_to_print]

    # Summary
    logger.info("Dataset Summary:")
    logger.info("  Total Questions: %d", len(questions_to_print))
    logger.info("  Topics: %s", ", ".join(sorted(topic.value for topic in topics)))
    logger.info("  Problem Types: %s", ", ".join(sorted(pt.value for pt in problem_types)))
    logger.info("  Difficulties: %s", ", ".join(sorted(str(difficulty) for difficulty in difficulties)))
    logger.info(
        "  Entropy Used: %.2f - %.2f (avg: %.2f)",
        min(entropy_values),
        max(entropy_values),
        sum(entropy_values) / len(entropy_values),
    )
    logger.info(
        "  Tool Calls Required: %d - %d (avg: %.1f)",
        min(tool_calls),
        max(tool_calls),
        sum(tool_calls) / len(tool_calls),
    )
    # Distributions
    by_difficulty = Counter(q.difficulty for q in questions_to_print)
    logger.info("  By Difficulty:")
    for diff, count in sorted(
        by_difficulty.items(), key=lambda x: x[0].value if hasattr(x[0], "value") else str(x[0])
    ):
        logger.info("    %s: %d", str(diff), count)

    # Per-difficulty averages (entropy and tool calls)
    buckets: dict = defaultdict(list)
    for q in questions_to_print:
        buckets[q.difficulty].append(q)
    logger.info("  Per-Difficulty Averages:")
    for diff, qs in sorted(buckets.items(), key=lambda x: x[0].value if hasattr(x[0], "value") else str(x[0])):
        avg_entropy = sum(q.entropy_used for q in qs) / len(qs)
        avg_tool_calls = sum(q.tool_calls_required for q in qs) / len(qs)
        logger.info("    %s -> entropy avg: %.2f, tool calls avg: %.2f", str(diff), avg_entropy, avg_tool_calls)
    logger.info("=" * 30)


def print_split_statistics(splits: DatasetDict) -> None:  # pragma: no cover
    """Display statistics about train/validation/test splits."""
    logger.info("=" * 30)
    logger.info("DATASET SPLITS")
    logger.info("=" * 30)

    total = sum(len(splits[split_name]) for split_name in splits)

    for split_name in ["train", "validation", "test"]:
        if split_name in splits:
            count = len(splits[split_name])
            percentage = 100 * count / total if total > 0 else 0
            logger.info("  %s: %d samples (%.1f%%)", split_name.capitalize(), count, percentage)

    logger.info("  Total: %d samples", total)
    logger.info("=" * 30)


def _question_to_example(q: Question) -> dict[str, Any]:
    """Map a Question to a flat example for Hugging Face datasets."""
    stepwise_truths: list[dict[str, Any]] = []
    for step in q.stepwise:
        tool_name = step.get("tool")
        result_value = parse_string(step.get("result"))
        if tool_name is None or result_value is None:
            continue
        stepwise_truths.append({tool_name: result_value})

    # Derive composition metadata from stepwise verification and problem_type
    composition_type = "sequential" if len(q.stepwise) > 1 else "single"
    dependency_edges: list[tuple[int, int]] = []
    for idx, step in enumerate(q.stepwise):
        verification = step.get("verification", {})
        if isinstance(verification, dict):
            dependent_on = verification.get("dependent_on")
            if isinstance(dependent_on, dict):
                for _, from_idx in dependent_on.items():
                    if isinstance(from_idx, int):
                        dependency_edges.append((from_idx, idx))

    dependency_type = "strict"

    return {
        "query": q.question,
        "ground_truth": q.golden.get("final_answer", q.answer),
        "stepwise_ground_truths": json.dumps(stepwise_truths),
        "difficulty": getattr(q.difficulty, "name", str(q.difficulty)),
        "problem_type": getattr(q.problem_type, "value", str(q.problem_type)),
        "composition_type": composition_type,
        "composition_dependencies": dependency_type,
        "dependency_edges": json.dumps(dependency_edges) if dependency_edges else None,
    }


def convert_to_dataset_dict(questions: list[Question]) -> DatasetDict:
    """Convert questions to a single-split DatasetDict (train)."""
    examples = [_question_to_example(q) for q in questions if q.is_valid]
    return DatasetDict({"train": Dataset.from_list(examples)})


def convert_to_dataset_splits(
    questions: list[Question],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    stratify_by: str | None = None,
) -> DatasetDict:
    """Create train/validation/test DatasetDict using HF's split utilities."""

    examples = [_question_to_example(q) for q in questions if q.is_valid]
    ds = Dataset.from_list(examples).shuffle(seed=seed)

    stratify_column = stratify_by if stratify_by in ds.column_names else None

    # Convert stratification column to ClassLabel if needed
    if stratify_column and ds.features[stratify_column]._type != "ClassLabel":
        from datasets import ClassLabel

        unique_values = ds.unique(stratify_column)
        ds = ds.cast_column(stratify_column, ClassLabel(names=sorted(unique_values)))

    split = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column=stratify_column)

    if val_size and val_size > 0:
        # Adjust val proportion relative to remaining train portion
        relative_val = val_size / (1 - test_size)
        train_val = split["train"].train_test_split(
            test_size=relative_val, seed=seed, stratify_by_column=stratify_column
        )
        return DatasetDict(train=train_val["train"], validation=train_val["test"], test=split["test"])  # type: ignore[reportCallIssue]

    return DatasetDict(train=split["train"], test=split["test"])  # type: ignore[reportCallIssue]
