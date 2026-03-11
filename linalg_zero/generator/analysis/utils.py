import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from linalg_zero.generator.models import DifficultyCategory, Question, Task
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


def print_statistics_summary(statistics: dict[str, Any]) -> None:  # pragma: no cover
    """Print a formatted summary of dataset statistics."""
    if not statistics:
        logger.info("No statistics available.")
        return

    logger.info("=" * 50)
    logger.info("DATASET STATISTICS SUMMARY")
    logger.info("=" * 50)

    # Overall statistics
    overall_min = statistics.get("overall_min")
    overall_max = statistics.get("overall_max")
    overall_min_abs = statistics.get("overall_min_abs")
    logger.info(f"Overall Range: {overall_min} to {overall_max}")
    logger.info(f"Overall Min Absolute: {overall_min_abs}")

    # Per-step statistics
    per_step = statistics.get("per_step", {})
    if per_step:
        logger.info("Per-Step Statistics:")
        for step_idx in sorted(per_step.keys()):
            step_stats = per_step[step_idx]
            logger.info(
                f"  Step {step_idx}: min={step_stats.get('min')}, max={step_stats.get('max')}, min_abs={step_stats.get('min_abs')}, count={step_stats.get('count')}"
            )

    # Per-problem-type statistics
    per_problem_type = statistics.get("per_problem_type", {})
    if per_problem_type:
        logger.info("Per-Problem-Type Statistics:")
        for problem_type, type_stats in per_problem_type.items():
            logger.info(
                f"  {problem_type}: min={type_stats.get('min')}, max={type_stats.get('max')}, min_abs={type_stats.get('min_abs')}, count={type_stats.get('count')}"
            )

    # Per-question statistics
    per_question = statistics.get("per_question", [])
    if per_question:
        logger.info(f"Per-Question Statistics: {len(per_question)} questions analyzed")
        # Show first few questions as examples
        for i, q_stats in enumerate(per_question[:3]):
            logger.info(
                f"  Q{i + 1}: min={q_stats.get('min')}, max={q_stats.get('max')}, min_abs={q_stats.get('min_abs')}, count={q_stats.get('count')}"
            )
        if len(per_question) > 3:
            logger.info(f"  ... and {len(per_question) - 3} more questions")

    logger.info("=" * 50)


def _extract_numeric_values_from_object(obj: Any) -> list[float]:
    """Recursively extract numeric values (as floats) from an arbitrary object."""
    values: list[float] = []

    if isinstance(obj, int | float):
        values.append(float(obj))
        return values

    if isinstance(obj, complex):
        raise TypeError(f"Complex number found: {obj}")

    if isinstance(obj, list | tuple):
        for item in obj:
            values.extend(_extract_numeric_values_from_object(item))
        return values

    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(_extract_numeric_values_from_object(v))
        return values

    return values


def compute_stepwise_value_statistics(questions: list[Question]) -> dict[str, Any]:
    """Scan stepwise results from all questions and compute statistics.

    Returns a dictionary with:
    - overall_min: float | None
    - overall_max: float | None
    - overall_min_abs: float | None
    - per_question: list of {index, min, max, min_abs, count}
    - per_step: dict[int, {min, max, min_abs, count}] aggregated across all questions by step index
    - per_problem_type: dict[str, {min, max, min_abs, count}] aggregated by problem type
    - all_values: flat list[float] of every numeric value encountered across all steps/questions
    """
    overall_min: float | None = None
    overall_max: float | None = None
    overall_min_abs: float | None = None

    per_question: list[dict[str, Any]] = []
    per_step: dict[int, dict[str, Any]] = {}
    per_problem_type: dict[str, dict[str, Any]] = {}

    all_values: list[float] = []

    for q_index, question in enumerate(questions):
        q_min: float | None = None
        q_max: float | None = None
        q_min_abs: float | None = None
        q_count: int = 0

        # Resolve problem type key once per question
        pt_key = getattr(question.problem_type, "value", str(question.problem_type))

        for step_index, step in enumerate(question.stepwise):
            # Parse the step result into a Python object
            result_str = step.get("result")
            if result_str is None:
                raise ValueError(f"Step {step_index} has no result")
            parsed = ast.literal_eval(result_str)
            # Extract numeric values
            numeric_values = _extract_numeric_values_from_object(parsed)

            if not numeric_values:
                # Initialize per-step entry with zero count if not present
                if step_index not in per_step:
                    per_step[step_index] = {"min": None, "max": None, "min_abs": None, "count": 0}
                continue

            step_min = min(numeric_values)
            step_max = max(numeric_values)
            step_min_abs = min(abs(v) for v in numeric_values)
            step_count = len(numeric_values)

            # Aggregate raw values
            all_values.extend(float(v) for v in numeric_values)

            # Update overall stats
            overall_min = step_min if overall_min is None else min(overall_min, step_min)
            overall_max = step_max if overall_max is None else max(overall_max, step_max)
            overall_min_abs = step_min_abs if overall_min_abs is None else min(overall_min_abs, step_min_abs)

            # Update question stats
            q_min = step_min if q_min is None else min(q_min, step_min)
            q_max = step_max if q_max is None else max(q_max, step_max)
            q_min_abs = step_min_abs if q_min_abs is None else min(q_min_abs, step_min_abs)
            q_count += step_count

            # Update per-step aggregated stats
            if step_index not in per_step:
                per_step[step_index] = {"min": step_min, "max": step_max, "min_abs": step_min_abs, "count": step_count}
            else:
                ps = per_step[step_index]
                ps_min = ps["min"]
                ps_max = ps["max"]
                ps_min_abs = ps["min_abs"]
                ps["min"] = step_min if ps_min is None else min(ps_min, step_min)
                ps["max"] = step_max if ps_max is None else max(ps_max, step_max)
                ps["min_abs"] = step_min_abs if ps_min_abs is None else min(ps_min_abs, step_min_abs)
                ps["count"] += step_count

            # Update per-problem-type aggregated stats
            if pt_key not in per_problem_type:
                per_problem_type[pt_key] = {
                    "min": step_min,
                    "max": step_max,
                    "min_abs": step_min_abs,
                    "count": step_count,
                }
            else:
                ppt = per_problem_type[pt_key]
                ppt_min = ppt["min"]
                ppt_max = ppt["max"]
                ppt_min_abs = ppt["min_abs"]
                ppt["min"] = step_min if ppt_min is None else min(ppt_min, step_min)
                ppt["max"] = step_max if ppt_max is None else max(ppt_max, step_max)
                ppt["min_abs"] = step_min_abs if ppt_min_abs is None else min(ppt_min_abs, step_min_abs)
                ppt["count"] += step_count

        per_question.append({"index": q_index, "min": q_min, "max": q_max, "min_abs": q_min_abs, "count": q_count})

    return {
        "overall_min": overall_min,
        "overall_max": overall_max,
        "overall_min_abs": overall_min_abs if overall_min_abs is not None else 0.0,
        "per_question": per_question,
        "per_step": per_step,
        "per_problem_type": per_problem_type,
        "all_values": all_values,
    }


def extract_all_numerical_values(statistics: dict[tuple, dict[str, Any]], use_min_max: bool = False) -> list[float]:
    """Extract all raw stepwise numerical values from all runs.

    Uses the aggregated `all_values` emitted by `compute_stepwise_value_statistics` for
    each combination rather than only per-question min/max boundaries.
    """
    all_values: list[float] = []

    for _, stats in statistics.items():
        if use_min_max:
            values = []
            per_question = stats.get("per_question", [])
            for q_stats in per_question:
                values.append(q_stats["min"])
                values.append(q_stats["max"])
        else:
            values = stats.get("all_values", [])
        all_values.extend(values)

    for value in all_values:
        assert value is not None, "Value is None"

    return all_values


def extract_values_by_combination(
    statistics: dict[tuple, dict[str, Any]], use_min_max: bool = False
) -> dict[tuple, list[float]]:
    """Extract raw stepwise numerical values grouped by entropy combination."""
    values_by_combination: dict[tuple, list[float]] = {}

    for combination, stats in statistics.items():
        if use_min_max:
            values = []
            per_question = stats.get("per_question", [])
            for q_stats in per_question:
                values.append(q_stats["min"])
                values.append(q_stats["max"])
        else:
            values = list(stats.get("all_values", []))
        values_by_combination[combination] = values

    for key, items in values_by_combination.items():
        assert all(value is not None for value in items), f"Values are None for combination {key}"

    return values_by_combination


def plot_overall_histogram(
    all_values: list[float], target_min: float = -1000, target_max: float = 1000, output_dir: Path | None = None
) -> None:
    """Plot histogram of all numerical values across all runs."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(all_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(target_min, color="red", linestyle="--", label=f"Target Min ({target_min})")
    plt.axvline(target_max, color="red", linestyle="--", label=f"Target Max ({target_max})")
    plt.xlabel("Numerical Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of All Numerical Values")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log scale version
    plt.subplot(1, 2, 2)
    # Filter out zero and negative values for log scale
    positive_values = [v for v in all_values if v > 0]
    if positive_values:
        plt.hist(positive_values, bins=50, alpha=0.7, color="lightcoral", edgecolor="black")
        plt.axvline(target_max, color="red", linestyle="--", label=f"Target Max ({target_max})")
        plt.xlabel("Numerical Values (log scale)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Positive Values (Log Scale)")
        plt.xscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "overall_distribution.png"
    filepath = output_dir / filename if output_dir else filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_combination_comparison(
    values_by_combination: dict[tuple, list[float]], max_combinations: int = 12, output_dir: Path | None = None
) -> None:
    """Plot comparison of value distributions across different entropy combinations."""
    # Limit to top combinations by number of values
    sorted_combinations = sorted(values_by_combination.items(), key=lambda x: len(x[1]), reverse=True)[
        :max_combinations
    ]

    _, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (combination, values) in enumerate(sorted_combinations):
        if i >= len(axes):
            break

        ax = axes[i]
        if values:  # Only plot if we have values
            ax.hist(values, bins=20, alpha=0.7, color=plt.colormaps["tab10"](i % 10), edgecolor="black")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"Combo: {combination}", fontsize=10)
            ax.set_xlabel("Values")
            ax.set_ylabel("Freq")
            ax.grid(True, alpha=0.3)

            # Add statistics text
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.text(
                    0.05,
                    0.95,
                    f"μ={mean_val:.1f}\no={std_val:.1f}\nn={len(values)}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

    # Hide unused subplots
    for i in range(len(sorted_combinations), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Value Distributions by Entropy Combination", fontsize=14)
    plt.tight_layout()
    filename = "by_combination.png"
    filepath = output_dir / filename if output_dir else filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_target_compliance(
    statistics: dict[tuple, dict[str, Any]],
    target_min: float = -1000,
    target_max: float = 1000,
    output_dir: Path | None = None,
) -> None:
    """Plot how well each combination complies with target ranges."""
    compliance_data = []

    for combination, stats in statistics.items():
        overall_min = stats.get("overall_min")
        overall_max = stats.get("overall_max")

        if overall_min is None or overall_max is None:
            raise ValueError(f"Overall min or max is None for combination {combination}")

        # overall_min/overall_max are guaranteed non-None above; avoid falsy-zero filtering
        within_range = (target_min <= overall_min) and (overall_max <= target_max)
        compliance_data.append({
            "combination": str(combination),
            "min": overall_min,
            "max": overall_max,
            "compliant": within_range,
        })

    if not compliance_data:
        logger.warning("No compliance data available for plotting")
        return

    compliant = [d for d in compliance_data if d["compliant"]]
    non_compliant = [d for d in compliance_data if not d["compliant"]]

    plt.figure(figsize=(12, 8))

    # Plot compliant combinations
    if compliant:
        plt.scatter(
            [d["min"] for d in compliant],
            [d["max"] for d in compliant],
            c="green",
            alpha=0.7,
            s=100,
            label=f"Compliant ({len(compliant)})",
        )

    # Plot non-compliant combinations
    if non_compliant:
        plt.scatter(
            [d["min"] for d in non_compliant],
            [d["max"] for d in non_compliant],
            c="red",
            alpha=0.7,
            s=100,
            label=f"Non-compliant ({len(non_compliant)})",
        )

    # Add target range box
    plt.axhline(target_max, color="blue", linestyle="--", alpha=0.5, label=f"Target Max ({target_max})")
    plt.axvline(target_min, color="blue", linestyle="--", alpha=0.5, label=f"Target Min ({target_min})")

    plt.xlabel("Overall Minimum Value")
    plt.ylabel("Overall Maximum Value")
    plt.title("Entropy Combinations: Target Range Compliance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "compliance.png"
    filepath = output_dir / filename if output_dir else filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    total_combinations = len(compliance_data)
    compliant_count = len(compliant)
    logger.info("Target Compliance Summary:")
    logger.info(f"  Total combinations: {total_combinations}")
    logger.info(f"  Compliant: {compliant_count} ({compliant_count / total_combinations * 100:.1f}%)")
    logger.info(
        f"  Non-compliant: {total_combinations - compliant_count} ({(total_combinations - compliant_count) / total_combinations * 100:.1f}%)"
    )


def extract_report_metadata(
    top_choice: dict[str, Any],
    problem_type: Task,
    entropy_config: tuple[float, float] | dict[Task, tuple[float, float]],
    min_value_abs: float,
    entropy_jitter: float,
    *,
    step_size: float,
    samples_per_test: int,
    target_min_value: float,
    target_max_value: float,
) -> dict[str, Any]:
    if isinstance(entropy_config, tuple):
        # Single-step problem
        is_single_step = True
        components = [problem_type.name]
        difficulty_category = DifficultyCategory.ONE_TOOL_CALL.name
    else:
        # Multi-step problem - get components from the dict keys
        is_single_step = False
        components = [task.name for task in entropy_config if isinstance(task, Task)]

        # Determine difficulty category based on number of components
        num_components = len(components)
        if num_components == 2:
            difficulty_category = DifficultyCategory.TWO_TOOL_CALLS.name
        elif num_components == 3:
            difficulty_category = DifficultyCategory.THREE_TOOL_CALLS.name
        else:
            raise ValueError(f"Unexpected number of components for {problem_type}: {num_components}")

    # Validate the ordered combination length matches the number of components
    selected_combination = (
        list(top_choice["combination"])
        if isinstance(top_choice["combination"], list | tuple)
        else [top_choice["combination"]]
    )
    if len(selected_combination) != len(components):
        raise ValueError(
            f"Mismatch between combination length ({len(selected_combination)}) and components ({len(components)}) for {problem_type.name}"
        )

    # Optimized within the searched grid if the chosen entropy is strictly below
    # the configured upper bound (i.e., the search did not hit the boundary).
    optimized = False
    if isinstance(entropy_config, tuple):
        if len(selected_combination) > 0 and selected_combination[0] < (entropy_config[1] - entropy_jitter):
            optimized = True
    else:
        # Multi-step: mark optimized if ANY component's chosen entropy is strictly
        # below its configured upper bound (didn't hit boundary for that component).
        component_index = 0
        for task in entropy_config:
            if isinstance(task, Task) and component_index < len(selected_combination):
                if selected_combination[component_index] < (entropy_config[task][1] - entropy_jitter):
                    optimized = True
                    break
                component_index += 1

    return {
        "combination": selected_combination,
        "score": top_choice["score"],
        "overall_min": top_choice["overall_min"],
        "overall_max": top_choice["overall_max"],
        "min_abs": top_choice["min_abs"],
        "count": top_choice["count"],
        "optimized": optimized,
        "metadata": {
            "is_single_step": is_single_step,
            "components": components,
            "difficulty_category": difficulty_category,
            "task_enum": problem_type.name,
            "entropy_jitter": entropy_jitter,
            "min_element_abs": min_value_abs,
            "step_size": step_size,
            "samples_per_test": samples_per_test,
            "target_min_value": target_min_value,
            "target_max_value": target_max_value,
        },
    }


def rank_entropy_combinations(
    statistics: dict[tuple, dict[str, Any]],
    *,
    target_min: float,
    target_max: float,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank entropy combinations by distribution quality.

    Returns a list of dicts sorted by descending score, each with keys:
    - combination: tuple of entropy values
    - score: float in [0, 1]
    - metrics: dict as computed by _compute_distribution_metrics
    """
    if weights is None:
        weights = {"compliance": 0.4, "center": 0.2, "coverage": 0.2, "balance": 0.1, "zero": 0.1}

    ranked: list[dict[str, Any]] = []
    for combination, stats in statistics.items():
        # Hard gate 1: overall range must be fully within targets if required
        overall_min = stats["overall_min"]
        overall_max = stats["overall_max"]
        if overall_min is None or overall_max is None:
            continue
        if not (target_min <= float(overall_min) <= float(target_max)):
            continue
        if not (target_min <= float(overall_max) <= float(target_max)):
            continue

        # Calculate distance from targets (lower is better)
        min_distance = abs(float(overall_min) - target_min)
        max_distance = abs(float(overall_max) - target_max)
        total_distance = min_distance + max_distance

        # Get min_abs and total count from statistics
        overall_min_abs = stats["overall_min_abs"]

        # Calculate total count across all questions
        total_count = sum(q_stats["count"] for q_stats in stats["per_question"])

        ranked.append({
            "combination": combination,
            "score": total_distance,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "overall_min": overall_min,
            "overall_max": overall_max,
            "min_abs": overall_min_abs,
            "count": total_count,
        })

    # Sort by total distance (ascending - closest first)
    ranked.sort(key=lambda d: d["score"])
    return ranked
