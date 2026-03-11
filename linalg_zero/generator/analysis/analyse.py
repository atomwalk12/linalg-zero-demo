import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np

import linalg_zero.generator.difficulty_config as dc
from linalg_zero.generator.analysis.utils import (
    compute_stepwise_value_statistics,
    extract_all_numerical_values,
    extract_report_metadata,
    extract_values_by_combination,
    plot_combination_comparison,
    plot_overall_histogram,
    plot_target_compliance,
    print_statistics_summary,
    rank_entropy_combinations,
)
from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.difficulty_config import (
    DETERMINISTIC_MODE,
    determine_difficulty,
    get_problem_config,
)
from linalg_zero.generator.models import Question, Task, Topic
from linalg_zero.generator.registry import (
    FactoryRegistry,
    register_problem_type,
)
from linalg_zero.generator.utils import set_seed
from linalg_zero.shared.utils import get_log_file_path, get_logger, setup_logging

MIN_VALUE_ABS = 2
STEP_SIZE = 0.1
SAMPLES_PER_TEST = 8000
DEFAULT_ENTROPY_JITTER = 0.2
WRITE_PER_PROBLEM_RANKINGS = True
PROBLEM_DIR = Path("results") / "entropy_analysis"

ALL_ENTROPY_RANGES = {
    # 1-Tool Call Problems (Foundation Layer)
    Task.ONE_DETERMINANT: {
        "entropy_ranges": (0.6, 1.2),
        "target_min": -500,
        "target_max": 500,
    },
    Task.ONE_TRACE: {
        "entropy_ranges": (1.5, 2.5),
        "target_min": -200,
        "target_max": 200,
    },
    Task.ONE_FROBENIUS_NORM: {
        "entropy_ranges": (2.5, 3.3),
        "target_min": 0,
        "target_max": 600,  # Always positive
    },
    Task.ONE_RANK: {
        "entropy_ranges": (2.5, 3.0),
        "target_min": 1,
        "target_max": 3,
    },
    Task.ONE_TRANSPOSE: {
        "entropy_ranges": (3.0, 3.5),
        "target_min": -800,
        "target_max": 800,
    },
    Task.ONE_COFACTOR: {
        "entropy_ranges": (1.4, 2.0),
        "target_min": -800,
        "target_max": 800,
    },
    # 2-Tool Call Problems (Sequential Reasoning)
    Task.TWO_TRANSPOSE_DETERMINANT: {
        "target_min": -400,
        "target_max": 400,
        Task.ONE_TRANSPOSE: (0.7, 1.8),
        Task.ONE_DETERMINANT: (0.0, 0.0),
    },
    Task.TWO_COFACTOR_TRACE: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_COFACTOR: (1.0, 2.0),
        Task.ONE_TRACE: (0.0, 0.0),
    },
    Task.TWO_COFACTOR_RANK: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_COFACTOR: (1.0, 2.0),
        Task.ONE_RANK: (0.0, 0.0),
    },
    Task.TWO_TRANSPOSE_FROBENIUS: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_TRANSPOSE: (1.8, 3.2),
        Task.ONE_FROBENIUS_NORM: (0.0, 0.0),
    },
    # 3-Tool Call Problems (Advanced Sequential/Fan-out)
    Task.THREE_TRANSPOSE_COFACTOR_RANK: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_TRANSPOSE: (2.5, 3.6),
        Task.ONE_COFACTOR: (0.0, 0.0),
        Task.ONE_RANK: (0.0, 0.0),
    },
    Task.THREE_COFACTOR_TRANSPOSE_TRACE: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_COFACTOR: (2.5, 3.6),
        Task.ONE_TRANSPOSE: (0.0, 0.0),
        Task.ONE_TRACE: (0.0, 0.0),
    },
    Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS: {
        "target_min": -800,
        "target_max": 800,
        Task.ONE_TRANSPOSE: (2.8, 3.3),
        Task.ONE_COFACTOR: (0.0, 0.0),
        Task.ONE_FROBENIUS_NORM: (0.0, 0.0),
    },
}


setup_logging()
logger = get_logger(__name__)


class EntropyOptimizer:
    def __init__(self, registry: FactoryRegistry, topic: Topic):
        self.registry: FactoryRegistry = registry
        self.generator = DatasetGenerator(topic=topic, registry=registry)

    def execute(
        self,
        component_entropy_ranges: dict[Task, tuple[float, float]],
        problem_type: Task,
    ) -> dict[tuple[float, ...], list[Question]]:
        # Get correct config based on problem type (based on whether it is a 1, 2, or 3 step problem)
        difficulty = determine_difficulty(problem_type)

        # Generate all combinations of entropy values
        grid_points = {}
        for component, (min_val, max_val) in component_entropy_ranges.items():
            grid_points[component] = np.arange(min_val, max_val + STEP_SIZE / 2, STEP_SIZE)

        for component, values in grid_points.items():
            grid_points[component] = np.round(values, 1)

        # Create all combinations
        component_names = list(grid_points.keys())
        value_combinations = list(itertools.product(*[grid_points[name] for name in component_names]))

        logger.info(f"Total configurations to test: {len(value_combinations)}")

        dataset_by_combination = {}
        failed_configurations = []
        logger.info(f"Testing {problem_type.value} with {SAMPLES_PER_TEST} questions per configuration")
        for i, combination in enumerate(value_combinations):
            logger.info(f"Testing configuration {combination} {i + 1}/{len(value_combinations)}")
            entropy_ranges = {
                Task(component_name): combination_value
                for component_name, combination_value in zip(component_names, combination, strict=True)
            }

            try:
                register_problem_type(
                    self.registry, problem_type, entropy_ranges, DEFAULT_ENTROPY_JITTER, MIN_VALUE_ABS
                )

                split = self.generator.generate_exact_for_categories(
                    requests={
                        difficulty: SAMPLES_PER_TEST,
                    }
                )
                dataset_by_combination[combination] = split
            except RuntimeError as e:
                logger.warning(f"Failed to generate for configuration {combination}: {e}")
                failed_configurations.append((combination, str(e)))
                continue

        # Log summary of failures
        if failed_configurations:
            logger.warning(
                f"Failed to generate for {len(failed_configurations)} out of {len(value_combinations)} configurations"
            )
            for combination, error in failed_configurations:
                logger.debug(f"Failed configuration {combination}: {error}")
        else:
            logger.info("All configurations generated successfully")

        return dataset_by_combination


def execute_analysis(
    topic: Topic,
    problem_type: Task,
    component_entropy_ranges: dict[Task, tuple[float, float]],
    target_min: float,
    target_max: float,
    print_individual_stats: bool = True,
) -> tuple[dict[tuple, dict[str, Any]], dict[str, Any]]:
    logger.info(f"Optimizing entropy for: {problem_type}")
    logger.info("This will systematically test different component-wise entropy configurations.")
    logger.info("Configuration:")
    logger.info(f"  Component ranges: {component_entropy_ranges}")
    logger.info(f"  Samples per test: {SAMPLES_PER_TEST}")
    logger.info(f"  Target max value: {target_max}")
    logger.info(f"  Target min value: {target_min}")

    config = get_problem_config(determine_difficulty(problem_type))

    logger.info(f"  Problem config: {config}")

    registry = FactoryRegistry()
    optimizer = EntropyOptimizer(registry, topic)
    dataset = optimizer.execute(
        component_entropy_ranges=component_entropy_ranges,
        problem_type=problem_type,
    )

    statistics = {}
    for combination, split in dataset.items():
        statistics[combination] = compute_stepwise_value_statistics(split)

    if print_individual_stats:
        for combination, stats in statistics.items():
            logger.info(f"Combination: {combination}")
            print_statistics_summary(stats)

    # After computing statistics, rank combinations and optionally write per-problem report + plots
    PROBLEM_DIR.mkdir(parents=True, exist_ok=True)

    # Ranking based on raw values
    ranked = rank_entropy_combinations(
        statistics,
        target_min=target_min,
        target_max=target_max,
    )

    top_k = min(10, len(ranked))

    report = {
        "top": ranked[:top_k],
    }
    # Prepare per-problem directory path regardless of flag to avoid unbound warnings
    per_problem_dir = PROBLEM_DIR / problem_type.value
    if WRITE_PER_PROBLEM_RANKINGS:
        per_problem_dir.mkdir(parents=True, exist_ok=True)
        with (per_problem_dir / "ranking.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            logger.info(f"ranking saved to: {per_problem_dir / 'ranking.json'}")

    logger.info(f"number of top combinations: {len(ranked)}")

    # Optional plots (use raw values)
    all_values = extract_all_numerical_values(statistics, use_min_max=False)
    values_by_combination = extract_values_by_combination(statistics, use_min_max=False)

    if WRITE_PER_PROBLEM_RANKINGS:
        plot_overall_histogram(all_values, output_dir=per_problem_dir, target_min=target_min, target_max=target_max)
        plot_combination_comparison(values_by_combination, output_dir=per_problem_dir)
        plot_target_compliance(statistics, output_dir=per_problem_dir, target_min=target_min, target_max=target_max)

    return statistics, report


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    all_reports = {}
    all_statistics = {}

    for problem_type, config in ALL_ENTROPY_RANGES.items():
        config_dict = cast(dict[str, Any], config)
        target_min = config_dict["target_min"]
        target_max = config_dict["target_max"]

        # Parse simple or nested config
        if "entropy_ranges" in config_dict:
            # Simple config
            entropy_ranges = config_dict["entropy_ranges"]
            component_entropy_ranges = {problem_type: entropy_ranges}
        else:
            # Nested config
            component_entropy_ranges = {}
            for key, value in config_dict.items():
                if not isinstance(key, str):
                    component_entropy_ranges[key] = value

        statistics, report = execute_analysis(
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=problem_type,
            component_entropy_ranges=component_entropy_ranges,
            target_min=target_min,
            target_max=target_max,
            print_individual_stats=False,
        )

        all_reports[problem_type] = report
        all_statistics[problem_type] = statistics

    logger.info("=" * 80)
    logger.info("FINAL ANALYSIS SUMMARY - ALL PROBLEM TYPES")
    logger.info("=" * 80)

    top_choices = {}

    for problem_type, report in all_reports.items():
        logger.info(f"{problem_type.value.upper()} RESULTS:")
        logger.info("-" * 50)

        # Print top combinations for this problem type
        if report["top"]:
            config = ALL_ENTROPY_RANGES[problem_type]
            config_dict = cast(dict[str, Any], config)
            entropy_config = config_dict.get("entropy_ranges", config_dict)
            top_choices[problem_type.value] = extract_report_metadata(
                top_choice=report["top"][0],
                problem_type=problem_type,
                entropy_config=entropy_config,
                min_value_abs=MIN_VALUE_ABS,
                entropy_jitter=DEFAULT_ENTROPY_JITTER,
                step_size=STEP_SIZE,
                samples_per_test=SAMPLES_PER_TEST,
                target_min_value=config_dict["target_min"],
                target_max_value=config_dict["target_max"],
            )
            top = 5
            logger.info(f"Top entropy combinations (closest to targets) {top}/{len(report['top'])}:")
            for i, entry in enumerate(report["top"][:top]):  # Show top 5
                combination = entry["combination"]
                score = entry["score"]
                overall_min = entry["overall_min"]
                overall_max = entry["overall_max"]
                overall_min_abs = entry["min_abs"]
                count = entry["count"]
                logger.info(
                    f"  {i + 1}. {combination} -> score={score:.2f}, range=[{overall_min:.2f}, {overall_max:.2f}], min_abs={overall_min_abs:.2f}, count={count}"
                )

    # Write consolidated top choices for all problem types
    consolidated_path = PROBLEM_DIR / "top_entropy_choices.json"
    with consolidated_path.open("w", encoding="utf-8") as f:
        json.dump(top_choices, f, indent=2)
        logger.info(f"Consolidated top choices saved to: {consolidated_path}")

    logger.info(f"Log file path: {get_log_file_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    argv = parser.parse_args()
    if argv.seed is not None:
        set_seed(argv.seed)
        if DETERMINISTIC_MODE:
            dc.DETERMINISTIC_BASE_SEED = int(argv.seed)
    main()
