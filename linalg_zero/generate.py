import argparse
import logging
from pathlib import Path

import linalg_zero.generator.difficulty_config as dc
from linalg_zero.generator.analysis.utils import (
    compute_stepwise_value_statistics,
    print_statistics_summary,
)
from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.difficulty_config import DETERMINISTIC_MODE
from linalg_zero.generator.models import DifficultyCategory, Question, Topic
from linalg_zero.generator.registry import create_default_registry, create_optimized_registry
from linalg_zero.generator.utils import (
    check_constraints,
    convert_to_dataset_splits,
    load_entropy_settings,
    print_dataset,
    print_split_statistics,
    set_seed,
    verify_dataset,
)
from linalg_zero.shared.utils import get_log_file_path, get_logger, push_to_hub, setup_logging


def main(
    push_dataset: bool, use_optimized_registry: bool, dataset_name: str, n_one: int, n_two: int, n_three: int
) -> None:  # pragma: no cover
    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=False)
    logger = get_logger(__name__)

    logger.info("Linear Algebra Dataset Generator")
    config_path = f"{Path(__file__).parent}/generator/config/gen_properties.json"

    # Create registry (either default or optimized)
    if use_optimized_registry:
        registry = create_optimized_registry(config_path=config_path)
        logger.info("Using optimized entropy settings from analysis results")
    else:
        registry = create_default_registry()

    logger.info("Available topics: %s", registry.list_topics())

    # -----------------------------------------------
    # Generate and display the linear algebra dataset
    # -----------------------------------------------
    def matrix_only_validator(question: Question) -> bool:
        # A filter to only include questions that satisfy specific conditions
        return len(question.answer) > 0

    generator = DatasetGenerator(
        topic=Topic.LINEAR_ALGEBRA, validator_factory=matrix_only_validator, registry=registry
    )

    # Generate custom amounts per difficulty category
    # Easy: 3000, Medium: 2000, Hard: 1000 (total: 6000)
    dataset = generator.generate_exact_for_categories(
        requests={
            DifficultyCategory.ONE_TOOL_CALL: n_one,
            DifficultyCategory.TWO_TOOL_CALLS: n_two,
            DifficultyCategory.THREE_TOOL_CALLS: n_three,
        }
    )
    statistics = compute_stepwise_value_statistics(dataset)
    print_dataset(dataset)
    print_statistics_summary(statistics)
    verify_dataset(dataset)
    if use_optimized_registry:
        config = load_entropy_settings(config_path)
        check_constraints(dataset, config, statistics)

    # Create stratified splits by difficulty for balanced evaluation
    splits = convert_to_dataset_splits(
        dataset,
        test_size=0.1,
        val_size=0.1,
        seed=argv.seed or 42,
        stratify_by="difficulty",
    )
    print_split_statistics(splits)
    if push_dataset:
        push_to_hub(splits, dataset_name, private=False, config_path=config_path)

    # --------------------------------------------------
    # This is an example on generating other topic types
    # --------------------------------------------------
    # arithmetic_generator = DatasetGenerator(topic="arithmetic")
    # arithmetic_questions = arithmetic_generator.generate_dataset(num_questions=2)
    # print_dataset(arithmetic_questions)
    logger.info(f"Log file path: {get_log_file_path()}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_dataset", action="store_true", default=False)
    parser.add_argument("--dataset_name", type=str, default="atomwalk12/linalgzero")
    parser.add_argument(
        "--use_optimized_registry",
        action="store_true",
        default=True,
        help="Use optimized entropy settings from analysis results for dataset generation",
    )
    parser.add_argument("--n_one", type=int, default=700, help="Per-generator 1-step samples")
    parser.add_argument("--n_two", type=int, default=900, help="Per-generator 2-step samples")
    parser.add_argument("--n_three", type=int, default=600, help="Per-generator 3-step samples")
    parser.add_argument("-scale", type=int, default=1.1, help="Scale the dataset by a factor")
    argv = parser.parse_args()
    if argv.seed is not None:
        set_seed(argv.seed)
        if DETERMINISTIC_MODE:
            # Let CLI seed control per-question reseed base when deterministic
            # Importing module and setting its global is sufficient
            dc.DETERMINISTIC_BASE_SEED = int(argv.seed)

    n_one = int(argv.n_one * argv.scale)
    n_two = int(argv.n_two * argv.scale)
    n_three = int(argv.n_three * argv.scale)

    main(argv.push_dataset, argv.use_optimized_registry, argv.dataset_name, n_one, n_two, n_three)
