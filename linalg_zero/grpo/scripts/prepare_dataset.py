import json
import logging
import re
from argparse import ArgumentParser
from typing import Any

from datasets import Dataset, DatasetDict, DownloadMode, load_dataset

from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.utils import (
    get_logger,
    get_representative_examples_indices,
    normalize_text,
    setup_logging,
)

# Log both to file and console
setup_logging(level=logging.INFO, include_timestamp=True)
logger = get_logger(__name__)


def load_datasets(src_train: str, src_test: str) -> DatasetDict:
    """Load datasets for GRPO training."""
    # Load training dataset (has solutions)
    logger.info(f"Loading train dataset from https://huggingface.co/datasets/{src_train}")
    train_dataset = load_dataset(src_train, split="train", download_mode=DownloadMode.FORCE_REDOWNLOAD)

    # Load base dataset to get validation and test splits
    logger.info(f"Loading base dataset from https://huggingface.co/datasets/{src_test}")
    base_dataset = load_dataset(src_test, download_mode=DownloadMode.FORCE_REDOWNLOAD)

    # Extract validation and test splits from the base dataset
    validation_dataset = base_dataset["validation"]
    test_dataset = base_dataset["test"]

    # Prepare results with proper three splits
    assert isinstance(train_dataset, Dataset)
    assert isinstance(validation_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    return DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})


def fix_think_tags(content: str) -> str:
    """Ensure exactly one newline after <think> and before </think>"""
    # First remove any existing whitespace around tags
    content = re.sub(r"<think>\s*", "<think>\n", content)
    content = re.sub(r"\s*</think>", "\n</think>", content)
    return content


def process_dataset_for_grpo(  # noqa: C901
    dataset: DatasetDict, normalize_unicode: bool, seed: int, per_category: int
) -> DatasetDict:
    """Process dataset specifically for GRPO training."""

    def ensure_tools(example: dict[str, Any]) -> dict[str, Any]:
        """Ensure tools field is present."""
        if "tools" not in example or example["tools"] is None:
            example["tools"] = get_tools()
        return example

    def normalize_query(example: dict[str, Any]) -> dict[str, Any]:
        """Normalize query text if unicode normalization is enabled."""
        if "query" in example:
            example["query"] = normalize_text(example["query"], normalize_unicode)
        return example

    def parse_messages_for_grpo(example: dict[str, Any]) -> dict[str, Any]:
        """Convert messages from JSON string to array and fix think tag formatting for GRPO."""
        if example.get("messages"):
            messages = json.loads(example["messages"])

            # Fix think tags in assistant messages and normalize content
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                    content = fix_think_tags(msg["content"])
                    msg["content"] = normalize_text(content, normalize_unicode)

            # Store processed messages for reference (optional)
            example["processed_messages"] = messages

        return example

    def process_train_split(train_data: Dataset) -> Dataset:
        """Process training dataset (has messages field with solutions)."""
        train_data = train_data.shuffle(seed=seed)
        train_data = train_data.map(parse_messages_for_grpo)
        train_data = train_data.map(ensure_tools)
        train_data = train_data.map(normalize_query)
        return train_data

    def process_eval_split(eval_data: Dataset) -> Dataset:
        """Process evaluation dataset (validation or test - no messages, just problems)."""
        eval_data = eval_data.shuffle(seed=seed)
        eval_data = eval_data.map(ensure_tools)
        eval_data = eval_data.map(normalize_query)
        eval_indices = get_representative_examples_indices(
            eval_data, per_category=per_category, include_remaining=False
        )
        eval_data = eval_data.select(eval_indices)
        return eval_data

    # The necessary columns for GRPO training
    keep_columns = [
        "query",
        "ground_truth",
        "stepwise_ground_truths",
        "tools",
    ]

    # Process all splits
    train_dataset = process_train_split(dataset["train"])
    validation_dataset = process_eval_split(dataset["validation"])
    test_dataset = process_eval_split(dataset["test"])

    # Remove unnecessary columns from all splits
    strip_cols = set(train_dataset.column_names) - set(keep_columns)
    if strip_cols:
        logger.info(f"Removing columns from train dataset: {strip_cols}")
        train_dataset = train_dataset.remove_columns(list(strip_cols))

    strip_cols = set(validation_dataset.column_names) - set(keep_columns)
    if strip_cols:
        logger.info(f"Removing columns from validation dataset: {strip_cols}")
        validation_dataset = validation_dataset.remove_columns(list(strip_cols))

    strip_cols = set(test_dataset.column_names) - set(keep_columns)
    if strip_cols:
        logger.info(f"Removing columns from test dataset: {strip_cols}")
        test_dataset = test_dataset.remove_columns(list(strip_cols))

    # Ensure all schemas align with the train dataset
    validation_dataset = validation_dataset.cast(train_dataset.features)
    test_dataset = test_dataset.cast(train_dataset.features)

    # Prepare results with all three splits
    assert isinstance(train_dataset, Dataset)
    assert isinstance(validation_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    return DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})


def validate_grpo_dataset(dataset: DatasetDict) -> None:
    """Validate that the dataset is properly formatted for GRPO training."""
    required_columns = ["query", "ground_truth", "stepwise_ground_truths", "tools"]

    for split_name, split_data in dataset.items():
        logger.info(f"Validating {split_name} split...")

        # Check required columns
        missing_cols = set(required_columns) - set(split_data.column_names)
        if missing_cols:
            raise ValueError(f"Missing required columns in {split_name}: {missing_cols}")

        # Validate sample entries
        if len(split_data) > 0:
            sample = split_data[0]

            # Check query field
            if not sample["query"] or not sample["query"].strip():
                raise ValueError(f"Empty query in {split_name} split")

            # Check ground_truth is valid JSON
            try:
                json.loads(sample["ground_truth"])
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid ground_truth JSON in {split_name}: {e}") from e

            # Check stepwise_ground_truths is valid JSON list
            try:
                stepwise = json.loads(sample["stepwise_ground_truths"])
                if not isinstance(stepwise, list):
                    raise TypeError(f"stepwise_ground_truths must be a list in {split_name}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid stepwise_ground_truths JSON in {split_name}: {e}") from e

            # Check tools field
            if not isinstance(sample["tools"], list):
                raise ValueError(f"Tools field must be a list in {split_name}")

    logger.info("*** Dataset validation completed successfully ***")


def prepare_debug(train: Dataset, validation: Dataset, test: Dataset, dataset_size: int) -> DatasetDict:
    """Prepare debug dataset with limited size."""
    train = train.select(range(min(dataset_size, len(train))))
    validation = validation.select(range(min(dataset_size, len(validation))))
    test = test.select(range(min(dataset_size, len(test))))
    return DatasetDict({"train": train, "validation": validation, "test": test})


def main(
    train_repo: str,
    test_repo: str,
    output_repo: str,
    push_to_hub: bool,
    normalize_unicode: bool,
    seed: int,
    per_category: int,
) -> None:
    """Main processing function for GRPO dataset preparation."""
    logger.info("*** Loading datasets for GRPO training ***")
    dataset = load_datasets(train_repo, test_repo)

    # Process for GRPO
    logger.info("*** Processing dataset for GRPO training ***")
    dataset = process_dataset_for_grpo(
        dataset, normalize_unicode=normalize_unicode, seed=seed, per_category=per_category
    )

    # Validate the processed dataset
    validate_grpo_dataset(dataset)

    # Push to hub
    if push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            dataset.push_to_hub(output_repo)
            logger.info(f"Successfully pushed dataset to https://huggingface.co/datasets/{output_repo}")
            logger.info("Dataset is now ready for GRPO training!")
        except Exception:
            logger.exception("Failed to push to hub")
    else:
        logger.info("Dataset processing completed. Use --push_to_hub to upload to HuggingFace Hub.")


if __name__ == "__main__":
    """Script entry point for GRPO dataset preparation."""
    parser = ArgumentParser(description="Prepare dataset for GRPO training")
    parser.add_argument(
        "--src_train", type=str, default="atomwalk12/linalgzero-distilled-clean", help="Source training dataset"
    )
    parser.add_argument("--src_test", type=str, default="atomwalk12/linalgzero", help="Source test dataset")
    parser.add_argument(
        "--output_repo", default="atomwalk12/linalgzero-grpo", type=str, help="Output repository name for GRPO dataset"
    )
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", help="Whether to push the dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--no_normalize_unicode",
        default=False,
        action="store_true",
        help="Disable Unicode NFKC normalization and minus-sign replacement during dataset prep",
    )
    parser.add_argument("--seed", default=20, type=int, help="Random seed for dataset shuffling")
    parser.add_argument(
        "--per_category",
        default=40,
        type=int,
        help="Number of representative examples per category for validation and test sets",
    )
    args = parser.parse_args()

    main(
        train_repo=args.src_train,
        test_repo=args.src_test,
        output_repo=args.output_repo,
        push_to_hub=args.push_to_hub,
        normalize_unicode=(not args.no_normalize_unicode),
        seed=args.seed,
        per_category=args.per_category,
    )
