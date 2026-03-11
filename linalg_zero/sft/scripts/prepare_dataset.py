import json
import logging
from argparse import ArgumentParser
from typing import Any

from datasets import Dataset, DatasetDict, DownloadMode, load_dataset

from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.system_prompts import get_sft_system_prompt
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
    """Load datasets"""
    # Load
    logger.info(f"Loading train dataset from https://huggingface.co/datasets/{src_train}")
    train_dataset = load_dataset(src_train, split="train", download_mode=DownloadMode.FORCE_REDOWNLOAD)

    logger.info(f"Loading validation dataset from https://huggingface.co/datasets/{src_test}")
    test_dataset = load_dataset(src_test, split="validation")

    # Prepare results
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    return DatasetDict({"train": train_dataset, "validation": test_dataset})


def process_dataset(dataset: DatasetDict, normalize_unicode: bool, per_category: int, seed: int) -> DatasetDict:
    """Load and process dataset for SFT training."""

    # The necessary columns for SFT
    keep_columns = [
        "tools",
        "messages",
        "ground_truth",
        "stepwise_ground_truths",
    ]

    def _normalize_messages(example: dict[str, Any]) -> dict[str, Any]:
        if not normalize_unicode:
            return example
        msgs = example.get("messages", [])
        for m in msgs:
            if isinstance(m, dict) and "content" in m:
                m["content"] = normalize_text(m["content"], normalize_unicode)
        example["messages"] = msgs
        return example

    # Add missing columns (messages & tools)
    def ensure_messages(example: dict[str, Any]) -> dict[str, Any]:
        example["messages"] = [
            {"role": "system", "content": normalize_text(get_sft_system_prompt(), normalize_unicode)},
            {"role": "user", "content": normalize_text(example["query"], normalize_unicode)},
        ]
        return _normalize_messages(example)

    def ensure_tools(example: dict[str, Any]) -> dict[str, Any]:
        if "tools" not in example or example["tools"] is None:
            example["tools"] = get_tools()
        return example

    def parse_messages(example: dict[str, Any]) -> dict[str, Any]:
        """Convert messages from JSON string to array and replace system prompt"""
        example["messages"] = json.loads(example["messages"])

        # Replace the system prompt with the SFT system prompt
        if example["messages"] and example["messages"][0]["role"] == "system":
            example["messages"][0]["content"] = normalize_text(get_sft_system_prompt(), normalize_unicode)

        return _normalize_messages(example)

    train_dataset = dataset["train"]
    train_dataset = train_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(parse_messages)

    test_dataset = dataset["validation"]
    test_dataset = test_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.map(ensure_messages)
    test_dataset = test_dataset.map(ensure_tools)
    indices = get_representative_examples_indices(test_dataset, per_category=per_category, include_remaining=False)
    test_dataset = test_dataset.select(indices)

    # Ensure only relevant columns are preserved
    strip_cols = set(train_dataset.column_names) - set(keep_columns)
    train_dataset = train_dataset.remove_columns(strip_cols)

    strip_cols = set(test_dataset.column_names) - set(keep_columns)
    test_dataset = test_dataset.remove_columns(strip_cols)

    # Ensure the two schemas align (in tools field)
    test_dataset = test_dataset.cast(train_dataset.features)

    # Prepare results
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def prepare_debug(train: Dataset, validation: Dataset, dataset_size: int) -> DatasetDict:
    train = train.select(range(dataset_size))
    validation = validation.select(range(dataset_size))
    return DatasetDict({"train": train, "validation": validation})


def main(output_repo: str, push_to_hub: bool, normalize_unicode: bool, per_category: int, seed: int) -> None:
    """Main processing function."""
    # Load
    train_repo = "atomwalk12/linalgzero-distilled-clean"
    test_repo = "atomwalk12/linalgzero"

    logger.info("*** Loading datasets ***")
    dataset = load_datasets(train_repo, test_repo)

    # Process
    logger.info("*** Processing dataset ***")
    dataset = process_dataset(dataset, normalize_unicode=normalize_unicode, per_category=per_category, seed=seed)

    # Push to hub
    if push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            dataset.push_to_hub(output_repo)
            logger.info(f"Successfully pushed dataset to https://huggingface.co/datasets/{output_repo}")
        except Exception:
            logger.exception("Failed to push to hub")


if __name__ == "__main__":
    """Script entry point for SFT training."""
    parser = ArgumentParser()
    parser.add_argument("--output_repo", default="atomwalk12/linalgzero-sft", type=str, help="Output repository name")
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", help="Whether to push the dataset to HuggingFace"
    )
    parser.add_argument(
        "--no_normalize_unicode",
        default=False,
        action="store_true",
        help="Disable Unicode NFKC normalization and minus-sign replacement during dataset prep",
    )
    parser.add_argument("--per_category", default=40, type=int, help="Number of representative examples per category")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for dataset shuffling")
    args = parser.parse_args()

    main(
        output_repo=args.output_repo,
        push_to_hub=args.push_to_hub,
        normalize_unicode=(not args.no_normalize_unicode),
        per_category=args.per_category,
        seed=args.seed,
    )
