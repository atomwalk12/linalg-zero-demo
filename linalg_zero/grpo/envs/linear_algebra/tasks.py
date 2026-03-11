from datasets import Dataset, load_dataset

from linalg_zero.grpo.types import Task


def load_tasks(hf_path: str, split: str, dev: bool = False) -> list[Task]:
    """Load tasks from HuggingFace dataset with optional development mode.

    Args:
        hf_path: HuggingFace dataset path
        split: Dataset split to load (train/test/validation)
        dev: If True, limit to first 10 samples for development

    Returns:
        List of Task instances converted from HuggingFace dataset entries
    """
    dataset = load_dataset(path=hf_path, split=split)

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")

    # Convert dataset entries to Task instances
    tasks = []
    for entry in dataset:
        task = Task.from_dataset_entry(entry)
        tasks.append(task)

    return tasks
