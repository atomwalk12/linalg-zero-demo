import logging
from concurrent.futures import Future
from typing import Any, cast

from huggingface_hub import (
    create_branch,
    create_repo,
    list_repo_commits,
    upload_folder,
)
from trl import GRPOConfig, SFTConfig

logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig | GRPOConfig | object, extra_ignore_patterns: list[str]) -> Future:
    """Pushes the model to branch on a Hub repo."""
    # Access attributes dynamically since we might get a DummyConfig
    hub_model_id = getattr(training_args, "hub_model_id", None)
    hub_model_revision = getattr(training_args, "hub_model_revision", None)
    output_dir = getattr(training_args, "output_dir", None)

    if not hub_model_id:
        raise ValueError("hub_model_id is required")
    if not hub_model_revision:
        raise ValueError("hub_model_revision is required")
    if not output_dir:
        raise ValueError("output_dir is required")

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=hub_model_id,
        branch=hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {hub_model_revision}...")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=hub_model_id,
        folder_path=output_dir,
        revision=hub_model_revision,
        commit_message=f"Add {hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {hub_model_revision} successfully!")

    return cast(Future[Any], future)
