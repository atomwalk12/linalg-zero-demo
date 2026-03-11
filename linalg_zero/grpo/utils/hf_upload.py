from __future__ import annotations

import os
import re
from pathlib import Path

import art
from art.utils.output_dirs import get_default_art_path, get_model_dir

from linalg_zero.grpo.types import LinAlgPolicyConfig

_HF_REPO_NAME_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]+")
_HF_UPLOAD_IGNORE_PATTERNS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.DS_Store",
)


def _sanitize_hf_repo_name(name: str) -> str:
    name = _HF_REPO_NAME_ALLOWED.sub("-", name).strip("-.")
    name = re.sub(r"-{2,}", "-", name)
    if not name:
        raise ValueError("Sanitized repo name is empty.")
    return name


def _should_push_experiment_to_hub() -> bool:
    if os.environ.get("HF_PUSH_EXPERIMENT", "").strip().lower() in {"0", "false", "no"}:
        return False
    return bool(os.environ.get("HF_HUB_NAMESPACE"))


def _get_rank() -> int:
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def push_experiment_dir_to_hf_sync(*, model: art.Model[LinAlgPolicyConfig]) -> None:
    """
    Upload `.art/<project>/models/<experiment>/` to the HF Hub.

    Enabled when `HF_HUB_NAMESPACE` is set and not explicitly disabled via `HF_PUSH_EXPERIMENT=0`.
    """
    if not _should_push_experiment_to_hub():
        print("[hf] Skipping upload: set `HF_HUB_NAMESPACE` to enable post-training push.")
        return

    if _get_rank() != 0:
        print("[hf] Skipping upload on non-zero rank.")
        return

    namespace = os.environ.get("HF_HUB_NAMESPACE")
    assert namespace is not None

    art_path = get_default_art_path()
    experiment_dir = Path(get_model_dir(model=model, art_path=art_path))
    if not experiment_dir.is_dir():
        print(f"[hf] Skipping upload: experiment dir not found: {experiment_dir}")
        return

    project = model.config.run_config.project
    experiment = model.name
    repo_name = _sanitize_hf_repo_name(f"{project}--{experiment}--experiment")
    repo_id = f"{namespace}/{repo_name}"

    private = os.environ.get("HF_REPO_PRIVATE", "").strip().lower() in {"1", "true", "yes"}

    try:
        from huggingface_hub import HfApi
    except Exception as e:  # pragma: no cover
        print("[hf] Skipping upload: missing dependency `huggingface_hub`.")
        print(f"[hf] Import error: {e}")
        return

    print(f"[hf] Uploading experiment dir: {experiment_dir} -> https://huggingface.co/{repo_id}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(experiment_dir),
        path_in_repo="",
        commit_message=f"Upload {project}/{experiment} experiment directory",
        ignore_patterns=list(_HF_UPLOAD_IGNORE_PATTERNS),
    )
    print("[hf] Upload complete.")
