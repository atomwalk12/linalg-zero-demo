"""
Upload a tau-bench experiment directory (or just its archived best checkpoints) to the Hugging Face Hub.

Default scope is the entire experiment directory:
  `.art/<project>/models/<experiment>/`

The legacy "best_models" scope uploads only archived checkpoints under:
  `.art/<project>/models/<experiment>/best_models/<split>/<step>/`

Usage (dry-run first):
  uv run python linalg_zero/grpo/tau-bench/push_best_models_to_hf.py \
    --project linalgzero-grpo \
    --experiment linalgzero-grpo-001 \
    --hub-namespace atomwalk12 \
    --dry-run

Auth:
  - Run `huggingface-cli login` beforehand, or set `HF_TOKEN` in your environment.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BestModelCheckpoint:
    model: str
    split: str
    step: int
    path: Path


_HF_REPO_NAME_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]+")
_DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.DS_Store",
)


def _sanitize_hf_repo_name(name: str) -> str:
    # Hugging Face repo names allow: letters, numbers, "-", "_", "."
    # Convert runs of other chars to a single "-".
    name = _HF_REPO_NAME_ALLOWED.sub("-", name).strip("-.")
    name = re.sub(r"-{2,}", "-", name)
    if not name:
        raise ValueError("Sanitized repo name is empty; choose a different template.")
    return name


def _discover_best_model_checkpoints(best_models_dir: Path) -> list[BestModelCheckpoint]:
    checkpoints: list[BestModelCheckpoint] = []
    if not best_models_dir.is_dir():
        return checkpoints

    for split_dir in sorted(p for p in best_models_dir.iterdir() if p.is_dir()):
        split = split_dir.name
        for step_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if not step_dir.name.isdigit():
                continue
            checkpoints.append(
                BestModelCheckpoint(
                    model=best_models_dir.parent.name,
                    split=split,
                    step=int(step_dir.name),
                    path=step_dir,
                )
            )
    return checkpoints


def _format_repo_id(
    *, namespace: str, project: str, experiment: str, scope: Literal["experiment", "best_models"]
) -> str:
    suffix = "experiment" if scope == "experiment" else "best-models"
    repo_name = _sanitize_hf_repo_name(f"{project}--{experiment}--{suffix}")
    return f"{namespace}/{repo_name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Push one experiment directory to HF Hub.")
    parser.add_argument("--project", required=True, help="Project directory under `.art/` (e.g. linalgzero-grpo).")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment/model directory name under `.art/<project>/models/` (e.g. linalgzero-grpo-001).",
    )
    parser.add_argument("--hub-namespace", required=True, help="HF namespace/user/org (e.g. atomwalk12).")
    parser.add_argument(
        "--scope",
        choices=["experiment", "best_models"],
        default="experiment",
        help="What to upload: the whole experiment directory (default) or only `best_models/` checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned uploads without pushing.")
    args = parser.parse_args()

    experiment_dir = Path(".art") / args.project / "models" / args.experiment
    if not experiment_dir.is_dir():
        print(f"Not found: {experiment_dir}")
        print("Expected layout: .art/<project>/models/<experiment>/")
        return 1

    scope: Literal["experiment", "best_models"] = args.scope
    repo_id = _format_repo_id(
        namespace=args.hub_namespace,
        project=args.project,
        experiment=args.experiment,
        scope=scope,
    )

    print(f"Repo: https://huggingface.co/{repo_id}")
    if scope == "experiment":
        print("Planned upload:")
        print(f"- {experiment_dir} -> (repo root)")
    else:
        best_models_dir = experiment_dir / "best_models"
        if not best_models_dir.is_dir():
            print(f"Not found: {best_models_dir}")
            print("Expected layout: .art/<project>/models/<experiment>/best_models/<split>/<step>/")
            return 1

        checkpoints = _discover_best_model_checkpoints(best_models_dir)
        if not checkpoints:
            print(f"No checkpoints found under {best_models_dir}/*/*")
            return 1

        planned: list[tuple[BestModelCheckpoint, str]] = [
            (c, f"best_models/{c.split}/{c.step:04d}") for c in checkpoints
        ]
        print("Planned uploads:")
        for ckpt, path_in_repo in planned:
            print(f"- {ckpt.path} -> {path_in_repo}/")

    if args.dry_run:
        return 0

    try:
        from huggingface_hub import HfApi
    except Exception as e:  # pragma: no cover
        print("Missing dependency: huggingface_hub. Install it in your environment to use this script.")
        print(f"Import error: {e}")
        return 1

    api = HfApi()

    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    if scope == "experiment":
        print(f"\nUploading {experiment_dir} -> {repo_id}:(repo root)")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(experiment_dir),
            path_in_repo="",
            commit_message=f"Upload {args.project}/{args.experiment} experiment directory",
            ignore_patterns=list(_DEFAULT_IGNORE_PATTERNS),
        )
    else:
        best_models_dir = experiment_dir / "best_models"
        checkpoints = _discover_best_model_checkpoints(best_models_dir)
        planned = [(c, f"best_models/{c.split}/{c.step:04d}") for c in checkpoints]

        for ckpt, path_in_repo in planned:
            print(f"\nUploading {ckpt.path} -> {repo_id}:{path_in_repo}/")
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(ckpt.path),
                path_in_repo=path_in_repo,
                commit_message=f"Upload {args.project}/{args.experiment}:{path_in_repo}/",
                ignore_patterns=list(_DEFAULT_IGNORE_PATTERNS),
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
