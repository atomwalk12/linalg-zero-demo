from __future__ import annotations

import os
import shutil

import art
from art.utils.output_dirs import get_default_art_path, get_model_dir, get_step_checkpoint_dir

from linalg_zero.grpo.types import LinAlgPolicyConfig

BEST_CHECKPOINT_METRIC = "val/optimal_trajectory"


async def delete_checkpoints_keep_best(
    model: art.TrainableModel[LinAlgPolicyConfig],
    *,
    best_checkpoint_metric: str = BEST_CHECKPOINT_METRIC,
) -> None:
    try:
        await model.delete_checkpoints(best_checkpoint_metric=best_checkpoint_metric)
    except Exception as e:
        print(f"Warning: delete_checkpoints failed for metric '{best_checkpoint_metric}': {e}")
        await model.delete_checkpoints()


def archive_checkpoint(*, model: art.Model[LinAlgPolicyConfig], step: int, split: str) -> None:
    """
    Copy the checkpoint directory for `step` into a persistent archive directory under the model output dir.

    This preserves all checkpoints that were evaluated, even if we later prune the main `checkpoints/` directory.
    """
    try:
        art_path = get_default_art_path()
        model_dir = get_model_dir(model=model, art_path=art_path)
        src = get_step_checkpoint_dir(model_dir, step)
        if not os.path.isdir(src):
            print(f"Warning: checkpoint dir not found for archiving: {src}")
            return

        dst_base = os.path.join(model_dir, "best_models", split)
        os.makedirs(dst_base, exist_ok=True)
        dst = os.path.join(dst_base, f"{step:04d}")
        if os.path.exists(dst):
            return
        shutil.copytree(src, dst)
    except Exception as e:
        print(f"Warning: failed to archive checkpoint at step {step} ({split}): {e}")
