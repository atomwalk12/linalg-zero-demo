"""
Runtime patches for third-party libraries used in GRPO training.

Python automatically imports `sitecustomize` at interpreter startup (via `site`),
including in multiprocessing "spawn" workers. Keep patches minimal and gated.
"""

from __future__ import annotations

import os
from typing import Any


def _patch_art_trainconfig_lr_alias() -> None:
    """
    ART's UnslothService warmup does:

        config.model_copy(update={"lr": 1e-9, "beta": 0.0, "kl_coef": 0.0})

    but `art.types.TrainConfig` uses the field name `learning_rate`, not `lr`.
    Without this patch, the warmup step can accidentally run at the *full*
    learning rate with `beta=0`, causing large KL/grad spikes on the first
    trainable batch after a service restart.
    """
    try:
        from art.types import TrainConfig
    except Exception:
        return

    original_model_copy = TrainConfig.model_copy

    # Avoid double-patching (important for interactive sessions / reloads).
    if getattr(original_model_copy, "__linalgzero_patched__", False):
        return

    def model_copy(self: Any, *, update: dict[str, Any] | None = None, deep: bool = False):
        if isinstance(update, dict):
            patched = dict(update)
            if "lr" in patched and "learning_rate" not in patched:
                # Keep both keys: some call sites may use `lr`, but ART TrainConfig uses
                # `learning_rate`. Unknown keys are ignored by pydantic, so retaining
                # `lr` is harmless and keeps the original update dict semantics.
                patched["learning_rate"] = patched["lr"]
            update = patched
        return original_model_copy(self, update=update, deep=deep)

    model_copy.__linalgzero_patched__ = True  # type: ignore[attr-defined]
    TrainConfig.model_copy = model_copy  # type: ignore[assignment]


def _install_art_unsloth_kl_guard_patch() -> None:
    """
    Patch ART's `art.unsloth.train` on import to support optional KL safety guards.

    This stays in-repo (no site-packages edits) and is applied via an import hook so
    it also affects subprocesses started by ART/Unsloth.
    """
    try:
        from linalg_zero.grpo.art_unsloth_kl_guard import install
    except Exception:
        return
    install()


if os.getenv("LINALGZERO_DISABLE_ART_PATCHES") != "1":
    _patch_art_trainconfig_lr_alias()
