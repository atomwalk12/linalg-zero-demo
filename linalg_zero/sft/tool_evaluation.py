import logging
from typing import Any

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
from trl.trainer.model_config import ModelConfig

from linalg_zero.sft.hub import push_to_hub_revision

logger = logging.getLogger(__name__)


class EvaluationState:
    """Tracks evaluation state."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.sample: dict[str, Any] | None = None

        self.strict_format_match = 0.0
        self.partial_format_score = 0.0
        self.tool_parse_success = False
        self.generated_answer = None
        self.early_stop_reason: str | None = None


class DummyConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken
            # Also if you instantiate a new SFTConfig, the accelerator dist state will also be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            _ = push_to_hub_revision(dummy_config, extra_ignore_patterns=["*.pt"])  # don't push the optimizer states
