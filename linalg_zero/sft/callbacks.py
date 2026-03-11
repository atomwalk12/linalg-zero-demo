from transformers.trainer_callback import (
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import ModelConfig, ScriptArguments
from trl.data_utils import DatasetDict

from linalg_zero.config.data import SFTRunConfig
from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback
from linalg_zero.sft.tool_evaluation import PushToHubRevisionCallback

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "tool_calling_accuracy": ToolCallingAccuracyCallback,
    "early_stopping": EarlyStoppingCallback,
}


def get_callbacks(
    train_config: SFTRunConfig, model_config: ModelConfig, script_args: ScriptArguments, dataset: DatasetDict
) -> list[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")

        # Different callbacks have different constructor signatures
        if callback_name == "tool_calling_accuracy":
            callbacks.append(
                CALLBACKS[callback_name](
                    model_name=model_config.model_name_or_path,
                    dataset_name=script_args.dataset_name,
                    eval_dataset=dataset[script_args.dataset_test_split],
                )
            )
        elif callback_name == "early_stopping":
            patience = train_config.early_stopping_patience
            threshold = train_config.early_stopping_threshold
            callbacks.append(
                CALLBACKS[callback_name](early_stopping_patience=patience, early_stopping_threshold=threshold)
            )
        else:
            callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
