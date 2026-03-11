import os
from typing import Any

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import unsloth  # noqa: I001, F401
import logging
import sys

import transformers
from datasets import DatasetDict, load_dataset
from datasets.load import DownloadMode
from datasets.utils.logging import set_verbosity
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl.scripts.utils import TrlParser
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from linalg_zero.config.data import ScriptArguments, SFTModelConfig, SFTRunConfig
from linalg_zero.sft.callbacks import get_callbacks
from linalg_zero.sft.utils import (
    ensure_tokenizer_has_defaults,
    get_unsloth_model,
    init_wandb_training,
    load_merged_model_for_sft,
)
from linalg_zero.shared.utils import get_logger, setup_logging


def main(  # noqa: C901
    script_args: ScriptArguments, training_args: SFTRunConfig, trl_training_args: SFTConfig, model_args: SFTModelConfig
) -> None:
    """Main training function."""
    # Reproducibility
    set_seed(trl_training_args.seed)

    #################
    # Setup logging #
    #################
    # Log both to file and console
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    # Adjust script logging level based on the node logging level (main process or replica)
    log_level = trl_training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"TRL training parameters: {trl_training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if trl_training_args.output_dir and os.path.isdir(trl_training_args.output_dir):
        last_checkpoint = get_last_checkpoint(trl_training_args.output_dir)
    if last_checkpoint is not None and trl_training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Initialize wandb if requested
    if trl_training_args.report_to and "wandb" in trl_training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    logger.info(f"Loading dataset from {script_args.dataset_name}...")
    dataset = load_dataset(
        script_args.dataset_name, script_args.dataset_config, download_mode=DownloadMode.FORCE_REDOWNLOAD
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected dataset to be a DatasetDict, but got {type(dataset)}")

    # Model, tokenizer, dataset
    logger.info("Loading model and tokenizer...")
    if getattr(model_args, "use_peft", True):
        # Standard LoRA SFT on base model
        model, tokenizer = get_unsloth_model(model_args, training_args, trl_training_args, resume_path=last_checkpoint)
    else:
        # Light touch-up on a merged model: train only I/O layers if requested
        max_seq_len = training_args.max_seq_length or trl_training_args.max_seq_length
        model, tokenizer = load_merged_model_for_sft(
            model_path=model_args.model_name_or_path,
            max_seq_length=max_seq_len,
            dtype=None,
            train_io_only=True,
            add_special_tokens=training_args.add_special_tokens,
        )

    # Ensure pad token and padding side are set consistently for SFT
    ensure_tokenizer_has_defaults(tokenizer, model)

    def ensure_text(x: dict[str, Any]) -> dict[str, Any]:
        x["text"] = tokenizer.apply_chat_template(x["messages"], tools=x["tools"], tokenize=False)
        return x

    def formatting_prompts_func(examples):
        convos = examples["messages"]  # List of 1000 conversations
        tools = examples.get("tools", None)  # List of 1000 tool specs

        texts = []
        for i, convo in enumerate(convos):
            example_tools = tools[i] if tools and isinstance(tools, list) else tools

            text = tokenizer.apply_chat_template(
                convo,
                tools=example_tools,  # Pass tools[i] for the i-th conversation
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    ##############################
    # Initialize the SFT Trainer #
    ##############################
    trl_training_args.max_eval_samples = training_args.max_eval_samples
    trl_training_args.eval_max_new_tokens = training_args.eval_max_new_tokens

    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if trl_training_args.eval_strategy != "no" else None),
        args=trl_training_args,
        callbacks=get_callbacks(training_args, model_args, script_args, dataset),
    )

    #################
    # Training loop #
    #################
    logger.info("*** Starting Training ***")
    checkpoint = None
    if trl_training_args.resume_from_checkpoint is not None:
        checkpoint = trl_training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed with an unexpected error")
        raise

    ####################################
    # Save model and create model card #
    ####################################
    logger.info("*** Saving Model ***")
    try:
        # Align the model's generation config with the tokenizer's eos token
        # to avoid unbounded generation in the transformers `pipeline()` function
        if trainer.model is not None and trainer.model.generation_config is not None:
            trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
            assert trainer.model.generation_config.pad_token_id == tokenizer.pad_token_id, "Pad token ID mismatch"

        # Restore k,v cache for fast inference before saving
        if trainer.model is not None:
            trainer.model.config.use_cache = True

        trainer.save_model(trl_training_args.output_dir)
        logger.info(f"Model saved to {trl_training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["linalg-zero", "sft", "tool-use", "linear-algebra"],
            "model_name": model_args.model_name_or_path,
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)

    except Exception:
        logger.exception("Failed to save model")
        raise

    ############
    # Evaluate #
    ############
    if trl_training_args.do_eval:
        logger.info("*** Final Evaluation on Full Dataset ***")
        try:
            # Temporarily override max_eval_samples to evaluate on full dataset
            original_max_eval_samples = getattr(trl_training_args, "max_eval_samples", None)
            trl_training_args.max_eval_samples = training_args.final_eval_max_samples

            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            logger.info("Evaluation completed successfully!")

            # Restore original value
            trl_training_args.max_eval_samples = original_max_eval_samples

        except Exception:
            logger.exception("Evaluation failed")

    ###############
    # Push to hub #
    ###############
    if trl_training_args.push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            trainer.push_to_hub(**kwargs)
            logger.info("Successfully pushed model to HuggingFace Hub!")
        except Exception:
            logger.exception("Failed to push to hub")


if __name__ == "__main__":
    """Script entry point for SFT training."""
    if "--config" not in sys.argv:
        sys.argv.append("--config")
        sys.argv.append("linalg_zero/config/sft/qwen2.5-3B/production_merged.yaml")
        # sys.argv.append("linalg_zero/config/sft/qwen2.5-3B/production_instruct.yaml")
        # sys.argv.append("linalg_zero/config/sft/qwen2.5-3B/production.yaml")

    parser = TrlParser([ScriptArguments, SFTRunConfig, SFTConfig, SFTModelConfig])
    script_args, training_args, trl_training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, trl_training_args, model_args)
