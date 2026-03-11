import logging
import os
from pathlib import Path

import torch
from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_kbit_device_map, get_quantization_config
from unsloth import FastLanguageModel
from unsloth.tokenizer_utils import SFTConfig

from linalg_zero.config.data import ScriptArguments, SFTModelConfig, SFTRunConfig
from linalg_zero.shared.system_prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
)

logger = logging.getLogger(__name__)


def is_using_deepspeed() -> bool:
    """Check if DeepSpeed is being used via environment variables"""
    return (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        or "deepspeed" in os.environ.get("ACCELERATE_CONFIG_FILE", "").lower()
    )


def ensure_tokenizer_has_defaults(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        assert model.generation_config is not None, "Generation config is not set"
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def init_wandb_training(training_args: SFTRunConfig) -> None:
    """Initialize Weights & Biases for training logging."""
    try:
        # Set environment variables for wandb
        if training_args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = training_args.wandb_entity
        if training_args.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = training_args.wandb_project
        if training_args.wandb_run_group is not None:
            os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
        if training_args.wandb_run_id is not None:
            os.environ["WANDB_RUN_ID"] = training_args.wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"

        logger.info("Set wandb environment variables from training args")

    except Exception:
        logger.exception("Failed to initialize wandb environment")


def get_tokenizer(model_args: ModelConfig, training_args: SFTRunConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def load_model_for_evaluation(
    model_path: str,
    max_seq_length: int = 2048,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a trained model for evaluation/inference.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def add_special_tokens_and_resize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> bool:
    """
    Add special reasoning/tool-calling tokens to tokenizer and resize model embeddings if needed.

    Returns True if any new tokens were added (regardless of whether a resize was needed),
    False if no new tokens were added.
    """
    special_tags = [THINK_OPEN, THINK_CLOSE, TOOL_CALL_OPEN, TOOL_CALL_CLOSE, ANSWER_OPEN, ANSWER_CLOSE]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tags})

    if num_added and num_added > 0:
        tok_vocab = len(tokenizer)
        model_vocab = model.get_input_embeddings().weight.size(0)

        # Mark embeddings as trainable so new token rows can be updated.
        model._need_to_train_embeddings = True

        if tok_vocab > model_vocab:
            pad_to_multiple_of = 128
            logger.info(
                "Added %s special tokens; resizing embeddings %s -> %s (padded to multiple of %s).",
                num_added,
                model_vocab,
                tok_vocab,
                pad_to_multiple_of,
            )
            model.resize_token_embeddings(tok_vocab, pad_to_multiple_of=pad_to_multiple_of)
            return True
        else:
            logger.info(
                "Added %s special tokens but model vocab (%s) already >= tokenizer vocab (%s); "
                "skipping embedding resize.",
                num_added,
                model_vocab,
                tok_vocab,
            )
            return True
    else:
        logger.info("No new special tokens added (tokens likely already present). Skipping resize.")
        return False


def load_merged_model_for_sft(
    model_path: str,
    max_seq_length: int = 2048,
    dtype: torch.dtype | None = None,
    train_io_only: bool = False,
    add_special_tokens: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a merged (non-LoRA) model for a light SFT touch-up.

    - `model_path` should point to the merged checkpoint directory
      (e.g. \"results/LinalgZero-SFT-merged\").
    - If `train_io_only` is True, all parameters are frozen except:
        * input embeddings (`embed_tokens`)
        * output head (`lm_head` / output embeddings)
    - If `add_special_tokens` is True, adds reasoning/tool-calling tokens and resizes embeddings
    """
    # Load with Unsloth wrapper for consistent config handling
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    # Make sure pad / eos are wired correctly before training
    ensure_tokenizer_has_defaults(tokenizer, model)

    # Optionally add special tokens and resize embeddings
    if add_special_tokens:
        add_special_tokens_and_resize(model, tokenizer)

    if train_io_only:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze embeddings
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True

        # Unfreeze LM head / output embeddings
        output_layer = getattr(model, "lm_head", None)
        if output_layer is None:
            output_layer = model.get_output_embeddings()
        for param in output_layer.parameters():
            param.requires_grad = True

    return model, tokenizer


def get_unsloth_model(
    model_args: SFTModelConfig,
    training_args: SFTRunConfig,
    trl_training_args: SFTConfig,
    resume_path: str | None = None,
    use_vllm: bool = False,
) -> tuple[FastLanguageModel, PreTrainedTokenizer]:
    """Fetch the model and optimizer for training."""
    # Checkpoint loading is handled by the Trainer via `resume_from_checkpoint`.
    # We keep `resume_path` for API compatibility but do not use it here.
    if resume_path is not None:
        logger.info(
            "Received resume_path=%s in get_unsloth_model, but checkpoint loading is "
            "handled by the Trainer. Ignoring this argument.",
            resume_path,
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        max_lora_rank=model_args.lora_r,
        # enforce_eager=model_args.enforce_eager,
        fast_inference=use_vllm,
        gpu_memory_utilization=training_args.gpu_memory_utilization,
    )

    # Add special tokens and resize embeddings
    has_added_tokens = False
    if training_args.add_special_tokens:
        has_added_tokens = add_special_tokens_and_resize(model, tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        modules_to_save=["embed_tokens", "lm_head"] if has_added_tokens else None,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        ensure_weight_tying=True,
    )

    if trl_training_args.chat_template_path is not None:
        template_path = Path(trl_training_args.chat_template_path)
        tokenizer.chat_template = template_path.read_text()

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    has_user_template = training_args.chat_template is not None
    has_config_template = trl_training_args.chat_template_path is not None

    assert has_user_template ^ has_config_template, (
        "Exactly one of tokenizer.chat_template or chat_template_path must be set, not both or neither"
    )

    return model, tokenizer


def get_model(model_args: ModelConfig, training_args: SFTRunConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = model_args.torch_dtype
    if torch_dtype not in (None, "auto"):
        assert torch_dtype is not None
        torch_dtype = getattr(torch, torch_dtype)
    quantization_config = get_quantization_config(model_args)

    using_deepspeed = is_using_deepspeed()
    device_map = None
    if quantization_config is not None and not using_deepspeed:
        device_map = get_kbit_device_map()
        logger.info(f"Setting device_map: {device_map}")
    else:
        # Device map is not compatible with quantization and deepspeed ZeRO-3``
        logger.info("Not setting device_map (DeepSpeed detected or no quantization)")

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }
    if model_args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be set for loading the model")

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    return model


def load_dataset(args: ScriptArguments) -> DatasetDict:
    """Load the dataset produced during the distillation step, removing unnecessary columns for SFT."""

    def remove_redundant_columns(dataset: DatasetDict) -> DatasetDict:
        """Remove columns from a dataset."""
        if dataset.column_names:
            splits = dict(dataset.column_names.items())

            # Remove any redundant columns not using during SFT training. Only 'tools' and 'messages' are relevant.
            dataset = dataset.remove_columns([
                col
                for split in splits.values()
                if split is not None
                for col in split
                if col not in ["tools", "messages"]
            ])
        return dataset

    dataset = hf_load_dataset(args.dataset_name, args.dataset_config)

    if args.take_n is not None:
        dataset = dataset.select(range(args.take_n))

    # Only the ["messages", "tools"] columns are relevant for SFT
    return remove_redundant_columns(dataset)
