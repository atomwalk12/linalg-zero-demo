from dataclasses import dataclass, field

from trl.scripts.utils import ScriptArguments as ScriptArgs
from trl.trainer.model_config import ModelConfig


@dataclass
class ScriptArguments(ScriptArgs):
    """
    Extended version of ScriptArguments with support for dataset mixtures.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "Training dataset name. Contains chain-of-thought solutions."}
    )
    eval_dataset_config: str | None = field(default=None, metadata={"help": "Evaluation dataset config."})
    take_n: int | None = field(default=None, metadata={"help": "Number of examples to take from the dataset."})


@dataclass
class SFTModelConfig(ModelConfig):
    enforce_eager: bool | None = field(default=None, metadata={"help": "Whether to enforce eager execution."})


@dataclass
class SFTRunConfig:
    add_special_tokens: bool = field(
        metadata={"help": "Whether to add special tokens to the model."},
    )

    early_stopping_patience: int = field(
        default=3, metadata={"help": "The number of epochs to wait before early stopping."}
    )

    early_stopping_threshold: float = field(
        default=0.0, metadata={"help": "Minimum improvement required to reset patience counter."}
    )

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: str | None = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: str | None = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    wandb_run_id: str | None = field(default=None, metadata={"help": {"The wandb run id."}})

    eval_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Max new tokens for evaluation callbacks (does not affect training)."},
    )

    max_seq_length: int | None = field(
        default=None,
        metadata={"help": "Max sequence length for evaluation callbacks (does not affect training)."},
    )

    gpu_memory_utilization: float | None = field(
        default=0.95, metadata={"help": "Fraction of GPU memory to be used by vLLM (0-1)"}
    )

    # Evaluation sampling
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of eval samples for periodic evaluations. None/-1 for full dataset."},
    )
    final_eval_max_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of eval samples for periodic evaluations. None/-1 for full dataset."},
    )


@dataclass
class DatasetGenerationConfig:
    """
    Data class that stores the dataset generation parameters.

    Args:
        dataset_name (str): The name of the dataset to generate.
    """

    dataset_name: str | None = field(
        metadata={"help": "Should be the name used to store the dataset on the Hugging Face Hub."},
    )


@dataclass
class LlamaCppServerConfig:
    """
    Data class that stores LlamaCPP server parameters with llama_cpp_ prefix.
    """

    def __post_init__(self) -> None:
        pass

    # Server parameters
    host: str = field(
        metadata={"help": "Host address to bind to"},
    )
    port: int = field(
        metadata={"help": "Port to listen on"},
    )
    n_ctx: int = field(
        metadata={"help": "Context size"},
    )
    split_mode: int = field(
        metadata={"help": "Split mode (0=none, 1=layer, 2=row)"},
    )

    # Model parameters
    n_gpu_layers: int = field(
        metadata={"help": "Number of GPU layers to offload"},
    )

    model: str = field(
        metadata={"help": "Model URL to download (GGUF format)"},
    )

    hf_pretrained_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Huggingface repository ID to ensure that the correct tokenizer is used."},
    )

    hf_model_repo_id: str | None = field(
        default=None,
        metadata={"help": "Path to the repository where the model is stored."},
    )


@dataclass
class VllmServerConfig:
    """
    Data class that stores vLLM server parameters with vllm_ prefix.
    """

    # Model parameters
    model: str = field(
        metadata={"help": "Model name (HuggingFace format)"},
    )

    # Server parameters
    host: str = field(
        metadata={"help": "Host address to bind to"},
    )
    port: int = field(
        metadata={"help": "Port to listen on"},
    )
    enable_auto_tool_choice: bool = field(
        metadata={"help": "Enable automatic tool choice"},
    )
    tool_call_parser: str = field(
        metadata={"help": "Tool call parser to use"},
    )
    chat_template: str | None = field(
        default=None,
        metadata={"help": "Chat template to use"},
    )
    quantization: str | None = field(
        default=None,
        metadata={"help": "Quantization to use"},
    )
    api_key: str = field(
        default="not-used",
        metadata={"help": "API key for authentication (use 'not-used' for local development)"},
    )

    # Memory / performance tuning parameters
    dtype: str | None = field(
        default=None,
        metadata={"help": "Computation dtype for model weights and activations (e.g., float16)"},
    )
    kv_cache_dtype: str | None = field(
        default=None,
        metadata={"help": "KV cache dtype (auto, fp8, fp8_e4m3, fp8_e5m2)"},
    )
    max_model_len: int | None = field(
        default=None,
        metadata={"help": "Maximum model context length (tokens)"},
    )
    max_num_seqs: int | None = field(
        default=None,
        metadata={"help": "Maximum number of concurrent sequences"},
    )
    gpu_memory_utilization: float | None = field(
        default=None,
        metadata={"help": "Fraction of GPU memory to be used by vLLM (0-1)"},
    )
    enforce_eager: bool | None = field(
        default=None,
        metadata={"help": "Disable CUDA graphs to reduce memory usage"},
    )
    swap_space: int | None = field(
        default=None,
        metadata={"help": "CPU swap space in GB per GPU for paging KV cache"},
    )
    max_num_batched_tokens: int | None = field(
        default=None,
        metadata={"help": "Limit number of tokens processed per batch (prefill)"},
    )
    tensor_parallel_size: int | None = field(
        default=None,
        metadata={"help": "Tensor parallelism degree"},
    )
    enable_chunked_prefill: bool | None = field(
        default=None,
        metadata={"help": "Enable chunked prefill to reduce peak prefill memory"},
    )

    # Model parameters
    reasoning_parser: str | None = field(
        default=None,
        metadata={"help": "Reasoning parser to use"},
    )


@dataclass
class DistillationConfig:
    """
    Data class that stores the distillation pipeline parameters.
    """

    # Dataset parameters
    dataset_name: str | None = field(
        metadata={"help": "HuggingFace dataset to load"},
    )

    # Prompt parameters
    prompt_column: str = field(
        metadata={"help": "Column name for prompt data"},
    )
    prompt_template: str = field(
        metadata={"help": "Template string for formatting prompts"},
    )

    # Generation parameters (non-defaults first)
    model_type: str | None = field(metadata={"help": "Model type for generation"})
    enable_reasoning: bool = field(metadata={"help": "Whether to enable thinking"})
    max_new_tokens: int = field(
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    num_generations: int = field(
        metadata={"help": "Number of generations per problem"},
    )

    # Processing parameters
    input_batch_size: int = field(
        metadata={"help": "Batch size for input processing"},
    )
    use_cache: bool = field(
        metadata={"help": "Whether to use cache for the pipeline. This can enable error recovery."},
    )

    timeout: int = field(
        metadata={"help": "Request timeout in seconds"},
    )
    retries: int = field(
        metadata={"help": "Number of retries for failed requests"},
    )

    # Output parameters
    hf_output_dataset: str | None = field(
        metadata={"help": "HuggingFace repo to push results to"},
    )
    argilla_output_dataset: str | None = field(
        metadata={"help": "Argilla dataset to push results to. This is used for manual annotation."},
    )
    private: bool = field(
        metadata={"help": "Whether to make the output dataset private when pushing to HF Hub"},
    )

    # Generation parameters
    n_turns: int = field(
        metadata={"help": "Number of turns to generate"},
    )
    min_successful_completions: int = field(
        default=-1,
        metadata={"help": "Minimum number of successful completions to generate"},
    )
    strip_think_prefix: bool = field(
        default=True,
        metadata={"help": "Whether to strip the think prefix from the conversation. This is needed for Qwen3 models."},
    )
    # Optional stopping sequences (must come after non-default fields)
    stop: list[str] | None = field(
        default=None,
        metadata={"help": "Stop sequences for generation (each string is a stop token)"},
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation"},
    )

    take_n: int | None = field(
        default=None,
        metadata={"help": "Number of examples to take from the dataset."},
    )

    structured_output: bool = field(
        default=False,
        metadata={"help": "Whether to use structured output"},
    )

    deterministic: bool = field(
        default=True,
        metadata={"help": "Make generation deterministic (temperature=0, top_p=1)"},
    )

    client_replicas: int | None = field(
        default=None,
        metadata={"help": "Number of client replicas for parallel processing"},
    )

    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Dataset config to use"},
    )
