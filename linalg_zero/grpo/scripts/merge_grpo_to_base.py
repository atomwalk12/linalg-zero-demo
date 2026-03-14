from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class MergeableModel(Protocol):
    def save_pretrained(self, save_directory: str) -> None: ...

    def push_to_hub(self, repo_id: str) -> None: ...


class PeftAdapterModel(Protocol):
    def merge_and_unload(self) -> MergeableModel: ...


class PeftModelLoader(Protocol):
    @staticmethod
    def from_pretrained(
        model: object,
        model_id: str | Path,
        *,
        is_trainable: bool = ...,
    ) -> PeftAdapterModel: ...


def load_peft_model() -> PeftModelLoader:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the 'peft' package. Install it before merging a GRPO adapter."
        ) from exc
    return cast(PeftModelLoader, PeftModel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a GRPO PEFT adapter into its base model and optionally push the merged model to HF Hub.",
    )
    parser.add_argument(
        "--base-model",
        default="atomwalk12/LinalgZero-SFT",
        help="Base model used during GRPO training.",
    )
    parser.add_argument(
        "--adapter-model",
        default="atomwalk12/LinalgZero-GRPO",
        help="GRPO adapter repo or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/LinalgZero-GRPO-merged",
        help="Where to save the merged model locally.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional HF repo id to push the merged model to.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map to use while loading and merging.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    return _DTYPE_MAP[dtype_name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(args.dtype)
    peft_model_loader = load_peft_model()

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base model from {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    print(f"Loading adapter from {args.adapter_model}")
    adapter_model = peft_model_loader.from_pretrained(
        base_model,
        args.adapter_model,
        is_trainable=False,
    )

    print("Merging adapter into base model")
    merged_model = adapter_model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.push_to_hub:
        print(f"Pushing merged model to {args.push_to_hub}")
        merged_model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("Done.")


if __name__ == "__main__":
    main()
