import unsloth  # noqa: F401
from unsloth import FastLanguageModel


def merge_lora_to_base_model() -> None:
    # adapter_path = "results/LinalgZero-SFT-LoRA-110/checkpoint-110"
    adapter_path = "results/LinalgZero-SFT-LoRA/checkpoint-400"

    # Load the checkpoint directly - Unsloth will detect it's a PEFT model
    # and load both base model + adapters with correct vocab size
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,  # Point directly to checkpoint
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=False,
        # resize_model_vocab=151680,
    )

    # Now merge and save
    model.save_pretrained_merged(
        # save_directory="results/LinalgZero-SFT-merged-110", tokenizer=tokenizer, save_method="merged_16bit"
        save_directory="results/LinalgZero-SFT-merged",
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )


if __name__ == "__main__":
    merge_lora_to_base_model()
