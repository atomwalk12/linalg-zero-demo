import unsloth  # noqa: I001, F401
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback


def load_unmerged():
    path = "results/LinalgZero-SFT-LoRA/checkpoint-400-best"
    # path = "results/LinalgZero-SFT-LoRA-110/checkpoint-110"
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    model, _ = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B",
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=False,
    )

    model = PeftModel.from_pretrained(
        model,
        path,
        is_trainable=False,
    )

    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT-LoRA")
    # model.push_to_hub("atomwalk12/LinalgZero-SFT-LoRA")

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def load_merged():
    # Best models
    # Notice that best LoRA is checkpoint 400, while best merged is 300
    checkpoint_path = "results/LinalgZero-SFT/checkpoint-300-best"
    # checkpoint_path = "results/LinalgZero-SFT-merged"

    # checkpoint_path = "atomwalk12/LinalgZero-SFT-merged"
    # checkpoint_path = "atomwalk12/LinalgZero-SFT"

    # GRPO prep.
    # DONE
    # checkpoint_path = "results/LinalgZero-SFT-110/checkpoint-110"
    # checkpoint_path = "results/LinalgZero-SFT-105/checkpoint-105"

    # DONE
    # checkpoint_path = "results/LinalgZero-SFT-110-checkpoint-300/checkpoint-300"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    model, tok2 = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=False,
    )
    assert len(tok2) == len(tokenizer)

    # Best models
    model.push_to_hub("atomwalk12/LinalgZero-SFT")
    tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT")

    # model.push_to_hub("atomwalk12/LinalgZero-SFT-merged")
    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT-merged")

    # GRPO prep.
    # DONE
    # model.push_to_hub("atomwalk12/LinalgZero-SFT-105")
    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT-105")

    # DONE
    # model.push_to_hub("atomwalk12/LinalgZero-SFT-110")
    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT-110")

    # DONE
    # model.push_to_hub("atomwalk12/LinalgZero-SFT-110-checkpoint-300")
    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT-110-checkpoint-300")

    # model.push_to_hub("atomwalk12/LinalgZero-SFT")
    # tokenizer.push_to_hub("atomwalk12/LinalgZero-SFT")

    FastLanguageModel.for_inference(model)

    return model, tokenizer


model, tokenizer = load_unmerged()

eval_ds = load_dataset("atomwalk12/linalgzero-sft", split="test")  # or whatever split you used

cb = ToolCallingAccuracyCallback(
    model_name="atomwalk12/LinAlgZero-SFT-merged",
    dataset_name="atomwalk12/linalgzero",
    eval_dataset=eval_ds,
)
gen_config = cb.get_generation_config(max_new_tokens=800, tokenizer=tokenizer)


def generate_like_sft_eval(sample_idx: int = 0):
    sample = eval_ds[sample_idx]
    context = list(sample["messages"])
    tools = sample["tools"]
    print(f"Query is: {sample['messages'][-1]['content']}")

    prompt = tokenizer.apply_chat_template(
        context,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = prompt

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            **gen_config,
        )

    # Decode only the generated continuation (optional: mimic callback's decoding)
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[:, prompt_len:]
    text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
    print(text)
    return text


result = generate_like_sft_eval(0)
