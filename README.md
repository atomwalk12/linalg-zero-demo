[![Release](https://img.shields.io/github/v/release/atomwalk12/linalg-zero)](https://img.shields.io/github/v/release/atomwalk12/linalg-zero)
[![Build status](https://img.shields.io/github/actions/workflow/status/atomwalk12/linalg-zero/main.yml?branch=main)](https://github.com/atomwalk12/linalg-zero/actions/workflows/main.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Linalg-Zero

Check out the poster [here](docs/poster.pdf) and the paper [here](docs/report.pdf).

<img width="14173" height="8504" alt="poster" src="https://github.com/user-attachments/assets/b7019c34-8dcf-45a3-830e-050a822e9ff0" />


## Overview

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#main-phases">Main Phases</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#artifacts">Artifacts</a></li>
    <li><a href="#reproducibility">Reproducibility</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

This repository offers tools for generating a linear algebra problem dataset and training an open-source base model (i.e. Qwen2.5-3B), aiming to explore planning and tool use using SFT and RL, distinct from Deepseek-R1's primary emphasis on reasoning.

The project is simple by design and mostly consists of:

- `linalg_zero/`: contains the scripts to train models as well as generate synthetic data:
    - `generate.py`: generates the linear algebra dataset and splits.
    - `distillation.py`: runs the distillation pipeline to create multi-turn tool-use data.
    - `sft_train.py`: performs a simple SFT of a model on a dataset.
    - `grpo_train.py`: trains a model with GRPO on a given dataset.
- `Makefile`: contains easy-to-run commands for the dataset and training workflows using previous scripts.

## Main Phases

We use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a loose guide, but the project phases are:

* Step 1: generate a linear algebra dataset with controlled difficulty and tool-call metadata.
* Step 2: distill multi-turn tool-use data from a teacher model.
* Step 3: SFT the base model on the dataset to teach the tool-calling format.
* Step 4: GRPO fine-tune on the tool-use tasks, using a curriculum.


## Installation

We use `uv` as the dependency management tool.
First, to install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

To run the experiments install the dependencies using:

* For generation/distillation: `make install-data-gen`
* For SFT: `make install-sft`
* For RL: `make install-grpo`

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

## Quickstart

After installing dependencies above, run the commands below. For modifications, see the config files.

```shell
# Phase 1: Generate dataset
uv run python linalg_zero/generate.py --dataset_name atomwalk12/linalgzero --push_dataset

# Phase 2: Distillation (setup once)
cp linalg_zero/config/distillation/env.example.sh env.sh
# Edit env.sh to set HF_TOKEN and ARGILLA_API_KEY.
source env.sh

# Terminal A
uv run python linalg_zero/distillation/launch_server.py --config linalg_zero/config/distillation/vllm_qwen3_32b.yaml

# Terminal B (new terminal; source env.sh again)
source env.sh
uv run python linalg_zero/distillation.py --config linalg_zero/config/distillation/vllm_qwen3_32b.yaml

# Phase 3: SFT
uv run python linalg_zero/sft_train.py --config linalg_zero/config/sft/qwen2.5-3B/lora.yaml

# Phase 4: GRPO
uv run python linalg_zero/grpo_train.py --config-name runpod.yaml
```

Training requires the dataset to follow the strict OpenAI tool-calling format (see [this link](https://huggingface.co/docs/trl/en/dataset_formats#tool-calling)). We provide scripts to prepare and validate the data accordingly:

- `linalg_zero/`
    - `sft/scripts/prepare_dataset.py`: prepares the SFT dataset.
    - `grpo/scripts/prepare_dataset.py`: prepares and validates the GRPO dataset.

## Results

We provide a recipe to encourage planning and tool-use capabilities in the [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) model, starting from a pre-trained (not instruction-tuned) base model.

This yields models like [Linalg-Zero-SFT](https://huggingface.co/atomwalk12/LinalgZero-SFT) and [Linalg-Zero-GRPO](https://huggingface.co/atomwalk12/LinalgZero-GRPO), with the following downstream performance on the test set:


| Metric             | LinAlgZero-SFT | LinAlgZero-GRPO |
|--------------------|----------------|-----------------|
| Optimal Trajectory | 89.87%         | 90.26%          |
| Correctness        | 91.86%         | 92.63%          |
| Format Validity    | 96.15%         | 96.66%          |
| Tool Success       | 100.00%        | 100.00%         |

### Artifacts

| Artifact | Link |
|---|---|
| SFT checkpoint | [atomwalk12/LinalgZero-SFT](https://huggingface.co/atomwalk12/LinalgZero-SFT) |
| GRPO checkpoint | [atomwalk12/LinAlgZero-GRPO](https://huggingface.co/atomwalk12/LinAlgZero-GRPO) |
| Base dataset | [atomwalk12/linalgzero](https://huggingface.co/datasets/atomwalk12/linalgzero) |
| Distilled dataset (clean) | [atomwalk12/linalgzero-distilled-clean](https://huggingface.co/datasets/atomwalk12/linalgzero-distilled-clean) |
| SFT dataset | [atomwalk12/linalgzero-sft](https://huggingface.co/datasets/atomwalk12/linalgzero-sft) |
| GRPO dataset | [atomwalk12/linalgzero-grpo](https://huggingface.co/datasets/atomwalk12/linalgzero-grpo) |

## Reproducibility
- **Distillation:** H100 80GB on [Runpod](https://www.runpod.io/) with [Qwen/Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8); 15 hours at $2.39/hr (~$25).
- **SFT:** Local 24GB RTX 4090 with [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).
- **GRPO:** RTX 6000 Ada on [Runpod](https://www.runpod.io/), improving on the SFT baseline; 57 hours at $0.77/hr (~$50).
- **Total:** ~$75 using a mix of cloud GPUs and local training.

## Acknowledgements
- We base our distillation pipeline on [distilabel](https://github.com/argilla-io/distilabel).
- We base the RL experiment on [ART](https://deepwiki.com/OpenPipe/ART).
- We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation

If you find this project is useful in your own work, please consider citing as follows:

```bibtex
@misc{linalg-zero,
    title = {Linalg-Zero: Distilling Neurosymbolic Reasoning for Linear Algebra in Small Language Models},
    url = {https://github.com/atomwalk12/linalg-zero},
    author = {{Razvan F. Vasile}},
    month = {March},
    year = {2026}
}
```
