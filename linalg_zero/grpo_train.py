from dotenv import load_dotenv

# For debugging purposes apply dotenv file
load_dotenv()

import asyncio

import art
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from linalg_zero.grpo.run_rl import train
from linalg_zero.grpo.types import LinAlgPolicyConfig


@hydra.main(
    version_base=None,
    config_path="config/grpo/Qwen/Qwen2.5-3B",
    config_name="local.yaml",
)
def main(cfg: DictConfig) -> None:
    # Convert all configs to plain dicts
    init_config = OmegaConf.to_container(cfg.init, resolve=True)
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    run_config = OmegaConf.to_container(cfg.run, resolve=True)
    peft_config = OmegaConf.to_container(cfg.peft, resolve=True)
    trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
    engine_args = OmegaConf.to_container(cfg.engine, resolve=True)
    # Use peft config for LoRA training instead of:
    # torchtune_args = OmegaConf.to_container(cfg.torchtune, resolve=True)
    print(f"Training model {cfg.run.base_model}")

    assert isinstance(init_config, dict), "Init config must be provided"
    assert isinstance(training_config, dict), "Training config must be provided"
    assert isinstance(run_config, dict), "Run config must be provided"
    assert isinstance(peft_config, dict), "Peft args must be provided"
    assert isinstance(trainer_args, dict), "Trainer args must be provided"
    assert isinstance(engine_args, dict), "Engine args must be provided"

    # Set dynamic values
    if "tensor_parallel_size" not in engine_args:
        engine_args["tensor_parallel_size"] = torch.cuda.device_count()

    # Build model and run training
    model = art.TrainableModel(
        name=run_config["project_id"],
        project=run_config["project"],
        base_model=run_config["base_model"],
        config=LinAlgPolicyConfig(
            training_config=training_config,
            run_config=run_config,
        ),
        _internal_config=art.dev.InternalModelConfig(
            init_args=init_config,
            engine_args=engine_args,
            trainer_args=trainer_args,
            peft_args=peft_config,
            # Use peft_config for LoRA training instead of:
            # torchtune_args=torchtune_args,
        ),
    )

    asyncio.run(train(model))


if __name__ == "__main__":
    main()
