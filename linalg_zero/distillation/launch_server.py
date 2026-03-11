"""Model Server Launcher - Supports both LlamaCPP and vLLM backends"""

import os
import subprocess
import sys
from dataclasses import asdict

from trl import TrlParser

from linalg_zero.config.data import DistillationConfig, LlamaCppServerConfig, VllmServerConfig
from linalg_zero.shared.utils import get_logger, setup_logging


def launch_llamacpp(config: LlamaCppServerConfig) -> None:
    """Launch LlamaCPP server"""
    print("Starting llama.cpp server...")

    config_dict = asdict(config)
    args = ["uv", "run", "python3", "-m", "llama_cpp.server"]
    for k, v in config_dict.items():
        if v is True:
            args.extend([f"--{k}"])
        elif v is None:
            continue
        else:
            args.extend([f"--{k}", str(v)])

    print(f"Command: {' '.join(args)}")
    _ = subprocess.run(args, check=True)  # noqa: S603


def launch_vllm(config: VllmServerConfig) -> None:
    """Launch vLLM server"""
    print("Starting vLLM server...")

    config_dict = asdict(config)
    args = ["uv", "run", "vllm", "serve"]
    for k, v in config_dict.items():
        if k == "model":
            args.extend([str(v)])
        elif v is True:
            args.extend([f"--{k}"])
        elif v is None:
            continue
        else:
            args.extend([f"--{k}", str(v)])

    print(f"Command: {' '.join(args)}")
    _ = subprocess.run(args)  # noqa: S603


def main() -> None:
    """Main function"""
    setup_logging(file_suffix="distillation_server.log")
    logger = get_logger(__name__)
    logger.info(f"Using the configuration file stored at: {os.path.abspath(sys.argv[2])}")

    # First parse to get the backend type
    backend = os.environ["INFERENCE_BACKEND"]

    # Determine which server config to use based on backend
    if backend == "llamacpp":
        # Parse again with LlamaCppServerConfig
        parser = TrlParser(dataclass_types=[DistillationConfig, LlamaCppServerConfig])
        llama_cpp_config: LlamaCppServerConfig = parser.parse_args_and_config()[1]

        logger.info("LlamaCPP Server configuration:")
        for field_name in llama_cpp_config.__dataclass_fields__:
            value = getattr(llama_cpp_config, field_name)
            logger.info(f"  {field_name}: {value}")

        launch_llamacpp(llama_cpp_config)

    elif backend == "vllm":
        # Parse again with VllmServerConfig
        parser = TrlParser(dataclass_types=[DistillationConfig, VllmServerConfig])
        vllm_config: VllmServerConfig = parser.parse_args_and_config()[1]

        logger.info("vLLM Server configuration:")
        for field_name in vllm_config.__dataclass_fields__:
            value = getattr(vllm_config, field_name)
            logger.info(f"  {field_name}: {value}")

        launch_vllm(vllm_config)

    else:
        logger.error(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
