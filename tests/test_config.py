from trl import TrlParser

from linalg_zero.config.data import DatasetGenerationConfig
from linalg_zero.shared.utils import get_config_dir


def test_config() -> None:
    # Create argument parser and parse the config file
    parser = TrlParser(DatasetGenerationConfig)
    arguments = ["--config", get_config_dir() + "/dataset/default_debug.yml"]
    config: DatasetGenerationConfig = parser.parse_args_and_config(arguments)[0]

    assert config is not None
