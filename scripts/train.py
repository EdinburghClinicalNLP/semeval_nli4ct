import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import huggingface_hub
import hydra
from omegaconf import OmegaConf

from src.configs import TrainingConfigs, register_base_configs
from src.trainer import Trainer
from src.utils import common_utils


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    print(configs)
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    common_utils.setup_random_seed(configs.random_seed)

    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    trainer = Trainer(configs)
    trainer.train()
    trainer.test(split="valid")
    trainer.test(split="test")


if __name__ == "__main__":
    register_base_configs()
    main()
