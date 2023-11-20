import logging
import os
import sys

sys.path.append(os.getcwd())

import hydra
from omegaconf import OmegaConf

from src import common_utils
from src.configs import TrainingConfigs, register_base_configs
from src.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    print(configs)
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    common_utils.setup_random_seed(configs.random_seed)

    trainer = Trainer(configs)
    trainer.train()


if __name__ == "__main__":
    register_base_configs()
    main()
