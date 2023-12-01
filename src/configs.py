from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfigs:
    data_dir: str = MISSING
    train_data_filename: str = MISSING
    valid_data_filename: str = MISSING
    test_data_filename: str = MISSING
    claims_dir: str = MISSING
    max_length: int = MISSING


@dataclass
class DataLoaderConfigs:
    dataset_loader: str = MISSING
    configs: dict = MISSING


@dataclass
class TrainerConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class LRSchedulerConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class ModelConfigs:
    name: str = MISSING
    pipeline: str = MISSING
    configs: dict = MISSING


@dataclass
class OptimizerConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class WandBConfigs:
    entity: str = MISSING
    project: str = MISSING


@dataclass
class TrainingConfigs:
    data: DataConfigs = MISSING
    dataloader: DataLoaderConfigs = MISSING
    trainer: TrainerConfigs = MISSING
    lr_scheduler: LRSchedulerConfigs = MISSING
    model: ModelConfigs = MISSING
    optimizer: OptimizerConfigs = MISSING
    wandb_project: str = MISSING
    wandb_entity: str = MISSING
    debug: bool = False
    random_seed: int = 1234


def register_base_configs() -> None:
    configs_store = ConfigStore.instance()
    configs_store.store(name="base_config", node=TrainingConfigs)
    configs_store.store(
        group="optimizer", name="base_optimizer_config", node=OptimizerConfigs
    )
    configs_store.store(group="data", name="base_data_config", node=DataConfigs)
    configs_store.store(
        group="dataloader", name="base_dataloader_config", node=DataLoaderConfigs
    )
    configs_store.store(
        group="trainer", name="base_trainer_config", node=TrainerConfigs
    )
    configs_store.store(
        group="lr_scheduler", name="base_lr_scheduler_config", node=LRSchedulerConfigs
    )
    configs_store.store(group="model", name="base_model_config", node=ModelConfigs)
