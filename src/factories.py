from typing import Optional

import torch
from transformers import PreTrainedModel, get_scheduler

from src import datasets, models, retrievers
from src.configs import (
    DataLoaderConfigs,
    LRSchedulerConfigs,
    ModelConfigs,
    OptimizerConfigs,
    RetrieverConfigs,
)


def get_optimizer(
    optimizer_config: OptimizerConfigs, model: PreTrainedModel
) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, optimizer_config.name)
    return optimizer_class(model.parameters(), **optimizer_config.configs)


def get_lr_scheduler(
    lr_scheduler_config: LRSchedulerConfigs,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if lr_scheduler_config.name is None:
        return None
    if hasattr(torch.optim.lr_scheduler, lr_scheduler_config.name):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler_config.name)
        return lr_scheduler_class(optimizer, **lr_scheduler_config.configs)

    return get_scheduler(
        name=lr_scheduler_config.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **lr_scheduler_config.configs,
    )


def get_dataset(dataloader_config: DataLoaderConfigs):
    return getattr(datasets, dataloader_config.dataset_loader)


def get_pipeline(model_config: ModelConfigs):
    pipeline = getattr(models, model_config.pipeline)
    return pipeline


def get_retriever(retriever_config: RetrieverConfigs):
    pipeline = getattr(retrievers, retriever_config.name)
    return pipeline
