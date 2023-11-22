import math
import os

import pandas as pd
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

from src.configs import TrainingConfigs
from src.datasets.baseline_dataset import BaselineDataset
from src.factories import get_dataset, get_lr_scheduler, get_optimizer


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        self.tokenizer = self._load_tokenizers()
        (
            self.train_dataloader,
            self.valid_dataloader,
        ) = self._load_dataset()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.configs.experiment.gradient_accumulation_steps,
            log_with="wandb",
        )

        if not configs.debug:
            self._setup_run()
        self._setup_training()

    def _load_tokenizers(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model.configs["model_name_or_path"]
        )

        if (
            getattr(tokenizer, "pad_token_id") is None
            or getattr(tokenizer, "pad_token") is None
        ):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _load_dataset(self) -> dict:
        train_dataloader = None
        valid_dataloader = None

        # Convert data into datasets
        train_dataset = get_dataset(self.configs.dataloader)(
            self.configs.data, self.tokenizer, "train"
        )
        valid_dataset = get_dataset(self.configs.dataloader)(
            self.configs.data, self.tokenizer, "valid"
        )

        data_collator = DefaultDataCollator(return_tensors="pt")

        print(f"Setup Train DataLoader")
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            **self.configs.dataloader.configs,
        )
        print(f"Setup Valid DataLoader")
        valid_dataloader = DataLoader(
            valid_dataset,
            shuffle=True,
            collate_fn=data_collator,
            **self.configs.dataloader.configs,
        )

        return train_dataloader, valid_dataloader

    def _setup_run(self):
        ## Set group name
        self.group_name = ""

        # Run name depends on the hyperparameters
        hyperparams = []
        self.run_name = "_".join(hyperparams)

        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.configs.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.configs.wandb_entity,
                        "name": self.run_name,
                        "group": self.group_name,
                    }
                },
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

    def _setup_training(self):
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs.model.configs.model_name_or_path,
        )

        # optimizer
        self.optimizer = get_optimizer(self.configs.optimizer, self.model)

        # lr scheduler
        num_training_steps = (
            math.ceil(
                len(self.train_dataloader)
                / self.configs.experiment.gradient_accumulation_steps
            )
            * self.configs.experiment.epochs
        )
        self.lr_scheduler = get_lr_scheduler(
            self.configs.lr_scheduler, self.optimizer, num_training_steps
        )

        (
            self.model,
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
            self.optimizer,
            self.lr_scheduler,
        )

    def compute_metrics(self, labels, predictions):
        f1 = f1_score(labels, predictions, average="weighted")
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)

        return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}

    def train(self):
        for epoch in range(self.configs.training_configs.epochs):
            self.model.train()
            break

    def test(self, split: str):
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        total_loss = 0
        all_labels = []
        all_predictions = []

        self.model.eval()

        metrics = None

        return metrics
