import math
import os

import pandas as pd
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM

from .configs import TrainingConfigs
from .dataset import ClinicalTrialDataset, generate_nli_data
from .factories import get_lr_scheduler, get_optimizer


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        (
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self._load_dataset()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.configs.experiment.gradient_accumulation_steps,
            log_with="wandb",
        )

        if not configs.debug:
            self._setup_run()
        self._setup_training()

    def _load_dataset(self) -> dict:
        train_dataloader = None
        valid_dataloader = None

        train_inputs, train_labels = generate_nli_data(
            self.configs.data.data_dir,
            self.configs.data.train_data_filename,
            self.configs.data.claims_dir,
        )

        valid_inputs, valid_labels = generate_nli_data(
            self.configs.data.data_dir,
            self.configs.data.valid_data_filename,
            self.configs.data.claims_dir,
        )

        print("train_inputs[:10]: ", train_inputs[:10])
        print("train_labels[:10]: ", train_labels[:10])
        exit()

        # Tokenize the data.
        tokenized_train_inputs = self.tokenizer(
            train_inputs,
            return_tensors="pt",
            truncation_strategy="only_first",
            add_special_tokens=True,
            padding=True,
        )
        tokenized_valid_inputs = self.tokenizer(
            valid_inputs,
            return_tensors="pt",
            truncation_strategy="only_first",
            add_special_tokens=True,
            padding=True,
        )

        # Convert data into datasets
        train_dataset = ClinicalTrialDataset(tokenized_train_inputs, train_labels)
        dev_dataset = ClinicalTrialDataset(tokenized_valid_inputs, valid_labels)

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
