import math
import os

import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.configs import TrainingConfigs
from src.factories import get_dataset, get_lr_scheduler, get_optimizer, get_pipeline


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self.pipeline = get_pipeline(self.configs.model)(self.configs.model)
        self.dataloaders = self._load_dataset()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.configs.trainer.configs.gradient_accumulation_steps,
            log_with="wandb",
        )

        if not configs.debug:
            self._setup_run()

    def _load_dataset(self) -> dict:
        dataloaders = {}

        # Convert data into datasets
        for split in ["train", "valid", "test"]:
            print(f"Setup {split} data loader")
            dataset = get_dataset(self.configs.dataloader)(
                self.configs.data,
                self.configs.instruction,
                self.configs.trainer,
                icl_examples_dir=self.configs.retriever.icl_examples_dir,
                tokenizer=self.pipeline.tokenizer,
                split=split,
            )
            dataloaders[split] = DataLoader(
                dataset,
                shuffle=True,
                **self.configs.dataloader.configs,
            )

        return dataloaders

    def _setup_run(self):
        ## Set group name by trainer name (i.e. zero_shot, fine_tune)
        self.wandb_group_name = self.configs.trainer.name

        # Naming by model name, instruction name, and in context examples name
        wandb_run_name = [
            self.configs.model.name,
            self.hydra_cfg.runtime.choices.instruction,
        ]
        if self.configs.trainer.name in ["one_shot", "two_shot"]:
            wandb_run_name += [self.configs.retriever.icl_examples_dir.split("/")[-1]]
        self.wandb_run_name = "__".join(wandb_run_name)
        print(self.wandb_run_name)

        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.configs.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.configs.wandb_entity,
                        "name": self.wandb_run_name,
                        "group": self.wandb_group_name,
                    }
                },
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

    @staticmethod
    def compute_metrics(labels, predictions):
        f1 = f1_score(labels, predictions, average="weighted")
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)

        return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}

    def _setup_training(self):
        # Setup PEFT
        self.pipeline.setup_finetuning(self.configs.trainer.configs.lora_configs)

        # optimizer
        self.optimizer = get_optimizer(self.configs.optimizer, self.pipeline.model)

        # lr scheduler
        num_training_steps = (
            math.ceil(
                len(self.dataloaders["train"])
                / self.configs.trainer.configs.gradient_accumulation_steps
            )
            * self.configs.trainer.configs.epochs
        )
        self.lr_scheduler = get_lr_scheduler(
            self.configs.lr_scheduler, self.optimizer, num_training_steps
        )

        (
            self.pipeline.model,
            self.dataloaders["train"],
            self.dataloaders["valid"],
            self.dataloaders["test"],
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.pipeline.model,
            self.dataloaders["train"],
            self.dataloaders["valid"],
            self.dataloaders["test"],
            self.optimizer,
            self.lr_scheduler,
        )

    def train(self):
        if self.configs.trainer.name == "fine_tune":
            self._setup_training()

            prev_best_valid_metric = 0
            for epoch in range(self.configs.trainer.configs.epochs):
                total_loss = 0
                for step, batch in enumerate(tqdm(self.dataloaders["train"])):
                    with self.accelerator.accumulate(self.pipeline.model):
                        self.pipeline.model.train()
                        try:
                            outputs = self.pipeline.train(
                                batch,
                                batch["labels"],
                                max_train_seq_len=self.configs.trainer.configs.max_train_seq_len,
                            )
                            loss = outputs.loss

                            total_loss += loss.detach().float()

                            self.accelerator.backward(loss)
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        except torch.cuda.OutOfMemoryError as e:
                            print(f"batch: {batch}")
                            raise ValueError(e)

                train_epoch_loss = total_loss / len(self.dataloaders["train"])
                train_ppl = torch.exp(train_epoch_loss)
                train_metrics = self.test("train", log_metrics=False)
                valid_metrics = self.test("valid", log_metrics=False)

                self.accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
                self.accelerator.log(
                    {
                        "train/loss": train_epoch_loss,
                        "train/ppl": train_ppl,
                    }
                    | train_metrics
                    | valid_metrics,
                    step=epoch,
                )

                # Save a checkpoint
                if valid_metrics["valid/f1"] >= prev_best_valid_metric:
                    prev_best_valid_metric = valid_metrics["valid/f1"]
                    self.accelerator.save_state(
                        os.path.join(self.output_dir, "best_checkpoint")
                    )

            # Load best checkpoint for test evaluation
            self.accelerator.load_state(
                os.path.join(self.output_dir, "best_checkpoint")
            )
            self.accelerator.wait_for_everyone()

            _ = self.test("valid", log_metrics=True)
            _ = self.test("test", log_metrics=True)

            print("Create a WandB artifact from embedding")
            artifact = wandb.Artifact(self.wandb_run_name, type="model")
            artifact.add_dir(os.path.join(self.output_dir, "best_checkpoint"))
            wandb.log_artifact(artifact)

    def test(self, split: str, log_metrics: bool = True):
        print(f"Test on {split}")

        predictions_df = pd.DataFrame(
            columns=[
                "id",
                "section",
                "type",
                "text",
                "input_length",
                "max_new_tokens",
                "labels",
                "predictions",
                "original_predictions",
            ]
        )
        self.pipeline.model.eval()
        for step, batch in enumerate(tqdm(self.dataloaders[split])):
            # Test split doesn't have labels
            if split == "test":
                postprocessed_label = [None] * len(batch["labels"])
            else:
                postprocessed_label = [
                    label.lower() if label is not None else None
                    for label in batch["labels"]
                ]
            try:
                prediction = self.pipeline.generate(batch)
                postprocessed_prediction = self.pipeline.postprocess_prediction(
                    prediction["decoded_text"]
                )
            except Exception as exc:
                print(f"Failed to predict: {batch}")
                print(f"Exception: {exc}")
                prediction = {
                    "input_length": None,
                    "max_new_tokens": None,
                    "decoded_text": None,
                }
                postprocessed_prediction = None

            batch_df = pd.DataFrame(
                {
                    "id": batch["id"],
                    "section": batch["section"],
                    "type": batch["type"],
                    "text": batch["text"],
                    "input_length": [prediction["input_length"]],
                    "max_new_tokens": [prediction["max_new_tokens"]],
                    "labels": postprocessed_label,
                    "predictions": [postprocessed_prediction],
                    "original_predictions": [prediction["decoded_text"]],
                }
            )

            # Append the batch DataFrame to the overall predictions DataFrame
            predictions_df = pd.concat([predictions_df, batch_df], ignore_index=True)

            # Save the updated DataFrame to a CSV file after each batch
            predictions_df.to_csv(
                os.path.join(self.output_dir, f"predictions_{split}.csv"), index=False
            )

        if split == "test":
            metrics = {}
        else:
            # Map labels and predictions to int labels for evaluation
            # 1 = entailment, 0 = contradiction
            mapped_labels = []
            mapped_predictions = []
            for label, prediction in predictions_df[["labels", "predictions"]].values:
                if label.lower() == "entailment":
                    mapped_label = 1
                elif label.lower() == "contradiction":
                    mapped_label = 0
                mapped_labels += [mapped_label]

                if prediction is not None:
                    if prediction.lower() == "entailment":
                        mapped_predictions += [1]
                    elif prediction.lower() == "contradiction":
                        mapped_predictions += [0]
                    else:
                        # Intentionally assign incorrect prediction just to bypass evaluation
                        mapped_prediction = 0 if mapped_label == 1 else 1
                        mapped_predictions += [mapped_prediction]
                else:
                    # Intentionally assign incorrect prediction just to bypass evaluation
                    mapped_prediction = 0 if mapped_label == 1 else 1
                    mapped_predictions += [mapped_prediction]

            metrics = self.compute_metrics(mapped_labels, mapped_predictions)
            metrics = {
                f"{split}/{metric_name}": metric_value
                for metric_name, metric_value in metrics.items()
            }
            print(metrics)

        # Save DataFrame
        if log_metrics:
            self.accelerator.log(
                metrics
                | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
            )

        return metrics
