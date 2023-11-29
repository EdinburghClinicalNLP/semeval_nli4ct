import math
import os

import hydra
import pandas as pd
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

import wandb
from src.configs import TrainingConfigs
from src.factories import get_dataset, get_lr_scheduler, get_optimizer
from src.models import ChatModelPipeline, LanguageModelPipeline


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = hydra_cfg["runtime"]["output_dir"]

        self.pipeline = ChatModelPipeline(self.configs.model)
        self.dataloaders = self._load_dataset()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.configs.trainer.gradient_accumulation_steps,
            log_with="wandb",
        )

        if not configs.debug:
            self._setup_run()
        # self._setup_training()

    def _load_dataset(self) -> dict:
        dataloaders = {}

        # Convert data into datasets
        for split in ["train", "valid", "test"]:
            print(f"Setup {split} data loader")
            dataset = get_dataset(self.configs.dataloader)(
                self.configs.data, tokenizer=self.pipeline.tokenizer, split=split
            )
            dataloaders[split] = DataLoader(
                dataset,
                shuffle=True,
                **self.configs.dataloader.configs,
            )

        return dataloaders

    def _setup_run(self):
        ## Set group name
        self.wandb_group_name = ""

        # TODO: This is just a shortcut for the moment. Naming should be more comprehensive
        self.wandb_run_name = self.configs.trainer.experiment_name

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

    # def _setup_training(self):
    #     # model
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.configs.model.configs.model_name_or_path,
    #     )

    #     # optimizer
    #     self.optimizer = get_optimizer(self.configs.optimizer, self.model)

    #     # lr scheduler
    #     num_training_steps = (
    #         math.ceil(
    #             len(self.train_dataloader)
    #             / self.configs.experiment.gradient_accumulation_steps
    #         )
    #         * self.configs.experiment.epochs
    #     )
    #     self.lr_scheduler = get_lr_scheduler(
    #         self.configs.lr_scheduler, self.optimizer, num_training_steps
    #     )

    #     (
    #         self.model,
    #         self.train_dataloader,
    #         self.valid_dataloader,
    #         self.test_dataloader,
    #         self.optimizer,
    #         self.lr_scheduler,
    #     ) = self.accelerator.prepare(
    #         self.model,
    #         self.train_dataloader,
    #         self.valid_dataloader,
    #         self.test_dataloader,
    #         self.optimizer,
    #         self.lr_scheduler,
    #     )

    @staticmethod
    def compute_metrics(labels, predictions):
        f1 = f1_score(labels, predictions, average="weighted")
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)

        return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}

    def train(self):
        for epoch in range(self.configs.trainer.epochs):
            self.pipeline.model.train()
            break

    def test(self, split: str):
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
            # try:
            prediction = self.pipeline.generate(batch)
            postprocessed_prediction = self.pipeline.postprocess_prediction(
                prediction["decoded_text"]
            )
            # except Exception as exc:
            #     print(f"Failed to predict: {batch}")
            #     print(f"Exception: {exc}")
            #     prediction = {
            #         "input_length": None,
            #         "max_new_tokens": None,
            #         "decoded_text": None,
            #     }
            #     postprocessed_prediction = None

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
        wandb.log(
            metrics | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
        )

        return metrics
