import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pickle
from itertools import combinations, product

from tqdm import tqdm
import hydra
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.configs import TrainingConfigs, register_base_configs
from src.datasets import ChatDataset


def generate_dataset_by_id(configs: TrainingConfigs, split: str):
    dataset = ChatDataset(
        configs.data,
        configs.instruction,
        configs.trainer,
        icl_examples_dir=configs.retriever.icl_examples_dir,
        tokenizer=None,
        split=split,
    )

    dataset_dict = {}
    for sample_idx in range(len(dataset.data["id"])):
        dataset_dict[dataset.data["id"][sample_idx]] = {
            "section": dataset.data["section"][sample_idx],
            "type": dataset.data["type"][sample_idx],
            "evidence": dataset.data["evidence"][sample_idx],
            "statement": dataset.data["statement"][sample_idx],
            "label": dataset.data["labels"][sample_idx],
        }

    return dataset_dict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    with open("data/semeval_train_faithfulness_pair.pkl", "rb") as f:
        train_pairs = pickle.load(f)

    # with open("data/semeval_dev_faithfulness_pair.pkl", "rb") as f:
    #     valid_pairs = pickle.load(f)

    # with open("data/semeval_test_faithfulness_pair.pkl", "rb") as f:
    #     test_pairs = pickle.load(f)

    train_dataset_dict = generate_dataset_by_id(configs, "train")
    # valid_dataset_dict = generate_dataset_by_id(configs, "valid")
    # test_dataset_dict = generate_dataset_by_id(configs, "test")

    model = AutoModel.from_pretrained(
        configs.model.configs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(configs.model.configs.model_name_or_path)
    common_lora_config = OmegaConf.to_container(
        configs.trainer.configs.common_lora_config
    )
    lora_config = LoraConfig(
        **common_lora_config,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config, adapter_name="contrastive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    for epoch in range(configs.trainer.configs.epochs):
        total_loss = 0
        for step, train_pair in enumerate(tqdm(train_pairs)):
            try:
                anchor = ""
                statement_pos = ""
                statement_neg = ""
                for train_id in train_pair:
                    train_data = train_dataset_dict[train_id]
                    if train_data["label"] == "Entailment":
                        statement_pos = train_data["statement"]
                    elif train_data["label"] == "Contradiction":
                        statement_neg = train_data["statement"]
                    if not anchor:
                        anchor = train_data["evidence"]
            except KeyError:
                continue

            model.train()

            anchor = tokenizer(anchor, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
            statement_pos = tokenizer(statement_pos, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]
            statement_neg = tokenizer(statement_neg, return_tensors="pt", max_length=1024, truncation=True)["input_ids"]

            anchor_logits = model(anchor)
            anchor_logits = anchor_logits.last_hidden_state.mean(dim=1)
            statement_pos_logits = model(statement_pos).last_hidden_state.mean(dim=1)
            statement_neg_logits = model(statement_neg).last_hidden_state.mean(dim=1)

            loss = triplet_loss(
                anchor_logits, statement_pos_logits, statement_neg_logits
            )
            total_loss += loss.detach().float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epoch_loss = total_loss / len(train_pairs)

        print(f"{epoch=}: {train_epoch_loss=}")

    print("Upload pretrained weights to HF")
    hf_username = os.getenv("HF_USERNAME")
    hf_upload_token = os.getenv("HF_UPLOAD_TOKEN")

    hf_repo_name = f"{hf_username}/{configs.model.name}_triplet_loss"
    model.push_to_hub(hf_repo_name, private=True, token=hf_upload_token)


if __name__ == "__main__":
    register_base_configs()
    main()
