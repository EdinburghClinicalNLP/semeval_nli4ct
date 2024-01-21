import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")
from typing import List

import hydra
import pandas as pd
import spacy
from omegaconf import OmegaConf
from openai import AzureOpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.configs import (
    DataConfigs,
    InstructionConfigs,
    TrainerConfigs,
    TrainingConfigs,
    register_base_configs,
)
from src.datasets import ChatDataset


def generate_cot_reasons(openai_client, text, label):
    prompts = [
        {
            "role": "system",
            "content": "You are a helpful clinician's assistant designed to identify if a clinical statement is a contradiction or an entailment to the presented evidence.",
        },
        {
            "role": "user",
            "content": f"{text}{label}\nReason the answer step by step",
        },
    ]

    response = openai_client.chat.completions.create(
        model="chatgpt_icd_coding",
        messages=prompts,
        temperature=0,
        top_p=0,
        max_tokens=256,
        frequency_penalty=0,
        presence_penalty=0,
    )

    cot_reason = response.choices[0].message.content

    return cot_reason


def main():
    # OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    dataset = ChatDataset(
        DataConfigs(
            data_dir="data",
            train_data_filename="train.json",
            valid_data_filename="dev.json",
            test_data_filename="practice_test.json",
            claims_dir="CT json",
            max_length=4096,
        ),
        InstructionConfigs(
            instruction_template="{icl_example}Evidence: {evidence}\nStatement: {statement}\nQuestion: Answer in 1 word. Is the statement a contradiction or an entailment?\nAnswer: "
        ),
        TrainerConfigs(name="0_shot", configs={}),
        icl_examples_dir="",
        tokenizer=None,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
    )

    cot_reasons = {}
    for batch in tqdm(dataloader):
        try:
            cot_reason = generate_cot_reasons(
                openai_client, batch["text"][0], batch["labels"][0]
            )
        except Exception as e:
            print(f"Failed to generate reason: {e}")
            cot_reason = ""
        cot_reasons[batch["id"][0]] = cot_reason

    # Save to json file
    with open("cot_reasons.json", "w") as f:
        json.dump(cot_reasons, f)


if __name__ == "__main__":
    main()
