import json
import os
from typing import Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.configs import DataConfigs


class ChatDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_configs: DataConfigs, tokenizer: PreTrainedTokenizer, split: str
    ):
        data_filename = getattr(data_configs, f"{split}_data_filename")
        self.evidence_texts, self.statement_texts, self.labels = self.generate_data(
            data_configs.data_dir,
            data_filename,
            data_configs.claims_dir,
        )

        self.inputs = [
            "\n".join(evidence_statement_pair)
            for evidence_statement_pair in zip(
                self.evidence_texts, self.statement_texts
            )
        ]

    def generate_data(
        self, data_dir: str, data_filename: str, claims_dir: str
    ) -> Tuple[list, list]:
        """
        Generates data from clinical trials for Task 1: Textual entailment (NLI).

        Parameters:
            file_path (str): Path to the JSON of the dataset.

        Returns:
            joint_data: List of training instances in form of "claim [SEP] evidence_text" (str)
            labels: List of labels, either 1 for "Entailment" or 0 for "Contradiction" (int)
        """

        # Read the file.
        df = pd.read_json(os.path.join(data_dir, data_filename))
        df = df.transpose()

        # Extract claims and labels.
        claims = df.Statement.tolist()
        labels = df.Label.tolist()

        # (Prepare to) Extract all evidence sentences from clinical trials
        evidence_texts = list()
        statement_texts = list()
        primary_cts, secondary_cts = df.Primary_id, df.Secondary_id
        sections, types = df.Section_id, df.Type

        # Generate evidence texts for each claim.
        claims_dir = os.path.join(data_dir, claims_dir)
        for claim_id, statement in enumerate(claims):
            file_name = os.path.join(claims_dir, primary_cts[claim_id] + ".json")

            section = sections[claim_id]
            evidence_prompt = "Evidence: "
            evidence = evidence_prompt

            with open(file_name, "r") as f:
                data = json.load(f)
                evidence += "primary trial: "
                evidence += " ".join(data[section])

            # If it is a comparative claim, also add evidence sentences from the 2nd trial.
            if types[claim_id] == "Comparison":
                file_name = os.path.join(claims_dir, secondary_cts[claim_id] + ".json")

                # Evidence for the secondary trial is in form:
                # "| secondary trial: sent_1. sent_2. (...) sent_n."
                with open(file_name, "r") as f:
                    data = json.load(f)
                    evidence += " | secondary trial: "
                    evidence += " ".join(data[section])

            evidence_texts.append(evidence)

            prompted_statement = f"Statement: {statement}\nAnswer:"
            statement_texts.append(prompted_statement)

        return evidence_texts, statement_texts, labels

    def __getitem__(self, idx):
        return {
            "text": self.inputs[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)
