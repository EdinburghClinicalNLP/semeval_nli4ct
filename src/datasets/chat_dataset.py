import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.configs import DataConfigs


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, data_configs: DataConfigs, split: str, **kwargs):
        data_filename = getattr(data_configs, f"{split}_data_filename")
        self.data = self.generate_data(
            data_configs.data_dir,
            data_filename,
            data_configs.claims_dir,
        )

    def generate_data(
        self, data_dir: str, data_filename: str, claims_dir: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generates data from clinical trials for Task 1: Textual entailment (NLI).

        Parameters:
            file_path (str): Path to the JSON of the dataset.

        Returns:
            joint_data (List[str]): List of training instances in form of:
                "
                    Evidence: primary trial: <primary_evidence_text> | secondary trial: <secondary_evidence_text>
                    Statement: <statement_text>
                    Answer:
                "
            labels (List[str]): List of labels, either "Entailment" or "Contradiction"
        """

        # Read the file.
        df = pd.read_json(os.path.join(data_dir, data_filename))
        df = df.transpose()
        statement_id = df.index.tolist()

        # Extract claims and labels.
        claims = df.Statement.tolist()

        labels = df.Label.tolist()

        # (Prepare to) Extract all evidence sentences from clinical trials
        evidence_texts = list()
        statement_texts = list()
        primary_cts, secondary_cts = df.Primary_id.tolist(), df.Secondary_id.tolist()
        sections, types = df.Section_id.tolist(), df.Type.tolist()

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

            prompted_statement = f"Statement: {statement}\nAnswer: "
            statement_texts.append(prompted_statement)

        return {
            "id": statement_id,
            "section": sections,
            "type": types,
            "evidence": evidence_texts,
            "statement": statement_texts,
            "labels": labels,
        }

    def __getitem__(self, idx):
        return {
            "id": self.data["id"][idx],
            "section": self.data["section"][idx],
            "type": self.data["type"][idx],
            "text": self.data["evidence"][idx] + "\n" + self.data["statement"][idx],
            "labels": self.data["labels"][idx],
        }

    def __len__(self):
        return len(self.data["labels"])
