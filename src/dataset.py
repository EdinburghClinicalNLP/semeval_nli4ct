import json
import os
from typing import Tuple

import pandas as pd
import torch


def generate_nli_data(
    data_dir: str, data_filename: str, claims_dir: str
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

    # Extract claims and labels. Map labels to binary values (0, 1).
    claims = df.Statement.tolist()
    labels = df.Label.tolist()
    labels = list(map(lambda x: 1 if x == "Entailment" else 0, labels))

    print(df.head())

    # (Prepare to) Extract all evidence sentences from clinical trials
    evidence_texts = list()
    primary_cts, secondary_cts = df.Primary_id, df.Secondary_id
    sections, types = df.Section_id, df.Type

    # Generate evidence texts for each claim.
    claims_dir = os.path.join(data_dir, claims_dir)
    for claim_id in range(len(claims)):
        file_name = os.path.join(claims_dir, primary_cts[claim_id] + ".json")

        with open(file_name, "r") as f:
            data = json.load(f)
            evidence = "primary trial: "
            evidence += "\n".join(data[sections[claim_id]])

        # If it is a comparative claim, also add evidence sentences from the 2nd trial.
        if types[claim_id] == "Comparison":
            file_name = os.path.join(claims_dir, secondary_cts[claim_id] + ".json")

            # Evidence for the secondary trial is in form:
            # "| secondary trial: sent_1. sent_2. (...) sent_n."
            with open(file_name, "r") as f:
                data = json.load(f)
                evidence += " | secondary trial: "
                evidence += "\n".join(data[sections[claim_id]])

        evidence_texts.append(evidence)

    # One training instance is: "claim [SEP] full_evidence_text"
    joint_data = list()
    for i in range(len(claims)):
        premise = claims[i]
        hypothesis = evidence_texts[i]
        joint = premise + " [SEP] " + hypothesis
        joint_data.append(joint)

    return joint_data, labels


class ClinicalTrialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
