import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.configs import DataConfigs, InstructionConfigs, TrainerConfigs


class ChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        instruction_configs: InstructionConfigs,
        trainer_configs: TrainerConfigs,
        split: str,
        **kwargs,
    ):
        self.data_configs = data_configs
        self.instruction_configs = instruction_configs
        self.trainer_configs = trainer_configs

        self.split = split

        data_filename = getattr(data_configs, f"{split}_data_filename")
        self.data_dir = data_configs.data_dir
        self.claims_dir = os.path.join(self.data_dir, data_configs.claims_dir)
        self.train_data_filename = (
            os.path.join(self.data_dir, data_configs.train_data_filename),
        )

        # Prepare ICL examples for one-shot and two-shot learning
        if self.trainer_configs.name in ["one_shot", "two_shot"]:
            # Query corpus is the training data
            self.query_corpus = pd.read_json(self.train_data_filename)
            self.icl_examples = pd.read_json(
                os.path.join(
                    self.trainer_configs.configs.in_context_examples_dir,
                    self.split + ".json",
                )
            )
            self.num_in_context_examples = (
                self.trainer_configs.configs.num_in_context_examples
            )

        # Prepare data
        self.data = self.generate_data(
            data_filename,
        )

        self.instruction_template = self.instruction_configs.instruction_template

    def generate_icl_examples(self, statement_id: str):
        # Get the ICL examples depending on the number of ICL examples allowed
        all_icl_examples = self.icl_examples[statement_id]

        selected_icl_examples = []
        if self.num_in_context_examples == 1:
            # If one-shot, choose the highest between the contradiction and entailment examples
            if (
                all_icl_examples["contradiction"][0]["score"]
                > all_icl_examples["entailment"][0]["score"]
            ):
                selected_icl_examples += [all_icl_examples["contradiction"][0]["id"]]
            else:
                selected_icl_examples += [all_icl_examples["entailment"][0]["id"]]
        elif self.num_in_context_examples == 2:
            # If two-shot, choose the highest of both the contradiction and entailment examples
            selected_icl_examples += [
                all_icl_examples["contradiction"][0]["id"],
                all_icl_examples["entailment"][0]["id"],
            ]

        return selected_icl_examples

    def generate_evidence_text(
        self,
        claim_section,
        claim_type,
        primary_cts_filename,
        secondary_cts_filename=None,
    ):
        # Generate evidence texts for each claim.
        file_name = os.path.join(self.claims_dir, primary_cts_filename + ".json")

        evidence = ""
        with open(file_name, "r") as f:
            data = json.load(f)
            evidence += "primary trial: "
            evidence += " ".join(data[claim_section])

        # If it is a comparative claim, also add evidence sentences from the 2nd trial.
        if claim_type == "Comparison":
            file_name = os.path.join(self.claims_dir, secondary_cts_filename + ".json")

            # Evidence for the secondary trial is in form:
            # "| secondary trial: sent_1. sent_2. (...) sent_n."
            with open(file_name, "r") as f:
                data = json.load(f)
                evidence += " | secondary trial: "
                evidence += " ".join(data[claim_section])

        return evidence

    def generate_data(
        self, data_filename: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generates data from clinical trials for Task 1: Textual entailment (NLI).

        Parameters:
            file_path (str): Path to the JSON of the dataset.

        Returns:
            joint_data (List[str]): List of training instances in form of the provided instruction template
            labels (List[str]): List of labels, either "Entailment" or "Contradiction"
        """

        # Read the file.
        df = pd.read_json(os.path.join(self.data_dir, data_filename))
        df = df.transpose()
        statement_ids = df.index.tolist()

        # Extract claims and labels.
        claims = df.Statement.tolist()

        labels = df.Label.tolist()

        # (Prepare to) Extract all evidence sentences from clinical trials
        icl_evidence_texts = list()
        icl_statement_texts = list()
        icl_label_texts = list()
        evidence_texts = list()
        statement_texts = list()
        primary_cts, secondary_cts = df.Primary_id.tolist(), df.Secondary_id.tolist()
        sections, types = df.Section_id.tolist(), df.Type.tolist()

        for claim_id, statement in enumerate(claims):
            # Generate ICL examples
            if self.trainer_configs.name in ["one_shot", "two_shot"]:
                icl_example_ids = self.generate_icl_examples(statement_ids[claim_id])
                icl_evidences = []
                icl_statements = []
                icl_labels = []
                for icl_example_id in icl_example_ids:
                    icl_evidences += [
                        self.generate_evidence_text(
                            sections[icl_example_id],
                            types[icl_example_id],
                            primary_cts[icl_example_id],
                            secondary_cts[icl_example_id],
                        )
                    ]
                    icl_statements += [df.loc[icl_example_id]["Statement"]]
                    icl_labels += [df.loc[icl_example_id]["Label"]]
                icl_evidence_texts += [icl_evidences]
                icl_statement_texts += [icl_statements]
                icl_label_texts += [icl_labels]

            # Generate evidence and statement texts
            evidence = self.generate_evidence_text(
                sections[claim_id],
                types[claim_id],
                primary_cts[claim_id],
                secondary_cts[claim_id],
            )

            evidence_texts.append(evidence)
            statement_texts.append(statement)

        return {
            "id": statement_ids,
            "section": sections,
            "type": types,
            "icl_evidence": icl_evidence_texts,
            "icl_statement": icl_statement_texts,
            "icl_label": icl_label_texts,
            "evidence": evidence_texts,
            "statement": statement_texts,
            "labels": labels,
        }

    def __getitem__(self, idx):
        examples = ""
        if self.trainer_configs.name in ["one_shot", "two_shot"]:
            for icl_evidence, icl_statement, icl_label in zip(
                self.data["icl_evidence"][idx],
                self.data["icl_statement"][idx],
                self.data["icl_label"][idx],
            ):
                example = (
                    self.instruction_template.format(
                        icl_example="",
                        evidence=icl_evidence,
                        statement=icl_statement,
                    )
                    + icl_label
                )
                examples += example + "\n\n"

        return {
            "id": self.data["id"][idx],
            "section": self.data["section"][idx],
            "type": self.data["type"][idx],
            "text": self.instruction_template.format(
                icl_example=examples,
                evidence=self.data["evidence"][idx],
                statement=self.data["statement"][idx],
            ),
            "labels": self.data["labels"][idx],
        }

    def __len__(self):
        return len(self.data["labels"])
