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
        self.train_data_filename = os.path.join(
            self.data_dir, data_configs.train_data_filename
        )

        # Prepare ICL examples for one-shot and two-shot learning
        if self.trainer_configs.name.startswith("icl_"):
            # Query corpus is the training data
            self.query_corpus = pd.read_json(
                self.trainer_configs.configs.query_corpus_filename
            )
            self.query_corpus = self.query_corpus.transpose()
            self.icl_examples = pd.read_json(
                os.path.join(
                    kwargs["icl_examples_dir"],
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
                all_icl_examples["contradictions"][0]["score"]
                > all_icl_examples["entailments"][0]["score"]
            ):
                selected_icl_examples += [all_icl_examples["contradictions"][0]["id"]]
            else:
                selected_icl_examples += [all_icl_examples["entailments"][0]["id"]]
        elif self.num_in_context_examples >= 2:
            # If two-shot or higher even numbers,
            # choose the top-k/2 of both the contradiction and entailment examples
            top_k_contradictions = [
                (example["id"], example["score"])
                for example in all_icl_examples["contradictions"][
                    : int(self.num_in_context_examples / 2)
                ]
            ]
            top_k_entailments = [
                (example["id"], example["score"])
                for example in all_icl_examples["entailments"][
                    : int(self.num_in_context_examples / 2)
                ]
            ]

            selected_icl_examples += top_k_contradictions
            selected_icl_examples += top_k_entailments

            # sorted by their scores, highest to lowest. Take only the ids
            selected_icl_examples = [
                example[0]
                for example in sorted(
                    selected_icl_examples, key=lambda x: x[1], reverse=True
                )
            ]

        return selected_icl_examples

    def generate_evidence_text(
        self,
        claim_section,
        claim_type,
        primary_cts_filename,
        secondary_cts_filename=None,
        primary_evidence_ids: List[int] = None,
        secondary_evidence_ids: List[int] = None,
    ):
        # Generate evidence texts for each claim.
        file_name = os.path.join(self.claims_dir, primary_cts_filename + ".json")

        evidence = ""
        with open(file_name, "r") as f:
            data = json.load(f)
            evidence += "primary trial: "
            if primary_evidence_ids:
                primary_evidences = []
                for primary_evidence_id in primary_evidence_ids:
                    primary_evidences += [data[claim_section][primary_evidence_id]]
                evidence += " ".join(primary_evidences)
            else:
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
                if secondary_evidence_ids:
                    secondary_evidences = []
                    for secondary_evidence_id in secondary_evidence_ids:
                        secondary_evidences += [
                            data[claim_section][secondary_evidence_id]
                        ]
                    evidence += " ".join(secondary_evidences)
                else:
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
            if self.trainer_configs.name.startswith("icl_"):
                icl_example_ids = self.generate_icl_examples(statement_ids[claim_id])
                icl_evidences = []
                icl_statements = []
                icl_labels = []
                for icl_example_id in icl_example_ids:
                    icl_example = self.query_corpus.loc[icl_example_id]
                    icl_evidences += [
                        self.generate_evidence_text(
                            icl_example["Section_id"],
                            icl_example["Type"],
                            icl_example["Primary_id"],
                            icl_example["Secondary_id"],
                            icl_example["Primary_evidence_index"]
                            if "Primary_evidence_index" in icl_example
                            else None,
                            icl_example["Secondary_evidence_index"]
                            if "Secondary_evidence_index" in icl_example
                            else None,
                        )
                    ]
                    icl_statements += [icl_example["Statement"]]
                    if self.trainer_configs.name.startswith("icl_cot_"):
                        icl_labels += [icl_example["CoT_label"]]
                    else:
                        icl_labels += [icl_example["Label"]]
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
        icl_examples = []
        icl_labels = []
        if "cot_" in self.trainer_configs.name:
            cot_prompt = self.trainer_configs.configs.cot_prompt
        else:
            cot_prompt = ""

        if self.trainer_configs.name.startswith("icl_"):
            for icl_evidence, icl_statement, icl_label in zip(
                self.data["icl_evidence"][idx],
                self.data["icl_statement"][idx],
                self.data["icl_label"][idx],
            ):
                example = self.instruction_template.format(
                    icl_example="",
                    evidence=icl_evidence,
                    statement=icl_statement,
                    cot_prompt=cot_prompt,
                )
                icl_examples += [example]
                icl_labels += [icl_label]

        return {
            "id": self.data["id"][idx],
            "section": self.data["section"][idx],
            "type": self.data["type"][idx],
            "icl_inputs": icl_examples,
            "icl_labels": icl_labels,
            "text": self.instruction_template.format(
                icl_example="",
                evidence=self.data["evidence"][idx],
                statement=self.data["statement"][idx],
                cot_prompt=cot_prompt,
            ),
            "labels": self.data["labels"][idx],
        }

    def __len__(self):
        return len(self.data["labels"])
