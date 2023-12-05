import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

from typing import List

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from src.configs import RetrieverConfigs, TrainingConfigs, register_base_configs
from src.datasets import RetrieverDataset
from src.factories import get_dataset, get_retriever


def load_data(configs: TrainingConfigs):
    datasets = {}
    # Convert data into datasets
    for split in ["train", "valid", "test"]:
        print(f"Setup {split} data loader")
        datasets[split] = get_dataset(configs.dataloader)(
            configs.data, tokenizer=None, split=split
        )

    return datasets


def bm25_retriever(
    retriever_configs: RetrieverConfigs,
    query_corpus: RetrieverDataset,
    knowledge_corpus: RetrieverDataset,
):
    # Create BM25 object
    bm25 = get_retriever(retriever_configs)(knowledge_corpus)

    query_corpus = pd.DataFrame(query_corpus.data)

    relevant_documents = {}
    for (
        statement_id,
        evidence_section,
        evidence_type,
        evidence,
        statement,
    ) in tqdm(query_corpus[["id", "section", "type", "evidence", "statement"]].values):
        query = "\n".join([evidence, statement])
        documents = bm25.get_document_scores(
            query, evidence_section, evidence_type, statement_id
        )
        relevant_documents[statement_id] = documents

    return relevant_documents


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs):
    print(configs)
    # Load data
    datasets: dict = load_data(configs)

    hydra_cfg = HydraConfig.get()
    outputs_dir = f"in_context_examples/{hydra_cfg.runtime.choices.retriever}"
    os.makedirs(outputs_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        # Run retrieval
        relevant_documents = bm25_retriever(
            configs.retriever,
            query_corpus=datasets[split],
            knowledge_corpus=datasets["train"],
        )
        # Save the in context examples as a json file
        with open(
            os.path.join(
                outputs_dir,
                f"{split}_in_context_examples.json",
            ),
            "w",
        ) as json_file:
            json.dump(relevant_documents, json_file)


if __name__ == "__main__":
    register_base_configs()
    main()
