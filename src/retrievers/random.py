import random
from typing import Dict

import numpy as np
import pandas as pd

from src.datasets import RetrieverDataset

from .utils import tokenize_and_stem


class Random:
    """
    Randomly select k examples within the same section and type as the query

    """

    def __init__(self, corpus: RetrieverDataset, top_k=5, **kwargs) -> None:
        print("Initialising Random Retriever")
        self.corpus_df = pd.DataFrame(corpus.data)
        self.top_k = top_k

    def get_document_scores(
        self, query, section, type, statement_id
    ) -> Dict[str, list]:
        # Given the section and type, narrow down the search space
        relevant_docs = self.corpus_df.loc[
            (self.corpus_df["section"] == section) & (self.corpus_df["type"] == type)
        ]

        contradiction_docs = relevant_docs.loc[
            (relevant_docs["labels"] == "Contradiction")
            & (relevant_docs["id"] != statement_id)
        ]["id"].tolist()
        entailment_docs = relevant_docs.loc[
            (relevant_docs["labels"] == "Entailment")
            & (relevant_docs["id"] != statement_id)
        ]["id"].tolist()
        relevant_contradiction_examples = [
            {"id": s_id, "score": 1.0}
            for s_id in random.sample(contradiction_docs, self.top_k)
        ]
        relevant_entailment_examples = [
            {"id": s_id, "score": 1.0}
            for s_id in random.sample(entailment_docs, self.top_k)
        ]

        return {
            "contradictions": relevant_contradiction_examples,
            "entailments": relevant_entailment_examples,
        }
