from typing import Dict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.datasets import RetrieverDataset

from .utils import tokenize_and_stem


class CTRBM25OkapiLengthPenalty(BM25Okapi):
    """
    Slightly modified version of the BM25Okapi that takes into consideration the statement id, type, and section

    """

    def __init__(
        self,
        corpus: RetrieverDataset,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        top_k=5,
    ) -> None:
        print("Initialising BM25 + length penalty")
        self.corpus_df = pd.DataFrame(corpus.data)

        self.top_k = top_k

        corpus = [
            "\n".join([evidence, statement])
            for evidence, statement in self.corpus_df[["evidence", "statement"]].values
        ]

        super().__init__(corpus, tokenizer, k1, b, epsilon)

    def compute_length_penalties(self, processed_query, doc_ids):
        """
        $$
        penalty(x, D_i) = \frac{\alpha (avg(|D|)) + avg(|S|) - |x|}{D_i} - 1
        $$
        """
        query_length = len(processed_query)
        document_lengths = np.array(self.doc_len)[doc_ids]

        penalties = (2 * self.avgdl - query_length) / document_lengths - 1

        return penalties

    @staticmethod
    def normalise_scores(scores):
        """
        Min-max scaling between 0 - 1
        """
        min_val = np.min(scores)
        max_val = np.max(scores)

        # Calculate the scaling parameters
        scale = 1 / (max_val - min_val)

        normalised_scores = (scores - min_val) * scale

        return normalised_scores

    def get_document_scores(
        self, query, section, type, statement_id
    ) -> Dict[str, list]:
        # Given the section and type, narrow down the search space
        relevant_docs = self.corpus_df.loc[
            (self.corpus_df["section"] == section) & (self.corpus_df["type"] == type)
        ]

        doc_ids = relevant_docs.index.tolist()

        # Tokenize and stem query
        processed_query = tokenize_and_stem(query)

        # Get BM25 score
        scores = self.get_batch_scores(processed_query, doc_ids)
        # Normalise score
        normalised_scores = self.normalise_scores(scores)
        # Penalise score by document length
        penalties = self.compute_length_penalties(processed_query, doc_ids)
        penalised_scores = normalised_scores - penalties

        # Rank documents based on scores
        ranked_documents = sorted(
            zip(doc_ids, penalised_scores), key=lambda x: x[1], reverse=True
        )

        # Print the most similar documents
        # k examples for contradiction and entailment
        relevant_contradiction_examples = []
        relevant_entailment_examples = []
        for idx, score in ranked_documents:
            doc = self.corpus_df.iloc[idx]
            if statement_id == doc["id"]:
                # Filter out the sentence itself if found in the ranking
                continue
            else:
                if doc["labels"].lower() == "contradiction":
                    # Take only top k examples per label
                    if len(relevant_contradiction_examples) >= self.top_k:
                        continue
                    relevant_contradiction_examples += [
                        {
                            "id": doc["id"],
                            "score": score,
                        }
                    ]
                elif doc["labels"].lower() == "entailment":
                    # Take only top k examples per label
                    if len(relevant_entailment_examples) >= self.top_k:
                        continue
                    relevant_entailment_examples += [
                        {
                            "id": doc["id"],
                            "score": score,
                        }
                    ]

            # Take only top k examples
            if (
                len(relevant_contradiction_examples) >= self.top_k
                and len(relevant_entailment_examples) >= self.top_k
            ):
                break

        return {
            "contradictions": relevant_contradiction_examples,
            "entailments": relevant_entailment_examples,
        }
