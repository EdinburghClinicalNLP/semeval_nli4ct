from typing import Dict

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.datasets import RetrieverDataset

from .utils import tokenize_and_stem


class CTRBM25OkapiDenseReranker(BM25Okapi):
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
        dense_retriever_name: str = "michiyasunaga/BioLinkBERT-base",
    ) -> None:
        print("Initialising BM25")
        self.corpus_df = pd.DataFrame(corpus.data)

        self.top_k = top_k

        corpus = []
        for evidence, statement in self.corpus_df[["evidence", "statement"]].values:
            processed_evidence = tokenize_and_stem(evidence)
            processed_statement = tokenize_and_stem(statement)
            corpus += [processed_evidence + processed_statement]

        self.dense_retriever_model = AutoModel.from_pretrained(dense_retriever_name)
        self.dense_retriever_tokenizer = AutoTokenizer.from_pretrained(
            dense_retriever_name
        )
        self.document_embeddings = self.get_document_embeddings()

        super().__init__(corpus, tokenizer, k1, b, epsilon)

    def get_document_embeddings(self):
        # Tokenize the input texts
        corpus = [
            "\n".join([evidence, statement])
            for evidence, statement in self.corpus_df[["evidence", "statement"]].values
        ]
        return self.dense_retriever_forward(corpus)

    def dense_retriever_forward(self, texts):
        inputs = self.dense_retriever_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )

        # Forward pass to get embeddings
        with torch.no_grad():
            outputs = self.dense_retriever_model(**inputs)

        # Extract the embeddings (CLS token embeddings in this case)
        embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.numpy()

    def get_reranked_document_scores(self, query, retrieved_documents):
        # Get 10 * tok_k of retrieval candidates
        retrieved_doc_ids = [
            doc_id for doc_id, _ in retrieved_documents[: self.top_k * 10]
        ]
        retrieved_document_embeddings = self.document_embeddings[retrieved_doc_ids]

        # Get embedding of the query text
        query_embedding = self.dense_retriever_forward([query])

        # Calculate cosine similarity between the query and document embeddings
        similarities = cosine_similarity(
            query_embedding, retrieved_document_embeddings
        )[0]

        # Get indices of top k most similar documents
        top_indices = np.argsort(similarities)[::-1]

        return zip(top_indices, similarities[top_indices])

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

        # Rank documents based on scores
        retrieved_documents = sorted(
            zip(doc_ids, scores), key=lambda x: x[1], reverse=True
        )

        ranked_documents = self.get_reranked_document_scores(query, retrieved_documents)

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
