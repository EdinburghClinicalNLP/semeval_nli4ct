import argparse
import json
from typing import List

import pandas as pd
import scispacy
import spacy
from tqdm import tqdm

ALLOCATED_KEYWORDS = {
    "primary",
    "secondary",
    "primary trial",
    "secondary trial",
    "eligibility",
    "adverse event",
    "adverse events",
    "intervention",
    "result",
    "results",
    "participant",
    "participants",
}


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Contrastive dataset for internal faithfulness test"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input clinical text file",
    )
    args = parser.parse_args()
    return args


def get_synonyms_for_entity(entity, linker):
    def is_valid_alias(text):
        if "," in text:
            return False
        if "(" in text or ")" in text:
            return False
        if "(" in text:
            return False
        return True

    synonyms = set()

    # Link entities to their CUI
    for umls_ent in entity._.kb_ents:
        # umls_ent is a Tuple(CUI, confidence_score)
        linked_entity = linker.kb.cui_to_entity[umls_ent[0]]
        # Update to the synonyms set
        for alias in linked_entity.aliases:
            if is_valid_alias(alias):
                synonyms.update([alias.lower()])

    # Remove the original entity text from the set
    synonyms.discard(entity.text.lower())

    return list(synonyms)


# Function to replace entities with synonyms
def replace_entities_with_synonyms(text, nlp):
    doc = nlp(text)

    replaced_texts = []
    for ent in doc.ents:
        # Skipping generic entity names, just in case
        if ent.text.lower() in ALLOCATED_KEYWORDS:
            continue

        # Use a method to get synonyms for the entity text
        synonyms = get_synonyms_for_entity(ent, nlp.get_pipe("scispacy_linker"))

        # Replace the entity with a random synonym
        if synonyms:
            for synonym in synonyms:
                replaced_text = text.replace(ent.text, synonym)
                replaced_texts += [replaced_text]

    return replaced_texts


def main():
    # Parse command line arguments
    args = argument_parser()

    # Load the ScispaCy model:
    # BC5CDR model only extracts chemical and diseases
    nlp = spacy.load("en_ner_bc5cdr_md")

    # Add the abbreviation pipe to the spacy pipeline.
    nlp.add_pipe("abbreviation_detector")

    # Add the entity linking pipe to the spacy pipeline
    linker_configs = {
        "resolve_abbreviations": True,
        "threshold": 0.85,
        "max_entities_per_mention": 1,
        "linker_name": "umls",
    }
    nlp.add_pipe("scispacy_linker", config=linker_configs)

    # Read text from the specified file
    # Load train data
    df = pd.read_json(args.data_path)
    df = df.transpose()
    # Extract statements
    statement_ids = df.index.tolist()
    statements = df.Statement.tolist()

    # Replace entities with synonyms
    modified_statements_dict = dict()
    for statement_id, statement in tqdm(
        zip(statement_ids, statements), total=len(statement_ids)
    ):
        modified_statements: List[str] = replace_entities_with_synonyms(statement, nlp)
        modified_statements_dict[statement_id] = {
            "original": statement,
            "modified": modified_statements,
        }

    # Save the dictionary to a JSON file
    data_name = args.data_path.split("/")[-1].replace(".json", "")
    with open(f"{data_name}_synonyms.json", "w") as json_file:
        json.dump(modified_statements_dict, json_file)


if __name__ == "__main__":
    main()
