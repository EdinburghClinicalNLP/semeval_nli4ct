import argparse
import json
import random

import pandas as pd
import scispacy
import spacy

from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker


def argument_parser():
    parser = argparse.ArgumentParser(description="Contrastive dataset for internal faithfulness test")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input clinical text file",
    )
    args = parser.parse_args()
    return args


# Initialize MetaMap
def initialize_metamap(metamap_path):
    return MetaMap.get_instance(metamap_path)


def get_synonyms_for_entity(entity, linker):
    synonyms = set()

    print(entity._)
    print(entity._.kb_ents)
    # Link entities to their CUI
    for umls_ent in entity._.kb_ents:
        print(umls_ent)
        print(linker.kb.cui_to_entity[umls_ent])


    # Extract concepts using MetaMap
    concepts, error = mm_instance.extract_concepts([entity])

    if not error and concepts:
        for concept in concepts[0]:
            preferred_name = concept.preferred_name
            synonyms.update(concept.synonyms)

    # Remove the original entity text from the set
    synonyms.discard(entity)

    return list(synonyms)


# Function to replace entities with synonyms
def replace_entities_with_synonyms(text, nlp):
    doc = nlp(text)

    replaced_texts = []
    for ent in doc.ents:
        # Skipping generic entity names
        if ent in ["primary", "secondary"]:
            continue

        # Use a method to get synonyms for the entity text
        synonyms = get_synonyms_for_entity(ent, nlp.get_pipe("scispacy_linker"))

        # Replace the entity with a random synonym
        if synonyms:
            replacement = random.choice(synonyms)
            replaced_text = text.replace(ent.text, replacement)
            replaced_texts += [replaced_text]

    return replaced_texts


def main():
    # Parse command line arguments
    args = argument_parser()

    # Load the ScispaCy model
    nlp = spacy.load("en_core_sci_lg")

    # Add the abbreviation pipe to the spacy pipeline.
    # abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe("abbreviation_detector")

    # Add the entity linking pipe to the spacy pipeline
    linker_configs = {
        "resolve_abbreviations": True,
        "threshold": 0.8,
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
    for statement_id, statement in zip(statement_ids, statements):
        modified_statements = replace_entities_with_synonyms(
            statement, nlp
        )
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
