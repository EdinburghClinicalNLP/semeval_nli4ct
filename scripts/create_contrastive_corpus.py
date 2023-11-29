import argparse
import json
import random

import pandas as pd
import spacy
from metamap import MetaMap


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input clinical text file",
    )
    parser.add_argument(
        "--metamap_path",
        type=str,
        required=True,
        help="Path to the MetaMap installation",
    )
    args = parser.parse_args()
    return args


# Initialize MetaMap
def initialize_metamap(metamap_path):
    return MetaMap.get_instance(metamap_path)


# Function to get synonyms for an entity using MetaMap
def get_synonyms_for_entity(entity_text, mm_instance):
    synonyms = set()

    # Extract concepts using MetaMap
    concepts, error = mm_instance.extract_concepts([entity_text])

    if not error and concepts:
        for concept in concepts[0]:
            preferred_name = concept.preferred_name
            synonyms.update(concept.synonyms)

    # Remove the original entity text from the set
    synonyms.discard(entity_text)

    return list(synonyms)


# Function to replace entities with synonyms
def replace_entities_with_synonyms(text, nlp, mm_instance):
    doc = nlp(text)

    replaced_texts = []
    for ent in doc.ents:
        # Use a method to get synonyms for the entity text
        synonyms = get_synonyms_for_entity(ent.text, mm_instance)

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
    nlp = spacy.load("en_core_sci_sm")

    # Initialize MetaMap
    mm_instance = initialize_metamap(args.metamap_path)

    # Read text from the specified file
    # Load train data
    df = pd.read_json(args.train_data_path)
    df = df.transpose()
    # Extract statements
    statement_ids = df.index.tolist()
    statements = df.Statement.tolist()

    # Replace entities with synonyms
    modified_statements_dict = dict()
    for statement_id, statement in zip(statement_ids, statements):
        modified_statements = replace_entities_with_synonyms(
            statement, nlp, mm_instance
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
