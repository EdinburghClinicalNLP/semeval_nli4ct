import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")
from typing import List

import pandas as pd
import spacy
from openai import AzureOpenAI
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
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


def generate_paraphrases(openai_client, statement, paraphrase_type):
    prompts = [
        {
            "role": "system",
            "content": "You are a helpful clinician's assistant designed to paraphrase clinical statements and to output a YAML list",
        },
    ]

    if paraphrase_type == "affirmative":
        prompts += [
            {
                "role": "user",
                "content": f"Create 5 paraphrased statements of: '{statement}'",
            },
        ]
    elif paraphrase_type == "full_negation":
        prompts += [
            {
                "role": "user",
                "content": f"Create 5 full negation statements of: '{statement}'",
            },
        ]

    response = openai_client.chat.completions.create(
        model="chatgpt_icd_coding",
        messages=prompts,
        temperature=0,
        top_p=0,
        max_tokens=1000,
        frequency_penalty=0,
        presence_penalty=0,
    )

    paraphrased_statements = response.choices[0].message.content

    # Split the string into a list of sentences
    paraphrased_statements = paraphrased_statements.split("\n")

    # Remove the leading '- ' from each sentence
    paraphrased_statements = [statement[2:] for statement in paraphrased_statements]

    return paraphrased_statements


def get_synonymised_statements(
    modified_statements, new_label, original_row, original_id, new_id_suffix
):
    modified_statements_df = []
    for idx, modified_statement in enumerate(modified_statements):
        modified_statements_df += [
            {
                "id": original_id + new_id_suffix + idx,
                "Type": original_row["Type"],
                "Section_id": original_row["Section_id"],
                "Primary_id": original_row["Primary_id"],
                "Secondary_id": original_row["Secondary_id"],
                "Statement": modified_statement,
                "Label": new_label,
            }
        ]

    return modified_statements_df


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

    # OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Read text from the specified file
    # Load train data
    df = pd.read_json(args.data_path)
    df = df.transpose()

    # Replace entities with synonyms
    modified_df = pd.DataFrame(columns=list(df.columns))
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement_id = index
        statement = row["Statement"]
        label = row["Label"]

        # Generate paraphrases
        affirmative_statements = generate_paraphrases(
            openai_client, statement, paraphrase_type="affirmative"
        )
        full_negated_statements = generate_paraphrases(
            openai_client, statement, paraphrase_type="full_negation"
        )

        negated_label = "Contradiction" if label == "Entailment" else "Entailment"

        # Insert the original in the dataframe
        modified_df = pd.concat([modified_df, pd.DataFrame(row).transpose()])

        # Replace entities with synonyms for the original statement
        modified_statements = replace_entities_with_synonyms(statement, nlp)
        modified_statements_df = get_synonymised_statements(
            modified_statements, label, row, statement_id, ""
        )
        modified_df = pd.concat([modified_df, pd.DataFrame(modified_statements_df)])

        # Replace entities with synonyms for each paraphrase
        for statement in affirmative_statements:
            modified_statements = replace_entities_with_synonyms(statement, nlp)
            modified_statements_df = get_synonymised_statements(
                modified_statements, label, row, statement_id, "_pos"
            )
            modified_df = pd.concat([modified_df, pd.DataFrame(modified_statements_df)])

        for statement in full_negated_statements:
            modified_statements = replace_entities_with_synonyms(statement, nlp)
            modified_statements_df = get_synonymised_statements(
                modified_statements, negated_label, row, statement_id, "_neg"
            )
            modified_df = pd.concat([modified_df, pd.DataFrame(modified_statements_df)])

    # Save to csv file
    data_paths = args.data_path.split("/")
    data_dir = "/".join(data_paths[:-1])
    data_name = data_paths[-1].replace(".json", "")
    modified_df.to_csv(os.path.join(data_dir, f"{data_name}_paraphrased.csv"))


if __name__ == "__main__":
    main()
