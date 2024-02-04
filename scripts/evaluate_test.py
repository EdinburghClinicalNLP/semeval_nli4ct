import argparse
import json
import os
import sys
import warnings
import zipfile

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter("ignore")


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the submissions on the test set"
    )
    parser.add_argument(
        "--submissions_dir",
        type=str,
        required=True,
        help="Path to the submission files directory",
    )
    args = parser.parse_args()
    return args


def extract_control_set(predictions, gold):
    control_predicitons = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            control_predicitons[key] = predictions[key]
    return control_predicitons


def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]
    return (
        para_predictions,
        cont_predictions,
        numerical_para_predictions,
        numerical_cont_predictions,
        definitions_predictions,
    )


def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering


def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness


def consistency(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] == gold[key]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency


def extract_contrast_set(predictions, gold):
    contrast_predicitons = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predicitons[key] = predictions[key]
    return contrast_predicitons


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = precision_score(gold_labels, pred_labels)
    Precision = recall_score(gold_labels, pred_labels)
    return F1, Recall, Precision


def evaluate(gold, predictions):
    # Control Test Set F1, Recall, Precision PUBLIC
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(
        extract_control_set(predictions, gold), gold
    )

    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(
        contrast_predictions, gold
    )
    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, gold)

    # Intervention-wise Consistency & Faithfullness HIDDEN
    (
        para_predictions,
        cont_predictions,
        numerical_para_predictions,
        numerical_cont_predictions,
        definitions_predictions,
    ) = extract_by_intervention(predictions, gold)
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    numerical_para_preserving = extract_by_causal_type(
        numerical_para_predictions, gold
    )[0]
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(
        numerical_cont_predictions, gold
    )
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    para_Consistency = consistency(para_preserving, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, gold)
    definitions_Consistency = consistency(definitions_preserving, gold)

    # Intervention-wise F1, Recall, Precision HIDDEN
    Contrast_F1, Contrast_Rec, Contrast_Prec = F1_Recall_Precision(
        contrast_predictions, gold
    )
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)
    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(
        numerical_para_predictions, gold
    )
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(
        numerical_cont_predictions, gold
    )
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(
        definitions_predictions, gold
    )

    return {
        "Control_F1": [Control_F1],
        "Control_Rec": [Control_Rec],
        "Control_Prec": [Control_Prec],
        "Contrast_F1": [Contrast_F1],
        "Contrast_Rec": [Contrast_Rec],
        "Contrast_Prec": [Contrast_Prec],
        "Faithfulness": [Faithfulness],
        "Consistency": [Consistency],
        "para_Consistency": [para_Consistency],
        "cont_Faithfulness": [cont_Faithfulness],
        "cont_Consistency": [cont_Consistency],
        "numerical_para_Consistency": [numerical_para_Consistency],
        "numerical_cont_Faithfulness": [numerical_cont_Faithfulness],
        "numerical_cont_Consistency": [numerical_cont_Consistency],
        "definitions_Consistency": [definitions_Consistency],
        "para_F1": [para_F1],
        "para_Rec": [para_Rec],
        "para_Prec": [para_Prec],
        "cont_F1": [cont_F1],
        "cont_Rec": [cont_Rec],
        "cont_Prec": [cont_Prec],
        "numerical_para_F1": [numerical_para_F1],
        "numerical_para_Rec": [numerical_para_Rec],
        "numerical_para_Prec": [numerical_para_Prec],
        "numerical_cont_F1": [numerical_cont_F1],
        "numerical_cont_Rec": [numerical_cont_Rec],
        "numerical_cont_Prec": [numerical_cont_Prec],
        "definitions_F1": [definitions_F1],
        "definitions_Rec": [definitions_Rec],
        "definitions_Prec": [definitions_Prec],
    }


def main():
    submissions_dir = argument_parser().submissions_dir

    # Load files
    gold_filename = os.path.join("data/gold_test.json")
    with open(gold_filename) as json_file:
        gold = json.load(json_file)

    # Evaluate
    df = pd.DataFrame(
        columns=[
            "Model",
            "Control_F1",
            "Control_Rec",
            "Control_Prec",
            "Contrast_F1",
            "Contrast_Rec",
            "Contrast_Prec",
            "Faithfulness",
            "Consistency",
            "para_Consistency",
            "cont_Faithfulness",
            "cont_Consistency",
            "numerical_para_Consistency",
            "numerical_cont_Faithfulness",
            "numerical_cont_Consistency",
            "definitions_Consistency",
            "para_F1",
            "para_Rec",
            "para_Prec",
            "cont_F1",
            "cont_Rec",
            "cont_Prec",
            "numerical_para_F1",
            "numerical_para_Rec",
            "numerical_para_Prec",
            "numerical_cont_F1",
            "numerical_cont_Rec",
            "numerical_cont_Prec",
            "definitions_F1",
            "definitions_Rec",
            "definitions_Prec",
        ]
    )
    for filepath in os.listdir(submissions_dir):
        if filepath.endswith(".zip"):
            submission_filepath = os.path.join(submissions_dir, filepath)
        else:
            continue

        with zipfile.ZipFile(submission_filepath, "r") as zip_file:
            with zip_file.open("results.json") as json_file:
                predictions = json.load(json_file)

        metrics = evaluate(gold, predictions)
        model = filepath.split(".")[0]

        metrics["Model"] = [model]

        model_results = pd.DataFrame(metrics)
        df = pd.concat([df, model_results])
    df.to_csv("evaluation_.csv", index=False)


if "__main__" == __name__:
    main()
