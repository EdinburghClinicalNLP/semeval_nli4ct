import argparse
import json
import zipfile

import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Prepare output file from the WandB output table"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the predictions JSON file",
    )
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    with open(args.data_path, "r") as f:
        data = json.load(f)
    data = pd.DataFrame(data["data"], columns=data["columns"])

    # result should be like this
    # "dbed5471-c2fc-45b5-b26f-430c9fa37a37": {
    #     "Prediction": "Entailment"
    # },
    submission = {}
    for idx, row in data.iterrows():
        if row["predictions"] not in ["entailment", "contradiction"]:
            raise ValueError(f"Invalid prediction {row['predictions']}")
        submission[row["id"]] = {"Prediction": row["predictions"].capitalize()}

    # Save submission dict into a JSON file
    with open("results.json", "w") as f:
        json.dump(submission, f)

    # Zip the JSON file, name the zip file as the original filepath (args.data_path) with .zip extension
    with zipfile.ZipFile(args.data_path.replace(".json", ".zip"), "w") as f:
        f.write("results.json")


if __name__ == "__main__":
    main()
