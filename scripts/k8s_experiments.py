import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml
from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="SemEval NLI4CT experiments")
    parser.add_argument("--run_configs_filepath", type=str, required=True)
    parser.add_argument("--user_email", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.run_configs_filepath, "r"))

    base_args = "git clone https://$GIT_TOKEN@github.com/EdinburghClinicalNLP/semeval_nli4ct.git && cd semeval_nli4ct && git clone https://huggingface.co/datasets/aryopg/nli4ct_practice data && "
    base_command = "python scripts/train.py experiment="

    secret_env_vars = configs["env_vars"]
    commands = {}
    for config in configs["runs"]:
        commands[config.replace("_", "-").replace("/", "-")] = {
            "command": base_command + config["experiment"],
            "gpu_product": config["gpu_product"],
        }

    for run_name, command in commands.items():
        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command['command']}")
        job = KubernetesJob(
            name=run_name,
            image=configs["image"],
            gpu_type="nvidia.com/gpu",
            gpu_limit=configs["gpu_limit"],
            gpu_product=command["gpu_product"],
            backoff_limit=4,
            command=["/bin/bash", "-c", "--"],
            args=[base_args + command["command"]],
            secret_env_vars=secret_env_vars,
            user_email=args.user_email,
        )

        # Run the Job on the Kubernetes cluster
        job.run()


if __name__ == "__main__":
    main()
