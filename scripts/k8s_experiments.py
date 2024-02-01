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
    parser.add_argument("--git_branch", type=str, default="main")
    parser.add_argument("--namespace", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.run_configs_filepath, "r"))

    base_args = f"pip uninstall -y huggingface_hub && pip install huggingface_hub && pip install -U git+https://github.com/huggingface/peft.git && git clone https://$GIT_TOKEN@github.com/EdinburghClinicalNLP/semeval_nli4ct.git --branch {args.git_branch} && cd semeval_nli4ct && huggingface-cli download aryopg/nli4ct_practice --repo-type dataset --local-dir data --token $HF_DOWNLOAD_TOKEN --quiet && "
    base_command = "python scripts/train.py experiment="

    secret_env_vars = configs["env_vars"]
    commands = {}
    for run in configs["runs"]:
        if "retriever" in run["experiment"]:
            run_name = (
                run["experiment"]
                .replace("_", "-")
                .replace("/", "-")
                .replace(" retriever=", "-")
            )
        else:
            run_name = run["experiment"].replace("_", "-").replace("/", "-")
        commands[run_name] = {
            "command": base_command + run["experiment"],
            "gpu_product": run["gpu_product"],
        }

    for run_name, command in commands.items():
        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command['command']}")
        job = KubernetesJob(
            name=run_name[:63],
            cpu_request="8",
            ram_request="64Gi",
            image=configs["image"],
            gpu_type="nvidia.com/gpu",
            gpu_limit=command["gpu_limit"],
            gpu_product=command["gpu_product"],
            backoff_limit=4,
            command=["/bin/bash", "-c", "--"],
            args=[base_args + command["command"]],
            secret_env_vars=secret_env_vars,
            user_email=args.user_email,
            namespace=args.namespace,
        )

        # Run the Job on the Kubernetes cluster
        job.run()


if __name__ == "__main__":
    main()
