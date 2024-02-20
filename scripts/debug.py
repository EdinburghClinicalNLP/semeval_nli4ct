import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

from itertools import combinations, product

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.configs import TrainingConfigs, register_base_configs
from src.datasets import ChatDataset


def _tokenize_input_late_coupled_fusion(
    inputs, system_prompt, tokenizer, max_seq_len=4096
):
    def pairs(*lists):
        for t in combinations(lists, 2):
            for pair in product(*t):
                yield pair

    # Separate ICL examples by label
    icl_label_to_inputs = {}
    for icl_input, icl_label in zip(inputs["icl_inputs"], inputs["icl_labels"]):
        if icl_label[0] in icl_label_to_inputs:
            icl_label_to_inputs[icl_label[0]] += [icl_input[0]]
        else:
            icl_label_to_inputs[icl_label[0]] = [icl_input[0]]

    paired_icl_examples = set(
        pairs(icl_label_to_inputs["Entailment"], icl_label_to_inputs["Contradiction"])
    )

    model_inputs = []
    for entailment_example, contradiction_example in paired_icl_examples:
        prompt = []
        if system_prompt:
            prompt += [{"role": "system", "content": system_prompt}]

        if len(inputs["icl_inputs"]) and len(inputs["icl_labels"]):
            prompt += [
                {"role": "user", "content": entailment_example},
                {"role": "assistant", "content": "Entailment"},
                {"role": "user", "content": contradiction_example},
                {"role": "assistant", "content": "Contradiction"},
            ]

        prompt += [{"role": "user", "content": inputs["text"][0]}]

        model_input = tokenizer.apply_chat_template(
            prompt, return_tensors="pt", max_length=max_seq_len
        )

        model_inputs += [model_input]

    return model_inputs


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    print(configs)

    dataset = ChatDataset(
        configs.data,
        configs.instruction,
        configs.trainer,
        icl_examples_dir=configs.retriever.icl_examples_dir,
        tokenizer=None,
        split="test",
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        **configs.dataloader.configs,
    )

    system_prompt = "SYSTEM PROMPT LET'S GO!"

    tokenizer = AutoTokenizer.from_pretrained(configs.model.configs.model_name_or_path)

    print("Starting loop")
    for step, batch in enumerate(dataloader):
        print(batch)
        if step > 5:
            break
        # if "_pos" in batch["id"][0] or "_neg" in batch["id"][0]:
        #     print(f"perturbed sample: {batch['id']}")
        # else:
        #     print(f"non-perturbed sample: {batch['id']}")
        # prompt = [{"role": "system", "content": system_prompt}]

        # if len(batch["icl_inputs"]) and len(batch["icl_labels"]):
        #     for icl_input, icl_label in zip(batch["icl_inputs"], batch["icl_labels"]):
        #         print("icl_input: ", icl_input)
        #         print("icl_label: ", icl_label)
        #         prompt += [
        #             {"role": "user", "content": icl_input[0]},
        #             {"role": "assistant", "content": icl_label[0]},
        #         ]

        # prompt += [{"role": "user", "content": batch["text"][0]}]

        # tokenized_inputs = _tokenize_input_late_coupled_fusion(batch, None, tokenizer)
        # print(len(tokenized_inputs))


if __name__ == "__main__":
    register_base_configs()
    main()
