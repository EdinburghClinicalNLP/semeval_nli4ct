import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

from itertools import combinations, product

import huggingface_hub
import hydra
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import TrainingConfigs, register_base_configs
from src.datasets import ChatDataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

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

    tokenizer = AutoTokenizer.from_pretrained(configs.model.configs.model_name_or_path)

    adapter_model = "aryopg/Mistral-7b-Instruct__fine_tune__explicit__multi_adapter"

    model = AutoModelForCausalLM.from_pretrained(
        configs.model.configs.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_model)
    print(model)

    # print("Starting loop")
    # for step, batch in enumerate(dataloader):
    #     print(batch)

    #     if step > 5:
    #         break
    #     # if "_pos" in batch["id"][0] or "_neg" in batch["id"][0]:
    #     #     print(f"perturbed sample: {batch['id']}")
    #     # else:
    #     #     print(f"non-perturbed sample: {batch['id']}")
    #     # prompt = [{"role": "system", "content": system_prompt}]

    #     # if len(batch["icl_inputs"]) and len(batch["icl_labels"]):
    #     #     for icl_input, icl_label in zip(batch["icl_inputs"], batch["icl_labels"]):
    #     #         print("icl_input: ", icl_input)
    #     #         print("icl_label: ", icl_label)
    #     #         prompt += [
    #     #             {"role": "user", "content": icl_input[0]},
    #     #             {"role": "assistant", "content": icl_label[0]},
    #     #         ]

    #     # prompt += [{"role": "user", "content": batch["text"][0]}]

    #     # tokenized_inputs = _tokenize_input_late_coupled_fusion(batch, None, tokenizer)
    #     # print(len(tokenized_inputs))


if __name__ == "__main__":
    register_base_configs()
    main()
