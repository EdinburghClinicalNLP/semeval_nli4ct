import re
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.configs import ModelConfigs
from src.factories import get_peft_config


class ChatModelPipeline:
    def __init__(self, model_configs: ModelConfigs):
        self.model_configs = model_configs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_configs.configs.model_name_or_path
        )

        self.system_prompt = model_configs.configs.system_prompt

        self.max_seq_len = model_configs.configs.max_seq_len

    def _tokenize_input(self, inputs):
        if "mistral" in self.model_configs.configs.model_name_or_path.lower():
            # Mistral doesn't allow system role
            prompt = [
                {"role": "user", "content": self.system_prompt + inputs["text"][0]},
            ]
        else:
            prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": inputs["text"][0]},
            ]

        model_input = self.tokenizer.apply_chat_template(
            prompt, return_tensors="pt", max_length=self.max_seq_len
        ).to(self.model.device)

        return model_input

    def setup_finetuning(self, peft_configs: dict):
        self.peft_config = LoraConfig(
            **peft_configs,
            task_type=TaskType.CausalLM,
        )
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def train(self, inputs, labels):
        self.model.train()

        # Tokenize the input
        model_input = self._tokenize_input(inputs)

        # Tokenize labels
        labels = self.tokenizer(labels, add_special_tokens=False).to(self.model.device)

        # Concatenate the input and labels to form the Language Model labels
        model_input["labels"] = (
            model_input["input_ids"]
            + labels["input_ids"]
            + [self.tokenizer.eos_token_id]
        )

        # Forward pass
        outputs = self.model(**model_input)

        return outputs

    def generate(
        self,
        inputs,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        model_input = self._tokenize_input(inputs)
        # Limit generation length
        max_new_tokens = min(256, self.max_seq_len - model_input.size(1))

        with torch.inference_mode():
            output = self.model.generate(
                model_input,
                temperature=self.model_configs.configs.temperature,
                top_p=self.model_configs.configs.top_p,
                top_k=self.model_configs.configs.top_k,
                repetition_penalty=self.model_configs.configs.repetition_penalty,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
            decoded_text = self.tokenizer.decode(
                output[0, model_input.size(1) :], skip_special_tokens=True
            )

        return {
            "decoded_text": decoded_text,
            "input_length": model_input.size(1),
            "max_new_tokens": max_new_tokens,
        }

    @staticmethod
    def postprocess_prediction(answer):
        """
        If the output is structured correctly, the first word will be the answer.
        If not, take note that it cannot be parsed correctly.
        """
        final_answer = re.split("[\.\s\:\n]+", answer.lower().strip())[0].strip()

        if final_answer in ["contradiction", "entailment"]:
            # Return the first word
            return final_answer
        else:
            # Return original answer if the string is empty
            return answer
