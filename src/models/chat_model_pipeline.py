import re
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.configs import ModelConfigs


class ChatModelPipeline:
    def __init__(self, model_configs: ModelConfigs):
        self.model_configs = model_configs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_configs.configs.model_name_or_path
        )

        self.system_prompt = None
        if "llama" in self.model_configs.configs.model_name_or_path.lower():
            self.system_prompt = model_configs.configs.system_prompt

        self.max_seq_len = model_configs.configs.max_seq_len

    def _tokenize_input(self, inputs):
        prompt = []
        if self.system_prompt:
            prompt += [{"role": "system", "content": self.system_prompt}]

        if len(inputs["icl_inputs"]) and len(inputs["icl_labels"]):
            for icl_input, icl_label in zip(inputs["icl_inputs"], inputs["icl_labels"]):
                prompt += [
                    {"role": "user", "content": icl_input[0]},
                    {"role": "assistant", "content": icl_label[0]},
                ]

        prompt += [{"role": "user", "content": inputs["text"][0]}]

        model_input = self.tokenizer.apply_chat_template(
            prompt, return_tensors="pt", max_length=self.max_seq_len
        )

        return model_input

    def setup_finetuning(self, peft_configs: dict):
        self.peft_config = LoraConfig(
            **peft_configs,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def train(self, inputs, labels, max_train_seq_len=None):
        if max_train_seq_len is None:
            max_train_seq_len = self.max_seq_len // 2

        self.model.train()

        # Tokenize the input
        tokenized_input = self._tokenize_input(inputs)
        labels = self.tokenizer(labels, return_tensors="pt", add_special_tokens=False)

        # Tokenize labels
        eos_tensor = torch.tensor([[self.tokenizer.eos_token_id]])
        label_input_ids = torch.cat((labels["input_ids"], eos_tensor), dim=1)
        model_input = torch.cat((tokenized_input, label_input_ids), dim=1)
        labels = torch.cat(
            (torch.tensor([[-100] * tokenized_input.size(1)]), label_input_ids), dim=1
        )
        attention_mask = torch.tensor([[1] * model_input.size(1)])

        # Truncate left side to max length
        model_input = model_input[:, -max_train_seq_len:]
        attention_mask = attention_mask[:, -max_train_seq_len:]
        labels = labels[:, -max_train_seq_len:]

        # Move to device
        model_input = model_input.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        labels = labels.to(self.model.device)

        # Forward pass
        outputs = self.model(
            input_ids=model_input, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def generate(
        self,
        inputs,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        model_input = self._tokenize_input(inputs).to(self.model.device)
        # Limit generation length
        max_new_tokens = 8
        if self.max_seq_len - model_input.size(1) > 4:
            max_new_tokens = min(8, self.max_seq_len - model_input.size(1))

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=model_input,
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
        Take the last occurence between contradiction and entailment
        """
        stripped_answer = answer.lower().strip()

        contradiction_loc = stripped_answer.rfind("contradiction")
        entailment_loc = stripped_answer.rfind("entailment")

        if entailment_loc > contradiction_loc:
            return "entailment"
        else:
            return "contradiction"
