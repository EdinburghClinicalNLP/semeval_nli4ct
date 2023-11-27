from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.configs import ModelConfigs


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_configs.configs.model_name_or_path)

        self.system_prompt = model_configs.configs.system_prompt
        self.system_prompt_len = len(self.tokenizer.encode(self.system_prompt))

        self.max_seq_len = model_configs.configs.max_seq_len

    def generate(
        self,
        inputs,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        # TODO: Mistral doesn't cater system message, GPT-4 is completely different
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": inputs["text"][0]},
        ]

        model_input = self.tokenizer.apply_chat_template(
            prompt, return_tensors="pt", max_length=self.max_seq_len
        )
        max_new_tokens = self.max_seq_len - self.system_prompt_len - model_input.size(1)

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
                num_return_sequences=1
            )
            decoded_text = self.tokenizer.decode(
                output[0, model_input.size(1) :], skip_special_tokens=True
            )

        return decoded_text
