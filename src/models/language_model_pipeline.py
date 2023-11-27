from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.configs import ModelConfigs


class LanguageModelPipeline:
    def __init__(self, model_configs: ModelConfigs):
        self.model_configs = model_configs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.model_name_or_path,
        )
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model.configs["model_name_or_path"]
        )

        if (
            getattr(tokenizer, "pad_token_id") is None
            or getattr(tokenizer, "pad_token") is None
        ):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def generate(
        self,
        inputs,
        max_gen_len: int,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded_text = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return decoded_text
