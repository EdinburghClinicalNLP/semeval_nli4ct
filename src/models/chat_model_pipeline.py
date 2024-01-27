from itertools import combinations, product
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf
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

        # Setup the flags for ease during prediction
        self.use_common_lora = True if self.model_configs.common_lora_config else False
        self.use_section_lora = (
            True if self.model_configs.section_lora_config else False
        )

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

        return [model_input]

    def _tokenize_input_late_fusion(self, inputs):
        model_inputs = []
        for icl_input, icl_label in zip(inputs["icl_inputs"], inputs["icl_labels"]):
            prompt = []
            if self.system_prompt:
                prompt += [{"role": "system", "content": self.system_prompt}]

            if len(inputs["icl_inputs"]) and len(inputs["icl_labels"]):
                prompt += [
                    {"role": "user", "content": icl_input[0]},
                    {"role": "assistant", "content": icl_label[0]},
                ]

            prompt += [{"role": "user", "content": inputs["text"][0]}]

            model_input = self.tokenizer.apply_chat_template(
                prompt, return_tensors="pt", max_length=self.max_seq_len
            )

            model_inputs += [model_input]

        return model_inputs

    def _tokenize_input_late_coupled_fusion(self, inputs):
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
            pairs(
                icl_label_to_inputs["Entailment"], icl_label_to_inputs["Contradiction"]
            )
        )

        model_inputs = []
        for entailment_example, contradiction_example in paired_icl_examples:
            prompt = []
            if self.system_prompt:
                prompt += [{"role": "system", "content": self.system_prompt}]

            if len(inputs["icl_inputs"]) and len(inputs["icl_labels"]):
                prompt += [
                    {"role": "user", "content": entailment_example},
                    {"role": "assistant", "content": "Entailment"},
                    {"role": "user", "content": contradiction_example},
                    {"role": "assistant", "content": "Contradiction"},
                ]

            prompt += [{"role": "user", "content": inputs["text"][0]}]

            model_input = self.tokenizer.apply_chat_template(
                prompt, return_tensors="pt", max_length=self.max_seq_len
            )

            model_inputs += [model_input]

        return model_inputs

    def setup_finetuning(self):
        if self.model_configs.common_lora_config:
            # Handle Hydra serialisation
            common_lora_config = OmegaConf.to_container(
                self.model_configs.common_lora_config
            )
            lora_config = LoraConfig(
                **common_lora_config,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config, adapter_name="common")

        if self.model_configs.section_lora_config:
            sections = ["intervention", "eligibility", "results", "adverse_events"]

            # Handle Hydra serialisation
            section_lora_config = OmegaConf.to_container(
                self.model_configs.section_lora_config
            )
            lora_config = LoraConfig(
                **section_lora_config,
                task_type=TaskType.CAUSAL_LM,
            )
            # Check if common lora has been added
            if not self.model_configs.common_lora_config:
                self.model = get_peft_model(
                    self.model, lora_config, adapter_name=sections[0]
                )
            else:
                self.model.add_adapter(
                    adapter_name=sections[0], peft_config=lora_config
                )

            for section in sections[1:]:
                self.model.add_adapter(adapter_name=section, peft_config=lora_config)

        self.model.print_trainable_parameters()

    def train(self, inputs, labels, max_train_seq_len: int = None):
        if max_train_seq_len is None:
            max_train_seq_len = self.max_seq_len // 2

        self.model.train()

        # Tokenize the input
        tokenized_input = self._tokenize_input(inputs)[0]
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
        adapters_in_use = []
        if self.use_common_lora:
            adapters_in_use += ["common"]
        if self.use_section_lora:
            section_name = inputs["section"][0].lower().replace(" ", "_")
            adapters_in_use += [section_name]

        self.model.set_adapter(adapters_in_use)

        outputs = self.model(
            input_ids=model_input, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def generate(
        self,
        inputs,
        fusion_strategy,
        use_cot=False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        if fusion_strategy == "late":
            model_inputs = self._tokenize_input_late_fusion(inputs)
        elif fusion_strategy == "late_coupled":
            model_inputs = self._tokenize_input_late_coupled_fusion(inputs)
        else:
            model_inputs = self._tokenize_input(inputs)

        decoded_texts = []
        for model_input in model_inputs:
            # Limit generation length
            if use_cot:
                # CoT needs longer generation length
                max_new_tokens = 100
                if self.max_seq_len - model_input.size(1) > 4:
                    max_new_tokens = self.max_seq_len - model_input.size(1)
            else:
                max_new_tokens = 8

            # Predict
            with torch.inference_mode():
                model_input = model_input.to(self.model.device)

                adapters_in_use = []
                if self.use_common_lora:
                    adapters_in_use += ["common"]
                if self.use_section_lora:
                    section_name = inputs["section"][0].lower().replace(" ", "_")
                    adapters_in_use += [section_name]

                output = self.model.generate(
                    input_ids=model_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                decoded_text = self.tokenizer.decode(
                    output[0, model_input.size(1) :], skip_special_tokens=True
                )
                decoded_texts += [decoded_text]

        # Postprocess predictions
        if fusion_strategy.startswith("late"):
            predictions = [
                self.postprocess_prediction(decoded_text)
                for decoded_text in decoded_texts
            ]
            prediction = max(set(predictions), key=predictions.count)
        else:
            prediction = self.postprocess_prediction(decoded_texts[0])

        return {
            "decoded_text": decoded_texts,
            "input_length": [model_input.size(1) for model_input in model_inputs],
            "max_new_tokens": max_new_tokens,
            "prediction": prediction,
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
