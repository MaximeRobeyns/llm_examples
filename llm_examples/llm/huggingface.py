# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Wrapper around huggingface transformers for simple generation interface
"""

import torch as t
import logging
import transformers

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from copy import copy
from typing import Optional
from omegaconf import OmegaConf
from transformers import GenerationConfig, BitsAndBytesConfig

from llm_examples.llm.base import LLM
from llm_examples.utils import (
    get_fst_device,
    str_to_torch_dtype,
    prepare_model_for_quantized_training,
)


class HuggingFaceLLM(LLM):
    def __init__(
        self,
        model_name_or_path: str,
        name: Optional[str] = None,
        config_class: str = "AutoConfig",
        config_kwargs: dict = dict(),
        tokenizer_class: str = "AutoTokenizer",
        tokenizer_kwargs: dict = dict(),
        tokenizer_special_tokens: dict = dict(),
        model_class: str = "AutoModelForCausalLM",
        model_kwargs: dict = dict(),
        global_gen_kwargs: dict = dict(),
        use_peft: bool = False,
        peft: Optional[LoraConfig] = None,
        quantization: Optional[BitsAndBytesConfig] = None,
    ):

        if name is None:
            name = model_name_or_path
        super().__init__(name)

        # Load the HF model config
        config_cls = getattr(transformers, config_class)
        if not isinstance(config_kwargs, dict):
            config_kwargs = OmegaConf.to_object(config_kwargs)
        model_config = config_cls.from_pretrained(model_name_or_path, **config_kwargs)

        # Load the HF model
        model_cls = getattr(transformers, model_class)
        try:
            model_kwargs = OmegaConf.to_object(model_kwargs)
        except Exception:
            pass
        assert isinstance(model_kwargs, dict)
        for k, v in model_kwargs.items():
            if "dtype" in k.lower() and v != "auto":
                model_kwargs[k] = str_to_torch_dtype(v)
        if quantization is not None:
            model_kwargs["quantization_config"] = quantization
        self.model = model_cls.from_pretrained(
            model_name_or_path, config=model_config, **model_kwargs
        )
        if quantization is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure PEFT if required
        if use_peft:
            logging.info("Using PEFT")
            self.peft_config = peft
            self.model = get_peft_model(self.model, self.peft_config)

        # Load HF tokenizer
        tokenizer_cls = getattr(transformers, tokenizer_class)
        if not isinstance(tokenizer_kwargs, dict):
            tokenizer_kwargs = OmegaConf.to_object(tokenizer_kwargs)
        self.tokenizer = tokenizer_cls.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        tokenizer_special_tokens = {
            k: getattr(self.tokenizer, v.split(".")[-1])
            if isinstance(v, str) and v.startswith("tokenizer")
            else v
            for k, v in tokenizer_special_tokens.items()
        }
        if len(tokenizer_special_tokens) > 0:
            self.tokenizer.add_special_tokens(tokenizer_special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the global generation config
        if not isinstance(global_gen_kwargs, dict):
            global_gen_kwargs = OmegaConf.to_object(global_gen_kwargs)
        self.gen_cfg = GenerationConfig.from_pretrained(
            model_name_or_path, **global_gen_kwargs
        )

    def generate(
        self, prompt: str | list[str], num_samples: int = 1, **kwargs
    ) -> list[str] | list[list[str]]:
        """A high-level generation function for HuggingFace transformers.

        If you need more control, use the underlying `model` and `tokenizer`.
        """
        device = get_fst_device(self.model)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        for k in inputs.keys():
            if isinstance(inputs[k], t.Tensor):
                inputs[k] = inputs[k].repeat_interleave(num_samples, 0)
        input_len = inputs.input_ids.size(-1)
        gen_cfg = copy(self.gen_cfg)
        setattr(gen_cfg, "pad_token_id", self.tokenizer.eos_token_id)
        for k, v in kwargs.items():
            setattr(gen_cfg, k, v)
        with t.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)
        gen_tokens = outputs[:, input_len:]
        outputs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        if inputs.input_ids.size(0) == 1:
            return outputs
        return [
            outputs[i : i + num_samples] for i in range(0, len(outputs), num_samples)
        ]

    def generate_seq_2_seq(
        self, enc_prompt: str, dec_prompt: str, **kwargs
    ) -> list[str]:
        """Generate for sequence 2 sequence models"""

        # TODO: update to reflect changes in `generate` above

        # Assumes both encoder and decoder are on the same device
        device = get_fst_device(self.model)
        enc_ids = self.tokenizer(enc_prompt, return_tensors="pt").to(device)
        dec_ids = self.tokenizer(dec_prompt, return_tensors="pt").input_ids.to(device)
        if hasattr(self.model, "_shift_right"):
            # This is used to get rid of the EOS token in the decoder input
            # sequence, to allow for autoregressive generation.
            dec_ids = self.model._shift_right(dec_ids)
        inputs = enc_ids | {"decoder_input_ids": dec_ids}
        input_len = dec_ids.size(-1)
        with t.inference_mode():
            outputs = self.model.generate(
                **inputs, generation_config=self.gen_cfg, **kwargs
            )
        gen_tokens = outputs[:, input_len:]
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
