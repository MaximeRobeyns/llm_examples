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
"""Thin wrapper around Anthopic's API"""

import os
import anthropic

from ft_example.llm.base import LLM


class AnthropicLLM(LLM):
    def __init__(self, name: str, model: str):
        super().__init__(name)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("Environment variable ANTHROPIC_API_KEY not set")
        self.client = anthropic.Client(api_key)
        self.model = model

    @property
    def tokenizer(self):
        raise NotImplementedError(f"{self.name} does not expose the tokenizer")

    def generate(self, prompt: str, **kwargs) -> str:
        args = {
            "prompt": f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            "stop_sequences": [anthropic.HUMAN_PROMPT],
            "model": self.model,
            "max_tokens_to_sample": kwargs.get("max_tokens_to_sample", 20),
        } | kwargs
        resp = self.client.completion(**args)
        return resp["completion"]
