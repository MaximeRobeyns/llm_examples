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
"""Thin wrapper around OpenAI's API"""

import os
import openai

from llm_examples.llm.base import LLM


class OpenAILLM(LLM):
    def __init__(self, name: str, model: str):
        super().__init__(name)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Environment variable OPENAI_API_KEYnot set")
        openai.api_key = api_key
        self.model = model

    @property
    def tokenizer(self):
        raise NotImplementedError(f"{self.name} does not expose the tokenizer")

    def generate(self, prompt: str, **kwargs) -> list[str]:
        completion = openai.Completion.create(
            model=self.model,  # e.g. "text-davinci-003"
            prompt=prompt,
            **kwargs,
        )
        return completion["choices"]
