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
"""Thin wrapper around vLLM's API"""

import vllm

from ft_example.llm.base import LLM


class VLLM(LLM):
    def __init__(self, name: str, model_name_or_path: str, **kwargs):
        super().__init__(name)
        self.model = vllm.LLM(model_name_or_path, **kwargs)

    @property
    def tokenizer(self):
        return self.model.get_tokenizer()

    def generate(
        self, prompt: str | list[str], *args, **kwargs
    ) -> list[vllm.RequestOutput]:
        kwargs = {"use_tqdm": False} | kwargs
        return self.model.generate(prompt, *args, **kwargs)
