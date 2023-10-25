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
"""Base LLM class"""

from abc import abstractmethod
from typing import Any


class LLM:
    """
    Abstract base implementation of an LLM that all other implementations
    should extend.

    This is a convenience method that is optional to use but povides a
    consistent API across the different model types, while remaining
    transparent and passing through any methods and fields from the wrapped
    classes.

    Wrappers should place the initialised models in the `self.model` field.
    """

    def __init__(self, name: str) -> None:
        """We include a `name` to identify the LLM method."""
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __getattr__(self, name: str) -> Any:
        """
        Redirect attribute access to the underlying model if attribute is not
        found in this wrapper class.
        """
        attr = getattr(self.model, name, None)
        if attr is not None:
            if callable(attr):
                return attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @abstractmethod
    def generate(self, prompt: str | list[str], *args, **kwargs) -> list[Any]:
        raise NotImplementedError()
