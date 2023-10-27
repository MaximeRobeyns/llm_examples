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
"""Utilities for the MCSB task"""

from torchtyping import TensorType as Tensor
from wonderwords import RandomWord
from transformers import PreTrainedTokenizer


def get_num_correct(
    logits: Tensor["batch", "vocab"],
    label_ids: Tensor["labels", 1],
    answer_idxs: Tensor["batch"],
) -> int:
    gather_idxs = label_ids.T.expand(logits.size(0), -1).to(logits.device)
    answer_logits = logits.gather(1, gather_idxs)
    max_logit = answer_logits.argmax(-1).cpu()
    assert max_logit.shape == answer_idxs.shape
    return int((max_logit == answer_idxs).sum().item())


def clean(seq: str, sep: str) -> str:
    """Returns the substring before the separator, if it exists"""
    return seq.split(sep)[0].strip() if sep in seq else seq


def get_new_words(tokenizer: PreTrainedTokenizer, r: RandomWord, n: int) -> list[str]:
    """
    Will return `n` random nouns which encode to a single token under the
    provided tokenizer.

    Args:
        tokenizer: The HF tokenizer to use.
        r: The RandomWord instance to use.
        n: The number of words to return.
    """
    words: list[str] = []
    for _ in range(n):
        while True:
            word = r.word(include_parts_of_speech=["nouns"])
            ids = tokenizer(word, add_special_tokens=False).input_ids
            if len(ids) == 1:
                words.append(word)
                break
    return words
