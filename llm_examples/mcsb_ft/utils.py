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

import torch as t

from datasets import load_dataset, Dataset
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
    return int((max_logit == answer_idxs.cpu()).sum().item())


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


def get_label_ids(tokenizer: PreTrainedTokenizer, n: int, numerical: bool) -> t.Tensor:
    """Returns the tokenized labels for a question with n choices"""
    return tokenizer(
        [f"{chr(ord('0') + i if numerical else ord('A') + i)}" for i in range(n)],
        return_tensors="pt",
    ).input_ids[:, -1:]


def prepare_wikitext_dset(
    tokenizer: PreTrainedTokenizer,
    *,
    block_size: int = 1024,
    blocks: int = 50,
    seed: int = 42,
    num_proc: int = 8,
    split: str = "validation",
    **_kwargs,
) -> Dataset:
    """
    Prepares a validation split of wikitext.
    """
    dset = load_dataset("wikitext", "wikitext-2-v1", split=split)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    def group(examples):
        # Concatenate everything
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the last bit to avoid having to use padding (some efficient
        # kernels don't support it)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dset = dset.map(
        preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text"]
    )
    tokenized_dset = tokenized_dset.map(group, batched=True, num_proc=num_proc)
    tokenized_dset = tokenized_dset.shuffle(seed=seed).select(range(blocks))
    return tokenized_dset
