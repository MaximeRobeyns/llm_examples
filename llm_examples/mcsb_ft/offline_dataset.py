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
"""Generate an offline dataset for the MCSB task."""

import os
import hydra
import shutil
import torch as t
import logging
import importlib
import transformers

from random import randint
from typing import Optional
from accelerate import Accelerator
from wonderwords import RandomWord
from torch.optim import Optimizer
from hydra.utils import instantiate
from transformers import PreTrainedTokenizer
from omegaconf.omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from llm_examples.utils import setup_loggers, setup_accelerator
from llm_examples.llm import HuggingFaceLLM, LLM
from llm_examples.mcsb_ft.utils import get_new_words, clean, get_num_correct
from llm_examples.mcsb_ft.prompts import description_prompt, question_preamble


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="mcsb_offline_finetuning",
)
def offline_dataset_mcsb(cfg: DictConfig):
    # Set up logging
    tb_logger, csv_logger = setup_loggers(cfg)

    if cfg.dataset.generate:
        logging.info(f"Re-Generating the dataset")


if __name__ == "__main__":
    offline_dataset_mcsb()
