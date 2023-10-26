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
"""Multiple-choice symbol binding task"""

import hydra
import torch as t
import logging
import importlib

from hydra.utils import instantiate
from omegaconf.omegaconf import DictConfig
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger


from llm_examples.utils import setup_run
from llm_examples.llm.huggingface import HuggingFaceLLM


def run(
    task_cfg: DictConfig,
    model: HuggingFaceLLM,
    tb_logger: TensorBoardLogger,
    csv_logger: CSVLogger,
):
    print("doing MCSB task")


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="mcsb_finetuning"
)
def do_mcsb_task(cfg: DictConfig):
    tb_logger, csv_logger = setup_run(cfg)

    model = instantiate(cfg.llm)

    opt_cfg = dict(cfg.opt)
    optclass = getattr(
        importlib.import_module(opt_cfg.pop("module")), opt_cfg.pop("classname")
    )
    opt = optclass(model.parameters(), **opt_cfg)

    # TODO: setup accelerate

    run(cfg, model, tb_logger, csv_logger)

    logging.info("successfully completed.")
    csv_logger.finalize("success")
    tb_logger.finalize("success")
    pass


if __name__ == "__main__":
    do_mcsb_task()
