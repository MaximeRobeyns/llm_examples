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
import transformers

from wonderwords import RandomWord
from torch.optim import Optimizer
from hydra.utils import instantiate
from omegaconf.omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from llm_examples.utils import setup_loggers, setup_accelerator
from llm_examples.llm.huggingface import HuggingFaceLLM
from llm_examples.mcsb_ft.utils import get_new_words, clean
from llm_examples.mcsb_ft.prompts import description_prompt, question_prompt


def run(
    task_cfg: DictConfig,
    model: HuggingFaceLLM,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    tb_logger: TensorBoardLogger,
    csv_logger: CSVLogger,
    device: t.device,
):
    # Resume from checkpoint if one is available.
    # Optionally retain only the last n checkpoints

    label_ids = model.tokenizer(
        [f"{chr(ord('A') + i)}" for i in range(5)], return_tensors="pt"
    ).input_ids[:, -1:]

    r = RandomWord()

    # For logging
    correct = 0
    loss_totla = 0.0

    for it in range(task_cfg.num_iters):

        # 1. Select a batch of n random words
        word_lists = [
            get_new_words(model.tokenizer, r, task_cfg.num_labels)[0]
            for _ in range(task_cfg.micro_batch_size)
        ]

        # 2. Generate synthetic data
        # 2.1 Generate descriptions
        description_prompts = [description_prompt.format(wl[0]) for wl in word_lists]
        with model.disable_adapter(), t.no_grad():
            outputs = model.generate(
                description_prompts,
                max_new_tokens=20,
                temperature=0.5,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        outputs = outputs[0]  # only one generation sampled
        print(outputs)
        return
        gen_descriptions = [clean(outputs, model.tokenizer.eos_token)]
        gen_descriptions = [
            clean(
                s,
            )
        ]


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="mcsb_finetuning"
)
def do_mcsb_task(cfg: DictConfig):
    # Set up logging
    tb_logger, csv_logger = setup_loggers(cfg)

    # Set up LLM
    model: HuggingFaceLLM = instantiate(cfg.llm)

    # Setup optimiser
    opt_cfg = dict(cfg.opt)
    optclass = getattr(
        importlib.import_module(opt_cfg.pop("module")), opt_cfg.pop("classname")
    )
    opt = optclass(model.parameters(), **opt_cfg)

    # Setup learning rate scheduler
    total_train_steps = cfg.num_iters // cfg.gradient_accumulation_steps
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=total_train_steps,
    )

    # Prepare HF accelerator
    accelerator = setup_accelerator(
        cfg.micro_batch_size, cfg.seed, project_dir=cfg.paths.output_dir
    )
    model, opt, lr_scheduler = accelerator.prepare(model, opt, lr_scheduler)

    # Run the training
    run(cfg, model, opt, lr_scheduler, tb_logger, csv_logger, accelerator.device)

    # Do post-processing and save final model

    logging.info("successfully completed.")
    csv_logger.finalize("success")
    tb_logger.finalize("success")


if __name__ == "__main__":
    do_mcsb_task()
