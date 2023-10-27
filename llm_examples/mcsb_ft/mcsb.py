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

import os
import hydra
import shutil
import torch as t
import logging
import importlib
import transformers

from accelerate import Accelerator
from wonderwords import RandomWord
from torch.optim import Optimizer
from hydra.utils import instantiate
from omegaconf.omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from llm_examples.utils import setup_loggers, setup_accelerator
from llm_examples.llm.huggingface import HuggingFaceLLM
from llm_examples.mcsb_ft.utils import get_new_words, clean, get_num_correct
from llm_examples.mcsb_ft.prompts import description_prompt, question_preamble


def run(
    task_cfg: DictConfig,
    model: HuggingFaceLLM,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    accelerator: Accelerator,
    tb_logger: TensorBoardLogger,
    csv_logger: CSVLogger,
):
    # Resume from checkpoint if one is available.
    check_dir = f"{task_cfg.paths.output_dir}/checkpoints"
    if os.path.exists(check_dir):
        ckpt = [c for c in os.listdir(check_dir) if c.startswith("ckpt")][-1]
        start_it = int(ckpt.split("-")[-1])
        logging.info(f"Resuming from {ckpt}")
        accelerator.load_state(f"{check_dir}/{ckpt}")
    else:
        start_it = 0

    r = RandomWord()
    device = accelerator.device
    batch_size = task_cfg.micro_batch_size * task_cfg.gradient_accumulation_steps

    label_ids = model.tokenizer(
        [f"{chr(ord('A') + i)}" for i in range(5)], return_tensors="pt"
    ).input_ids[:, -1:]

    # For logging
    correct = 0
    loss_total = 0.0

    for it in range(start_it, task_cfg.num_iters):

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
        outputs = [o[0] for o in outputs]  # only one generation sampled
        gen_descriptions = [clean(s, model.tokenizer.eos_token) for s in outputs]
        gen_descriptions = [clean(s, "\n") for s in gen_descriptions]

        # 2.2 Generate question prompts
        question_prompts: list[str] = []
        answer_idxs = t.empty((task_cfg.micro_batch_size,)).int()
        for i in range(len(word_lists)):
            random_ord = t.randperm(task_cfg.num_labels)
            answer_idxs[i] = (random_ord == 0).nonzero()
            question_prompt = question_preamble
            question_prompt += f"Description: {gen_descriptions[i]}\n"
            question_prompt += "\n".join(
                [
                    f"{chr(ord('A') + j)}) {word_lists[i][k]}"
                    for (j, k) in enumerate(random_ord)
                ]
            )
            question_prompt += "\nAnswer (A to E):"
            question_prompts.append(question_prompt)

        # 3. Answer the question
        question_inputs = model.tokenizer(
            question_prompts, return_tensors="pt", padding=True
        ).to(device)
        logits = model.model(**question_inputs).logits

        # Calculate the log-likelihood loss
        answer_ids = t.tensor([label_ids[a] for a in answer_idxs])
        answer_dist = t.distributions.Categorical(logits=logits[:, -1])
        LL = answer_dist.log_prob(answer_ids.to(logits.device))
        loss = -LL.mean()
        loss = loss / task_cfg.gradient_accumulation_steps

        accelerator.backward(loss)

        loss_total += loss.item()

        # Calculate the accuracy
        correct += get_num_correct(logits[:, -1], label_ids, answer_idxs)

        real_iter = it // task_cfg.gradient_accumulation_steps

        # Logging
        if (it + 1) % task_cfg.gradient_accumulation_steps == 0:
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

            avg_batch_loss = loss_total / task_cfg.gradient_accumulation_steps
            accuracy = correct / batch_size
            logging.info(
                "Step %d: loss %.4f, accuracy %.4f", real_iter, avg_batch_loss, accuracy
            )
            tb_logger.log_metrics(
                {
                    "loss": avg_batch_loss,
                    "accuracy": accuracy,
                    "lr": lr_scheduler.get_last_lr()[0],
                },
                step=real_iter,
            )
            tb_logger.experiment.add_text(
                "question_prompts", "\n\n".join(question_prompts), real_iter
            )

            correct = 0
            loss_total = 0.0

        # Save checkpoints
        if (real_iter + 1) % task_cfg.checkpoint.freq == 0:
            check_name = f"ckpt-{real_iter+1}"
            accelerator.save_state(f"{check_dir}/{check_name}")
            ckpts = [c for c in os.listdir(check_dir) if c.startswith("ckpt")]
            if task_cfg.checkpoint.keep is not None and accelerator.is_main_process:
                if len(ckpts) > task_cfg.checkpoint.keep:
                    shutil.rmtree(f"{check_dir}/{ckpts[0]}")


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="mcsb_finetuning"
)
def do_mcsb_task(cfg: DictConfig):
    # Set up logging
    tb_logger, csv_logger = setup_loggers(cfg)

    # Set up LLM to train
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

    model.model, opt, lr_scheduler = accelerator.prepare(model.model, opt, lr_scheduler)

    # Run the training
    run(cfg, model, opt, lr_scheduler, accelerator, tb_logger, csv_logger)

    # Perform any post-processing and save final model
    logging.info("Doing post-processing")

    unwrapped_model = accelerator.unwrap_model(model.model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(f"{cfg.paths.output_dir}/{cfg.task_name}_model")

    logging.info("successfully completed.")
    csv_logger.finalize("success")
    tb_logger.finalize("success")


if __name__ == "__main__":
    do_mcsb_task()
