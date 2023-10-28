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

from random import randint
from typing import Optional
from datasets import Dataset
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
from llm_examples.mcsb_ft.utils import (
    get_new_words,
    clean,
    get_num_correct,
    prepare_wikitext_dset,
)
from llm_examples.mcsb_ft.prompts import description_prompt, question_preamble


def get_label_ids(tokenizer: PreTrainedTokenizer, n: int) -> t.Tensor:
    """Returns the tokenized labels for a question with n choices"""
    return tokenizer(
        [f"{chr(ord('A') + i)}" for i in range(n)], return_tensors="pt"
    ).input_ids[:, -1:]


def generate_batch(
    generation_model: LLM,
    batch_size: int,
    min_labels: int,
    max_labels: Optional[int] = None,
) -> tuple[list[str], list[t.Tensor], t.Tensor]:
    """Syntheticallly generate a batch of data.

    Args:
        generation_model: The model to use for generation (can be different
            from the one we're trainiing; 3rd party API, etc.)
        batch_size: number of entries to genreate
        min_labels: (min) number of labels each batch question should have
        max_labels: (optional) if provided each batch element has a random
            number of labels between min and max-labels.

    TODO: add the option to use random characters / numbers of labels for an
        out-of-distribution validation task.

    Returns:
        Batch of: 1) questions, 2) tokenized answer labels, 3) correct answer indexes
    """
    r = RandomWord()
    tok = generation_model.tokenizer

    if max_labels is None:
        max_labels = min_labels

    # 1. Select a batch of n random words
    word_lists, label_ids = [], []
    for _ in range(batch_size):
        n = randint(min_labels, min_labels if max_labels is None else max_labels)
        word_lists.append(get_new_words(tok, r, n))
        label_ids.append(get_label_ids(tok, n))

    # 2. Generate synthetic data
    # 2.1 Generate descriptions
    description_prompts = [description_prompt.format(wl[0]) for wl in word_lists]
    with generation_model.disable_adapter(), t.no_grad(), t.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        outputs = generation_model.generate(
            description_prompts,
            max_new_tokens=20,
            temperature=0.5,
            pad_token_id=generation_model.tokenizer.eos_token_id,
        )
    outputs = [o[0] for o in outputs]  # only one generation sampled
    gen_descriptions = [clean(s, generation_model.tokenizer.eos_token) for s in outputs]
    gen_descriptions = [clean(s, "\n") for s in gen_descriptions]

    # 2.2 Generate question prompts
    question_prompts: list[str] = []
    answer_idxs = t.empty((batch_size,)).int()
    for i in range(len(word_lists)):
        n = len(word_lists[i])
        random_ord = t.randperm(n)
        answer_idxs[i] = (random_ord == 0).nonzero()
        question_prompt = question_preamble
        question_prompt += f"Description: {gen_descriptions[i]}\n"
        question_prompt += "\n".join(
            [
                f"{chr(ord('A') + j)}) {word_lists[i][k]}"
                for (j, k) in enumerate(random_ord)
            ]
        )
        question_prompt += f"\nAnswer (A to {chr(ord('A') + n-1)}):"
        question_prompts.append(question_prompt)

    return question_prompts, label_ids, answer_idxs


def evaluate_accuracy(
    model: LLM,
    device: t.device,
    batch_size: int = 32,
    min_labels: int = 3,
    max_labels: int = 8,
) -> float:
    """Evaluates the model by generating MCSB questions with random numbers of
    labels.
    """
    question_prompts, label_ids, answer_idxs = generate_batch(
        model, batch_size, min_labels, max_labels
    )

    question_inputs = model.tokenizer(
        question_prompts, return_tensors="pt", padding=True
    ).to(device)
    with t.inference_mode():
        logits = model.model(**question_inputs).logits

    # Calculate the log-likelihood loss
    correct = 0
    for i in range(batch_size):
        answer_logits = logits[i, -1][label_ids[i].squeeze()]
        correct += answer_logits.argmax() == answer_idxs[i]

    return correct / batch_size


def run(
    task_cfg: DictConfig,
    model: HuggingFaceLLM,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    accelerator: Accelerator,
    regression_dset: Dataset,
    tb_logger: TensorBoardLogger,
    csv_logger: CSVLogger,
):
    # Resume from checkpoint if one is available.
    check_dir = f"{task_cfg.paths.output_dir}/checkpoints"
    if os.path.exists(check_dir):
        ckpt = [c for c in os.listdir(check_dir) if c.startswith("ckpt")][-1]
        start_mb = int(ckpt.split("-")[-1])
        logging.info(f"Resuming from {ckpt}")
        accelerator.load_state(f"{check_dir}/{ckpt}")
    else:
        start_mb = 0

    device = accelerator.device
    batch_size = task_cfg.micro_batch_size * task_cfg.gradient_accumulation_steps
    num_micro_batches = int(task_cfg.training_examples / task_cfg.micro_batch_size)
    num_batches = int(task_cfg.training_examples / batch_size)

    # For logging
    correct = 0
    loss_total = 0.0

    for mb in range(start_mb, num_micro_batches):

        # 1. Select a batch of n random words & 2. Generate synthetic data
        question_prompts, label_ids, answer_idxs = generate_batch(
            model, task_cfg.micro_batch_size, task_cfg.num_labels
        )

        # 3. Answer the question
        question_inputs = model.tokenizer(
            question_prompts, return_tensors="pt", padding=True
        ).to(device)
        logits = model.model(**question_inputs).logits

        # Calculate the log-likelihood loss
        answer_ids = t.tensor([label_ids[0][a] for a in answer_idxs])
        answer_dist = t.distributions.Categorical(logits=logits[:, -1])
        LL = answer_dist.log_prob(answer_ids.to(logits.device))
        loss = -LL.mean()
        loss = loss / task_cfg.gradient_accumulation_steps

        accelerator.backward(loss)

        loss_total += loss.item()

        # Calculate the accuracy
        correct += get_num_correct(logits[:, -1], label_ids[0], answer_idxs)

        # Logging
        if (mb + 1) % task_cfg.gradient_accumulation_steps == 0:
            batch = mb // task_cfg.gradient_accumulation_steps
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

            avg_batch_loss = loss_total / task_cfg.gradient_accumulation_steps
            accuracy = correct / batch_size
            progress = batch / num_batches * 100
            logging.info(
                f"Step {batch} ({progress:.2f}%): loss {avg_batch_loss:.4f}, accuracy {accuracy:.4f}"
            )
            tb_logger.log_metrics(
                {
                    "loss": avg_batch_loss,
                    "accuracy": accuracy,
                    "lr": lr_scheduler.get_last_lr()[0],
                },
                step=batch,
            )
            tb_logger.experiment.add_text(
                "question_prompts", "\n\n".join(question_prompts), batch
            )

            correct = 0
            loss_total = 0.0

            # Save checkpoints
            if (batch + 1) % task_cfg.checkpoint.freq == 0:
                check_name = f"ckpt-{batch+1}"
                accelerator.save_state(f"{check_dir}/{check_name}")
                ckpts = [c for c in os.listdir(check_dir) if c.startswith("ckpt")]
                if task_cfg.checkpoint.keep is not None and accelerator.is_main_process:
                    if len(ckpts) > task_cfg.checkpoint.keep:
                        shutil.rmtree(f"{check_dir}/{ckpts[0]}")

            if (batch + 1) % task_cfg.eval.freq == 0:
                # Evaluate the accuracy on a 'validation' task
                eval_acc = evaluate_accuracy(
                    model,
                    device,
                    task_cfg.eval.batch_size,
                    task_cfg.eval.min_labels,
                    task_cfg.eval.max_labels,
                )
                logging.info("Step %d: validation accuracy %.4f", batch, eval_acc)
                tb_logger.log_metrics({"eval_accuracy": eval_acc}, step=batch)

                # Evaluate the perplexity of the model on some 'pretraining'
                # data

            if (batch + 1) % task_cfg.regression.freq == 0:
                rloss = 0.0
                for rb in regression_dset:
                    rbatch = {k: t.tensor(v).to(device) for k, v in rb.items()}
                    with t.inference_mode(), t.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=False, enable_mem_efficient=False
                    ):
                        rloss += model.model(**rbatch).loss.item()
                rppl = rloss / len(regression_dset)
                logging.info("Step %d: regression perplexity %.4f", batch, rppl)
                tb_logger.log_metrics({"regression_ppl": rppl}, step=batch)


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
    total_steps = cfg.training_examples / (
        cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare HF accelerator
    accelerator = setup_accelerator(
        cfg.micro_batch_size, cfg.seed, project_dir=cfg.paths.output_dir
    )

    model.model, opt, lr_scheduler = accelerator.prepare(model.model, opt, lr_scheduler)

    # Prepare dataset for regression evaluations
    regress_dset = prepare_wikitext_dset(model.tokenizer, **cfg.regression)

    print(regress_dset)

    # Run the training
    run(cfg, model, opt, lr_scheduler, accelerator, regress_dset, tb_logger, csv_logger)

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
