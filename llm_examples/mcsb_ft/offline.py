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

from tqdm import tqdm
from vllm import SamplingParams
from random import randint
from typing import Optional
from datasets import Dataset, DatasetDict, load_dataset
from contextlib import nullcontext, contextmanager
from accelerate import Accelerator
from wonderwords import RandomWord
from torch.optim import Optimizer
from hydra.utils import instantiate
from transformers import PreTrainedTokenizer
from datasets.splits import Split
from torch.utils.data import DataLoader
from omegaconf.omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from llm_examples.utils import setup_loggers, setup_accelerator
from llm_examples.llm import HuggingFaceLLM, VLLM
from llm_examples.mcsb_ft.utils import (
    get_new_words,
    get_num_correct,
    get_label_ids,
    prepare_wikitext_dset,
)
from llm_examples.mcsb_ft.prompts import description_prompt, question_preamble


def generate_batch(
    generation_model: VLLM,
    batch_size: int,
    min_labels: int,
    max_labels: Optional[int] = None,
    numerical: bool = False,
) -> tuple[list[str], list[t.Tensor], t.Tensor]:
    """A vLLM version of generate_batch for generating an offline dataset of data.

    Args:
        generation_model: The model to use for generation (can be different
            from the one we're trainiing; 3rd party API, etc.)
        batch_size: number of entries to genreate
        min_labels: (min) number of labels each batch question should have
        max_labels: (optional) if provided each batch element has a random
            number of labels between min and max-labels.
        numerical: whether to use numerical (as opposed to alphabetical)
            labels. Can be used to generate OOD data as a validation split.

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
        label_ids.append(get_label_ids(tok, n, numerical))

    sampling_params = SamplingParams(max_tokens=24, stop="\n")

    # 2. Generate synthetic data
    # 2.1 Generate descriptions
    #
    description_prompts = [description_prompt.format(wl[0]) for wl in word_lists]
    outputs = generation_model.generate(
        description_prompts, sampling_params=sampling_params
    )
    gen_descriptions = [o.outputs[0].text.strip() for o in outputs]

    # 2.2 Generate question prompts
    question_prompts: list[str] = []
    answer_idxs = t.empty((batch_size,)).int()
    base = ord("0") if numerical else ord("A")
    for i in range(len(word_lists)):
        n = len(word_lists[i])
        random_ord = t.randperm(n)
        answer_idxs[i] = (random_ord == 0).nonzero()
        question_prompt = question_preamble
        question_prompt += f"Description: {gen_descriptions[i]}\n"
        question_prompt += "\n".join(
            [f"{chr(base + j)}) {word_lists[i][k]}" for (j, k) in enumerate(random_ord)]
        )
        question_prompt += f"\nAnswer ({chr(base)} to {chr(base + n-1)}):"
        question_prompts.append(question_prompt)

    return question_prompts, label_ids, answer_idxs


def generate_dataset(cfg: DictConfig):
    generation_model = instantiate(cfg.generation_llm)

    # 1. Generate train split
    dset_questions, dset_label_ids, dset_answer_labels = [], [], []
    for _ in tqdm(range(cfg.dataset.num_examples // cfg.dataset.gen_batch_size)):
        questions, label_ids, answer_labels = generate_batch(
            generation_model, cfg.dataset.gen_batch_size, cfg.dataset.num_labels
        )
        dset_questions.extend(questions)
        dset_label_ids.extend([l.squeeze().tolist() for l in label_ids])
        dset_answer_labels.extend(answer_labels.tolist())
    train_dset = Dataset.from_dict(
        {
            "questions": dset_questions,
            "labels": dset_answer_labels,
            "answer_tokens": dset_label_ids,
        },
        split=Split.TRAIN,
    )

    # 2. Generate validation split
    dset_questions, dset_label_ids, dset_answer_labels = [], [], []
    for _ in tqdm(range(cfg.dataset.validation_examples // cfg.dataset.gen_batch_size)):
        questions, label_ids, answer_labels = generate_batch(
            generation_model,
            cfg.dataset.gen_batch_size,
            cfg.dataset.val_num_labels,
            numerical=True,
        )
        dset_questions.extend(questions)
        dset_label_ids.extend([l.squeeze().tolist() for l in label_ids])
        dset_answer_labels.extend(answer_labels.tolist())
    val_dset = Dataset.from_dict(
        {
            "questions": dset_questions,
            "labels": dset_answer_labels,
            "answer_tokens": dset_label_ids,
        },
        split=Split.VALIDATION,
    )

    dataset = DatasetDict(
        {str(Split.TRAIN): train_dset, str(Split.VALIDATION): val_dset}
    )

    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    dataset.save_to_disk(f"{cfg.paths.data_dir}/{cfg.dataset.name}")

    del generation_model
    t.cuda.empty_cache()

    return dataset


def evaluate_accuracy_offline(
    model: HuggingFaceLLM,
    device: t.device,
    val_loader: DataLoader,
    iters: int = 10,
) -> float:
    """Evaluates the model on an OOD MCSB dataset"""
    acc = 0.0
    for i, batch in enumerate(val_loader):
        question_prompts, label_ids, answer_idxs = batch
        batch_size = answer_idxs.size(0)

        question_inputs = model.tokenizer(
            question_prompts, return_tensors="pt", padding=True
        ).to(device)
        with t.inference_mode():
            logits = model.model(**question_inputs).logits

        # Calculate the log-likelihood loss
        correct = 0
        for j in range(batch_size):
            answer_logits = logits[j, -1][label_ids[j].squeeze()]
            correct += answer_logits.argmax() == answer_idxs[j]
        acc += correct / batch_size

        if i == iters:
            break

    return acc / iters


def run_offline(
    task_cfg: DictConfig,
    model: HuggingFaceLLM,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    accelerator: Accelerator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    regression_dset: Dataset,
    tb_logger: TensorBoardLogger,
    csv_logger: CSVLogger,
):
    # Resume from checkpoint if one is available.
    check_dir = f"{task_cfg.paths.output_dir}/checkpoints"
    grad_acc_steps = task_cfg.gradient_accumulation_steps
    start_batch = 0
    start_mb = 0
    if os.path.exists(check_dir):
        ckpt = [c for c in os.listdir(check_dir) if c.startswith("ckpt")][-1]
        start_batch = int(ckpt.split("-")[-1])
        start_mb = grad_acc_steps * start_batch
        logging.info(f"Resuming from {ckpt}")
        accelerator.load_state(f"{check_dir}/{ckpt}")
        train_loader = accelerator.skip_first_batches(train_loader, start_mb)

    # For logging
    correct = 0
    loss_total = 0.0

    for mb, microbatch in enumerate(train_loader):
        question_prompts, label_ids, answer_idxs = microbatch

        question_inputs = model.tokenizer(
            question_prompts, return_tensors="pt", padding=True
        ).to(accelerator.device)

        with accelerator.no_sync(model) if mb % grad_acc_steps != 0 else nullcontext():
            # Accumulate gradients
            logits = model.model(**question_inputs).logits
            # Calculate the log-likelihood loss
            answer_ids = t.tensor([label_ids[0][a] for a in answer_idxs])
            answer_dist = t.distributions.Categorical(logits=logits[:, -1])
            LL = answer_dist.log_prob(answer_ids.to(logits.device))
            loss = -LL.mean()
            loss = loss / grad_acc_steps
            accelerator.backward(loss)

        loss_total += loss.item()

        # Calculate the accuracy
        correct += get_num_correct(logits[:, -1], label_ids[0], answer_idxs.squeeze())

        # Logging
        if (mb + 1) % grad_acc_steps == 0:
            batch = start_batch + (mb // grad_acc_steps)
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

            avg_batch_loss = loss_total / grad_acc_steps
            accuracy = correct / task_cfg.dataset.batch_size
            progress = (start_mb + mb) / (len(train_loader) + start_mb) * 100
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
                eval_acc = evaluate_accuracy_offline(
                    model,
                    accelerator.device,
                    val_loader,
                    task_cfg.eval.iters,
                )
                logging.info("Step %d: validation accuracy %.4f", batch, eval_acc)
                tb_logger.log_metrics({"eval_accuracy": eval_acc}, step=batch)

                # Evaluate the perplexity of the model on some 'pretraining'
                # data

            if (batch + 1) % task_cfg.regression.freq == 0:
                rloss = 0.0
                for rb in regression_dset:
                    rbatch = {
                        k: t.tensor(v).to(accelerator.device) for k, v in rb.items()
                    }
                    with t.inference_mode(), t.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=False, enable_mem_efficient=False
                    ):
                        rloss += model.model(**rbatch).loss.item()
                rppl = rloss / len(regression_dset)
                logging.info("Step %d: regression perplexity %.4f", batch, rppl)
                tb_logger.log_metrics({"regression_ppl": rppl}, step=batch)


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(examples):
        qs, ans, ls = [], [], []
        for e in examples:
            qs.append(e["questions"])
            ls.append(e["labels"])
            ans.append(e["answer_tokens"].reshape(-1, 1))
        # qs = tokenizer(qs, return_tensors="pt", padding=True)
        return qs, t.hstack(ans).T, t.tensor(ls).reshape(-1, 1)

    return collate_fn


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="mcsb_offline_finetuning",
)
def offline_dataset_mcsb(cfg: DictConfig):
    # Set up logging
    tb_logger, csv_logger = setup_loggers(cfg)

    if cfg.dataset.generate:
        logging.info("Generating offline dataset")
        dataset = generate_dataset(cfg)
    else:
        logging.info("Attempting to offline dataset")
        dset_path = f"{cfg.paths.data_dir}/{cfg.dataset.name}"
        dataset = DatasetDict.load_from_disk(dset_path)
        logging.info("Loading successful")

    # Set up LLM to train
    model: HuggingFaceLLM = instantiate(cfg.llm)

    dataset.set_format(type="torch")
    assert cfg.dataset.batch_size % cfg.gradient_accumulation_steps == 0
    micro_batch_size = cfg.dataset.batch_size // cfg.gradient_accumulation_steps
    train_loader = DataLoader(
        dataset["train"],
        batch_size=micro_batch_size,
        collate_fn=get_collate_fn(model.tokenizer),
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=cfg.dataset.val_batch_size,
        collate_fn=get_collate_fn(model.tokenizer),
    )

    # Setup optimiser
    opt_cfg = dict(cfg.opt)
    optclass = getattr(
        importlib.import_module(opt_cfg.pop("module")), opt_cfg.pop("classname")
    )
    opt = optclass(model.model.parameters(), **opt_cfg)

    # Setup learning rate scheduler
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=len(train_loader) // cfg.gradient_accumulation_steps,
    )

    # Prepare HF accelerator
    assert cfg.dataset.batch_size % cfg.gradient_accumulation_steps == 0
    accelerator = setup_accelerator(
        micro_batch_size,
        cfg.seed,
        project_dir=cfg.paths.output_dir,
        # gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    # Prepare dataset for regression evaluations
    regress_dset = prepare_wikitext_dset(model.tokenizer, **cfg.regression)

    model.model, opt, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model.model, opt, lr_scheduler, train_loader, val_loader
    )

    run_offline(
        cfg,
        model,
        opt,
        lr_scheduler,
        accelerator,
        train_loader,
        val_loader,
        regress_dset,
        tb_logger,
        csv_logger,
    )

    # Perform any post-processing and save final model
    logging.info("Doing post-processing")

    unwrapped_model = accelerator.unwrap_model(model.model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(f"{cfg.paths.output_dir}/{cfg.task_name}_model")

    logging.info("successfully completed.")
    csv_logger.finalize("success")
    tb_logger.finalize("success")


if __name__ == "__main__":
    offline_dataset_mcsb()
