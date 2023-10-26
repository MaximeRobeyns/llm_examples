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
"""Some utility functions"""

import os
import torch as t
import torch.nn as nn
import numpy as np
import shutil
import logging
import importlib

from transformers import PreTrainedModel
from omegaconf.omegaconf import OmegaConf, DictConfig
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger


def is_peft_available():
    return importlib.util.find_spec("peft") is not None


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(t.float32)


def get_fst_device(model: nn.Module) -> t.device:
    """Given the model, return the device of the first layer."""
    return next(model.parameters()).device


def setup_run(cfg: DictConfig):

    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper(), "INFO"))

    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    if cfg.paths.output_dir.split("/")[-1] == "dev_run":
        logging.info("Cleaning development log directory")
        clean_dir(cfg.paths.output_dir)

    # Save the configuration values in a file in the outout directory for later
    # reference
    with open(os.path.join(cfg.paths.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Setup TB logger
    op = cfg.paths.output_dir.split("/")
    tb_logger = TensorBoardLogger("/".join(op[:-2]), op[-2], op[-1])
    csv_logger = CSVLogger(
        "/".join(op[:-2]), op[-2], op[-1], flush_logs_every_n_steps=1
    )
    return tb_logger, csv_logger


def prepare_model_for_quantized_training(
    model: PreTrainedModel, use_gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """
    Modification of peft.prepare_model_for_int8_training to also support int4
    quantization from bitsandbytes.

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == t.float16) or (param.dtype == t.bfloat16):
            param.data = param.data.to(t.float32)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    if is_quantized and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                print(f"making {output} require grad")
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    # model.lm_head = CastOutputToFloat(model.lm_head)
    return model


def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Total params: {total_params}, trainable: {trainable_params} ({100 * trainable_params / total_params:.02f}%)"
    )


def clean_dir(dir_path: str) -> None:
    """Empties a directory by deleting the directory and creating a new empty
    directory in its place.

    Args:
        dir_path: path to directory to clean.
    """
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def str_to_torch_dtype(name: str) -> t.dtype:
    dt = t.__dict__[name]
    assert isinstance(dt, t.dtype)
    return dt


def np_to_torch_dtype(np_type: str | np.dtype) -> t.dtype:
    match np_type:
        case "bool" | np.bool_:
            return t.bool
        case "uint8" | np.uint8:
            return t.uint8
        case "int8" | np.int8:
            return t.int8
        case "int16" | np.int16:
            return t.int16
        case "int32" | np.int32:
            return t.int32
        case "int64" | np.int64:
            return t.int64
        case "float16" | np.float16:
            return t.float16
        case "float32" | np.float32:
            return t.float32
        case "float64" | np.float64:
            return t.float64
        case "complex64" | np.complex64:
            return t.complex64
        case "complex128" | np.complex128:
            return t.complex128
        case _:
            raise ValueError(f"Unrecognized type, {np_type}")
