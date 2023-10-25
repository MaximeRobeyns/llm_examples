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
"""Base class for tasks in this project"""

from abc import abstractmethod
from omegaconf.omegaconf import DictConfig
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from llm_examples.llm.base import LLM


class Task:
    def __init__(self, cfg: DictConfig):
        """
        Setup the task, load required data, initialise any constants...
        Args:
            cfg: the task configuration
        """
        self.cfg = cfg

    @abstractmethod
    def run(
        self,
        cfg: DictConfig,
        model: LLM,
        tb_logger: TensorBoardLogger,
        csv_logger: CSVLogger,
    ):
        """Run the task.

        Args:
            cfg: the entire run configuration (not just task config)
            model: the loaded LLM
            tb_logger: tensorboard logger
            csv_logger: csv logger
        """
        raise NotImplementedError()
