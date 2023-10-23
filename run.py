import os
import hydra
import logging

from omegaconf import OmegaConf
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig):

    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    # Save the configuration values in a file in the outout directory for later
    # reference
    with open(os.path.join(cfg.paths.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper(), "INFO"))

    # Setup TB logger
    op = cfg.paths.output_dir.split("/")
    tb_logger = TensorBoardLogger("/".join(op[:-2]), op[-2], op[-1])
    csv_logger = CSVLogger(
        "/".join(op[:-2]), op[-2], op[-1], flush_logs_every_n_steps=1
    )

    model = instantiate(cfg.llm)

    # start the task
    task = instantiate(cfg.task)
    task.run(cfg, model, tb_logger, csv_logger)

    logging.info("successfully completed.")
    csv_logger.finalize("success")
    tb_logger.finalize("success")


if __name__ == "__main__":
    main()
