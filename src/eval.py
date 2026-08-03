# original refer to: https://github.com/ashleve/lightning-hydra-template/blob/main/src/eval.py

# modify refer to: https://github.com/nathanpainchaud/lightning-hydra-template/blob/main/src/lightning_hydra_template/eval.py
# modify all type hints to higher python version (small case, built-in)
# add `predict` logic in evaluate function

# ⚠️ You may need to add `callback` logic in evaluate function if you refer to: https://github.com/nathanpainchaud/lightning-hydra-template/blob/main/src/lightning_hydra_template/eval.py

from typing import Any

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    # ⚠️ You may need to add `callback` logic in evaluate function if you refer to: https://github.com/nathanpainchaud/lightning-hydra-template/blob/main/src/lightning_hydra_template/eval.py
    # log.info("Instantiating callbacks...")
    # callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # ⚠️ remember to add `weights_only=False` for pytorch2.6+ version, but not needed for pytorch2.5 and below
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)

    metric_dict = trainer.callback_metrics

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    # ⚠️ we add `predict` logic here but we do not implement predict_dataloader() in src/data/mnist_datamodule.py
    # use test_dataloader() instead
    # and we do not store the prediction results here
    # just an example of how to use trainer.predict(...) for future reference
    if cfg.get("predict"):
        log.info("Starting predicting!")
        trainer.predict(
            model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=cfg.ckpt_path, weights_only=False
        )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
