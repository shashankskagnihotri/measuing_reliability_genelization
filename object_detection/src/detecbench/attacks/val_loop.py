import logging
from collections.abc import Callable, Sequence
from copy import deepcopy
from time import sleep
from typing import Dict, List, Union

import torch
import wandb
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner import Runner
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader

from .common import Attack


def replace_val_loop(attack: Attack, use_wandb: bool):
    """"""

    LOOPS.module_dict.pop("ValLoop")

    @LOOPS.register_module()
    class ValLoop(BaseLoop):
        def __init__(
            self,
            runner,
            dataloader: DataLoader | dict,
            evaluator: Evaluator | dict | list,
            fp16: bool = False,
        ) -> None:
            super().__init__(runner, dataloader)

            if isinstance(evaluator, (dict, list)):
                self.evaluator = runner.build_evaluator(evaluator)
            else:
                assert isinstance(evaluator, Evaluator), (
                    "evaluator must be one of dict, list or Evaluator instance, "
                    f"but got {type(evaluator)}."
                )
                self.evaluator = evaluator
            if hasattr(self.dataloader.dataset, "metainfo"):
                self.evaluator.dataset_meta = getattr(self.dataloader.dataset, "metainfo")
                self.runner.visualizer.dataset_meta = getattr(self.dataloader.dataset, "metainfo")
            else:
                print_log(
                    f"Dataset {self.dataloader.dataset.__class__.__name__} has no "
                    "metainfo. ``dataset_meta`` in evaluator, metric and "
                    "visualizer will be None.",
                    logger="current",
                    level=logging.WARNING,
                )
            self.fp16 = fp16

        def run(self) -> list[dict]:
            self.runner.call_hook("before_val")
            self.runner.call_hook("before_val_epoch")
            self.runner.model.eval()

            steps = getattr(attack, "steps", 1)
            # +1 since we want to evaluate the original data as well
            evaluators = [deepcopy(self.evaluator) for _ in range(steps + 1)]

            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, evaluators)

            metrics = [
                evaluator.evaluate(len(self.dataloader.dataset))  # type: ignore
                for evaluator in evaluators
            ]

            if use_wandb:
                # Log history of metrics to wandb
                wandb.define_metric("step")
                wandb.define_metric("*", step_metric="step")

                for step, metric in enumerate(metrics):
                    wandb.log({**metric, "step": step})
                    sleep(1)

            self.runner.call_hook(
                "after_val_epoch", metrics=metrics[-1]
            )  # metrics now a list of dicts, standard hooks expect dict
            self.runner.call_hook("after_val")
            return metrics

        def run_iter(
            self,
            idx,
            data_batch: dict,
            evaluators: list[Evaluator],
        ):
            self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)

            data_batch_prepro = attack.run_batch(
                data_batch=data_batch, model=self.runner.model, evaluators=evaluators
            )

            self.runner.model.bbox_head.adv_attack = (
                False  # needed to avoid issue with reppoint heads
            )
            with torch.no_grad():
                outputs = self.runner.model(**data_batch_prepro, mode="predict")

            self.runner.call_hook(
                "after_val_iter",
                batch_idx=idx,
                data_batch=data_batch_prepro,
                outputs=outputs,
            )
