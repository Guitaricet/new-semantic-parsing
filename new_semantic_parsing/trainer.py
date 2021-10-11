# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
We had to switch from lightning to our training scripts, because it made the training setup too complicated.
The main issue was an unclear when optimizer/model parameters are getting loaded
from the checkpoint
(for example, optimizer state and model state are loaded on .test)
and what model version is used.


Trainer largely mimics the Lightning interfaces we use in our setup, but has a much simpler implementation.
"""
import logging
import os
import sys

import torch
import wandb


from new_semantic_parsing import data, PointerModule


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("wandb.sdk.internal.internal").setLevel(logging.WARNING)


class Trainer:
    """
    The class that handles training and evaluation loops, checkpointing, logging and more.

    Structurally similar to Lightning Trainer (specifically for the case of this project), but simpler.

    Args:
        max_epochs: int, max number of training epochs
        device: torch.device to train and evaluate on
        wandb_run: wandb run object used for logging
        gradient_clip_val: float, maximum gradient norm
        early_stopping_metric: metric used for early stopping and saving the best model.
            By default the model is saved every epoch and no early stopping is used.
        patience: int, early stoppig patience
        maximize_early_stopping_metric:
        min_epochs: int
        min_steps: int
        max_steps: int, max number of training iterations
        limit_val_batches: 0 <= float <= 1, amount of validation set to use when training.
            The final evaluation will use the full dataset.
        save_dir: directory to save the model, tokenizer and trainer checkpoints
    """

    def __init__(
        self,
        max_epochs,
        device,
        gradient_clip_val=1.0,
        early_stopping_metric=None,
        patience=None,
        maximize_early_stopping_metric=True,
        min_epochs=0,
        min_steps=0,
        max_steps=None,
        limit_val_batches=1.0,
        save_dir="checkpoint_dir",
    ):
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs or 0
        self.max_steps = max_steps or float("inf")
        self.min_steps = min_steps or 0

        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.limit_val_batches = limit_val_batches
        self.save_dir = save_dir

        self.early_stopping_metric = early_stopping_metric
        self.maximize_early_stopping_metric = maximize_early_stopping_metric
        self.patience = patience or self.max_epochs
        self._current_patience = 0
        self._best_metric = float("-inf") if maximize_early_stopping_metric else float("inf")

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._epoch = None
        self.train_dataloader = None
        self.valid_dataloader = None

    def fit(self, model: PointerModule, optimizer_and_scheduler=None, eval_before_training=False):
        logger.info(f"Loading the model to {self.device}")
        self.model: PointerModule = model.to(self.device)
        self.model.global_step = 0
        self._epoch = 0

        logger.info(f"Creating train and validation dataloaders")
        self.train_dataloader = self.model.train_dataloader()
        self.valid_dataloader = self.model.val_dataloader()

        if eval_before_training:
            self._validation_loop(self.valid_dataloader)

        if optimizer_and_scheduler is None:
            logger.info(f"Creating new optimizers")
            optimizer_and_scheduler = self.model.configure_optimizers()
        else:
            logger.info(f"Using provided optimizers")
        self.optimizer, self.scheduler = self._unpack_optimizer_and_scheduler(optimizer_and_scheduler)

        # Training loop
        logger.info(f"Starting training")

        should_stop = False
        for epoch in range(self.max_epochs):
            logger.info("-" * 10 + f"Starting epoch {epoch}" + "-" * 10)

            self._epoch = epoch
            if should_stop:
                logger.info("Training stopped via maximum number of steps reached")
                break

            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.model.global_step > self.max_steps:
                    should_stop = True
                    break
                wandb.log(
                    {"global_step": self.model.global_step, "epoch": self._epoch},
                    step=self.model.global_step,
                )

                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()

                training_step_dict = self.model.training_step(batch, batch_idx)
                loss = training_step_dict["loss"]

                if "log" in training_step_dict:
                    wandb.log(training_step_dict["log"], step=self.model.global_step)
                if "aggregate_log" in training_step_dict:
                    wandb.log(training_step_dict["aggregate_log"], step=self.model.global_step)

                loss.backward()
                self.model.on_after_backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

            # NOTE: unlike Lightning, we do not call .training_epoch_end() here

            # Validation loop
            should_stop_early = self._validation_loop(self.valid_dataloader)
            if should_stop_early:
                logger.info("Early stopping condition. Interrupting the training...")
                break

        # Resample if using the sample dataset
        if isinstance(self.model.train_dataset, data.SampleConcatSubset):
            self.model.train_dataset.resample()

        logger.info("Training has finished")

        if self.early_stopping_metric:
            logger.info(f"Loading the best model from {self.save_dir}")
            self.model.model = self.model.model.from_pretrained(self.save_dir)

    def save(self, checkpoint_dir=None):
        """Save the Trainer, model and tokenizer into checkpoint_dir

        If checkpoint_path is not specified, self.save_dir is used
        """
        if self.model is None:
            raise RuntimeError("There is no model so save. Call .fit(model) first")

        checkpoint_dir = checkpoint_dir or self.save_dir

        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None,
            "epoch": self._epoch,
            "global_step": self.model.global_step,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "trainer_checkpoint.ckpt"))

        self.model.model.save_pretrained(checkpoint_dir)
        self.model.schema_tokenizer.save(checkpoint_dir)

    @classmethod
    def load_optimizer_and_scheduler_states(cls, optimizer_and_scheduler, checkpoint_dir, only_scheduler=False):
        """Load the saved state into optimizer and lr scheduler

        Args:
            optimizer_and_scheduler: either a tuple ([optimizer], [scheduler]) or optimizer
            checkpoint_dir: path to the directory with a trainer_checkpoint.ckpt file
                that contains a pickled dictionary with the keys "optimizer_state_dict" and "scheduler_state_dict"

        Returns:
            same optimizer_and_scheduler, but with the restored states
        """
        optimizer, scheduler = cls._unpack_optimizer_and_scheduler(optimizer_and_scheduler)
        checkpoint = torch.load(os.path.join(checkpoint_dir, "trainer_checkpoint.ckpt"))

        if not only_scheduler:
            logger.info("Loading the optimizer state")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            logger.info("Loading the scheduler")
            if checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                logger.warning(
                    "Scheduler state in the checkpoint is empty. Continuing with the current state."
                )

        return optimizer, scheduler

    def _validation_loop(self, valid_dataloader):
        logger.info("Validating")

        eval_logs = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if batch_idx > len(valid_dataloader) * self.limit_val_batches:
                    logger.info("Validaiton is interrupted early because of limit_val_batches")
                    break

                eval_log = self.model.validation_step(batch, batch_idx)
                eval_logs.append(eval_log)

            aggregated_eval_logs = self.model.validation_epoch_end(eval_logs)
            # aggregated_eval_logs is a dict with a single key "log"
            self._maybe_save(aggregated_eval_logs["log"])
            wandb.log(aggregated_eval_logs["log"], step=self.model.global_step)

        self.model.train()
        return self._should_stop_early(aggregated_eval_logs)

    @staticmethod
    def _unpack_optimizer_and_scheduler(optimizer_and_scheduler):
        """
        Args:
            optimizer_and_scheduler: an optimizer object or
                a tuple of type
                ([optimizer], [{"scheduler": scheduler}])
                    or
                    (optimizer, scheduler)
                    or
                    ([optimizer], [scheduler])
                    or
                    or something in-between

        This secific form of the input follows lightning format.

        Returns:
            tuple (optimizer, maybe_scheduler) where maybe_scheduler may be None
        """
        if isinstance(optimizer_and_scheduler, tuple):
            optimizer_list, scheduler_list = optimizer_and_scheduler

            if isinstance(optimizer_list, list):
                if len(optimizer_list) != 1:
                    raise ValueError(
                        f"Only a single optimizer is supported, got {len(optimizer_list)} instead"
                    )
                optimizer = optimizer_list[0]

            elif isinstance(optimizer_list, torch.optim.Optimizer):
                optimizer = optimizer_list

            else:
                raise ValueError(optimizer_list)

            if isinstance(scheduler_list, list):
                if len(scheduler_list) != 1:
                    raise ValueError(
                        f"Only a single lr scheduler is supported, got {len(scheduler_list)} instead"
                    )
                scheduler = scheduler_list[0]

                if isinstance(scheduler, dict):
                    scheduler = scheduler["scheduler"]

            elif isinstance(scheduler_list, torch.optim.lr_scheduler._LRScheduler):
                scheduler = scheduler_list

            else:
                raise ValueError(scheduler_list)

            return optimizer, scheduler

        if isinstance(optimizer_and_scheduler, torch.optim.Optimizer):
            return optimizer_and_scheduler, None

    def _should_stop_early(self, validation_metrics):
        if (
            self.early_stopping_metric is None
            or self.min_epochs < self._epoch
            or self.min_steps < self.model.global_step
        ):
            return

        if self.early_stopping_metric not in validation_metrics:
            raise ValueError(
                f"Metric {self.early_stopping_metric} used for early stopping"
                f" is not found in validation metrics"
            )

        metric = validation_metrics[self.early_stopping_metric]

        if self._is_metric_better(metric):
            self._current_patience = 0
            self._best_metric = metric

        else:
            self._current_patience += 1

        if self._current_patience >= self.patience:
            return True

        return False

    def _maybe_save(self, validation_metrics):
        if self.early_stopping_metric is None:
            self.save()

        if self.early_stopping_metric not in validation_metrics:
            raise ValueError(
                f"Metric {self.early_stopping_metric} used for early stopping"
                f" is not found in validation metrics. Validation metrics: {validation_metrics.keys()}"
            )

        metric = validation_metrics[self.early_stopping_metric]

        if self._is_metric_better(metric):
            self.save()

    def _is_metric_better(self, metric):
        if self.maximize_early_stopping_metric:
            return metric >= self._best_metric
        else:
            return metric <= self._best_metric
