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
"""LightningModule to handle training."""

from typing import Union, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

import new_semantic_parsing.optimization as opt
from new_semantic_parsing import metrics, data, config
from new_semantic_parsing.dataclasses import EncDecFreezingSchedule
from new_semantic_parsing.modeling_encoder_decoder_wpointer import EncoderDecoderWPointerModel
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.utils import make_subset


class PointerModule(nn.Module):
    """Handles training.

    Creates optimizers and implements training loop with multi-GPU and TPU support.

    Attributes:
        model: EncoderDecoderWPointerModel to train
        schema_tokenizer: TopSchemaTokenizer, mainly used for getting tokenizer options like BOS/EOS/PAD token ids
        text_tokenizer: transformers.PreTrainedTOkenizer, ^^
        train_dataset: PointerDataset
        valid_dataset: PointerDataset
        lr: learning rate, either float of dictionary with keys encoder_lr and decoder_lr
        batch_size: int, batch size used for training and evaluation
        warmup_steps: int
        weight_decay: float
        log_every: int, log to wandb each log_every steps
        monitor_classes: list of class names, monitor_classes[0] and all
            set of monitor_classes are logged specificly log_every steps
        max_tgt_len: maximum target length for evaluation
        freezing_schedule: EncDecFreezingSchedule, when to freeze and unfreeze encoder, decoder and head
    """

    def __init__(
        self,
        model: EncoderDecoderWPointerModel,
        schema_tokenizer: TopSchemaTokenizer,
        train_dataset: data.PointerDataset,
        valid_dataset: data.PointerDataset,
        lr: Union[float, Dict],
        batch_size=32,
        warmup_steps=0,
        weight_decay=0.0,
        test_dataset=None,
        log_every=50,
        monitor_classes=None,
        max_tgt_len=68,
        no_lr_scheduler=False,
        freezing_schedule: EncDecFreezingSchedule = None,
    ):
        super().__init__()
        self.model = model

        self.schema_tokenizer = schema_tokenizer
        self.text_tokenizer = schema_tokenizer.src_tokenizer

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.lr = lr

        if batch_size is None:
            raise ValueError("batch_size cannot be None")

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.monitor_classes = monitor_classes
        self.freezing_schedule = freezing_schedule
        self.max_tgt_len = max_tgt_len
        self.no_lr_scheduler = no_lr_scheduler
        self.global_step = None  # updated in the Trainer

        self._collator = data.Seq2SeqDataCollator(
            pad_id=self.text_tokenizer.pad_token_id,
            decoder_pad_id=self.schema_tokenizer.pad_token_id,
        )

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.model.get_input_embeddings().weight.dtype

    def forward(self, *args, **kwargs):
        """Coinsides with EncoderDecoderWPointerModel.forward"""
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        self.global_step = 0 if self.global_step is None else self.global_step + 1

        outputs = self(**batch)
        loss = outputs[0]

        if self.freezing_schedule is not None:
            self._freezer_step()

        # return initial move norm before the first update as a sanity check
        if self.log_every == 0 and self.model.initial_params is not None:
            move_norm = self.model.get_move_norm()
            if wandb.run is not None and move_norm > 1e-7:
                wandb.alert(f"nonzero move norm={move_norm} in the beginning of training")

            return {"loss": loss, "log": {"loss": loss, "move_norm": move_norm}}

        # if not logging, return loss
        if self.log_every == 0 or (self.global_step % self.log_every != 0):
            return {"loss": loss, "log": {"loss": loss}}

        # if logging, compute metrics
        logits = outputs[1]
        preds = logits.max(-1).indices

        labels = batch["labels"]
        label_masks = batch["decoder_attention_mask"]

        # accuracy, EM and first intent precision
        stop_token_ids = [self.schema_tokenizer.eos_token_id, self.schema_tokenizer.pad_token_id]
        batch_metrics = metrics.compute_metrics_from_batch(
            preds, labels, label_masks, stop_token_ids
        )
        batch_metrics = {f"train_batch_{k}": v for k, v in batch_metrics.items()}

        # tree path metrics
        pred_ids = preds.detach().unbind()
        target_ids = labels.detach().unbind()

        pred_tokens = [self.schema_tokenizer.decode(p, return_tokens=True) for p in pred_ids]
        true_tokens = [self.schema_tokenizer.decode(t, return_tokens=True) for t in target_ids]

        tree_metrics = metrics.get_tree_path_metrics(
            pred_tokens, true_tokens, self.monitor_classes, "train_batch"
        )

        if self.model.initial_params is not None:
            batch_metrics["move_norm"] = self.model.get_move_norm()

        log_dict = {"loss": loss, **batch_metrics, **tree_metrics}
        log_dict = {k: self._maybe_torchify(v) for k, v in log_dict.items()}

        return {"loss": loss, "aggregate_log": log_dict}

    def training_epoch_end(self, outputs):
        if isinstance(self.train_dataset, data.SampleConcatSubset):
            self.train_dataset.resample()

        # extract log_dict from outputs and ignore the outputs from no-log iterations
        aggregate_output = [x["aggregate_log"] for x in outputs if ("aggregate_log" in x)]
        if len(aggregate_output) == 0:
            return {}

        avg_log = self._average_logs(aggregate_output)

        # do not log average apoch loss to the same place as in-batch loss
        avg_log["epoch_loss"] = avg_log.pop("loss")
        return {"log": avg_log}

    def configure_optimizers(self):
        optimizer = opt.get_optimizers(
            model=self.model,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            use_synaptic_intelligence=True,  # activates AdamSI optimizer
        )
        if self.no_lr_scheduler:
            return optimizer

        scheduler = opt.get_noam_schedule(
            optimizer,
            self.warmup_steps,
            self.model.decoder.config.hidden_size,
        )

        # to call scheduler every step instead of every epoch
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collator.collate_batch,
        )
        return loader

    # --- Validation

    def validation_step(self, batch, batch_idx):
        return self._eval_step(
            batch,
            prefix="eval",
            monitor_classes=self.monitor_classes,
            compute_metrics_for_every_class=False,
        )

    def validation_epoch_end(self, outputs):
        avg_log = self._average_logs(outputs)
        return {"log": avg_log}

    def val_dataloader(self, subset_size=1.0):
        valid_dataset = self.valid_dataset
        if subset_size < 1:
            valid_dataset = make_subset(valid_dataset, subset_size)

        loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=self._collator.collate_batch,
            shuffle=False,
        )
        return loader

    # --- testing

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError(".test_dataloader invoked for the model without test_dataset")

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=self._collator.collate_batch,
        )
        return loader

    def test_step(self, batch, batch_idx):
        return self._eval_step(
            batch,
            prefix="test",
            monitor_classes=self.schema_tokenizer.vocab,
            compute_metrics_for_every_class=True,
        )

    def test_epoch_end(self, outputs):
        avg_log = self._average_logs(outputs)
        return {"test_metrics": avg_log}

    # --- Internal

    @staticmethod
    def _average_logs(logs):
        keys = logs[0].keys()
        avg_log = dict()

        for key in keys:
            avg_log[key] = sum(log[key] for log in logs) / len(logs)

        return avg_log

    def _maybe_torchify(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=self.dtype, device=self.device)

    def _freezer_step(self):
        step = self.global_step

        act = self.freezing_schedule.freeze_encoder
        if act is not None and step == act:
            self.model.freeze_encoder(freeze=True)

        act = self.freezing_schedule.unfreeze_encoder
        if act is not None and step == act:
            self.model.freeze_encoder(freeze=False)

        act = self.freezing_schedule.freeze_decoder
        if act is not None and step == act:
            self.model.freeze_decoder(freeze=True)

        act = self.freezing_schedule.unfreeze_decoder
        if act is not None and step == act:
            self.model.freeze_decoder(freeze=False)

        act = self.freezing_schedule.freeze_head
        if act is not None and step == act:
            self.model.freeze_head(freeze=True)

        act = self.freezing_schedule.unfreeze_head
        if act is not None and step == act:
            self.model.freeze_head(freeze=False)

    def _eval_step(self, batch, prefix, monitor_classes, compute_metrics_for_every_class=False):
        prediction_batch: torch.LongTensor = self.model.generate(
            input_ids=batch["input_ids"].to(self.device),
            pointer_mask=batch["pointer_mask"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            max_length=self.max_tgt_len,
            num_beams=1,
            pad_token_id=self.text_tokenizer.pad_token_id,
            bos_token_id=self.schema_tokenizer.bos_token_id,
            eos_token_id=self.schema_tokenizer.eos_token_id,
        )

        pred_ids = []
        target_ids = []

        for i, prediction in enumerate(prediction_batch):
            prediction = [
                p for p in prediction.cpu().numpy() if p not in self.schema_tokenizer.special_ids
            ]
            pred_ids.append(prediction)

            target = [
                p
                for p in batch["decoder_input_ids"][i].cpu().numpy()
                if p not in self.schema_tokenizer.special_ids
            ]
            target_ids.append(target)

        pred_tokens = [self.schema_tokenizer.decode(p, return_tokens=True) for p in pred_ids]
        true_tokens = [self.schema_tokenizer.decode(t, return_tokens=True) for t in target_ids]

        metrics_dict = metrics.get_metrics(
            pred_tokens,
            true_tokens,
            monitor_classes,
            prefix,
            self.schema_tokenizer,
            compute_metrics_for_every_class,
        )

        log_dict = {k: self._maybe_torchify(v) for k, v in metrics_dict.items()}
        return log_dict
