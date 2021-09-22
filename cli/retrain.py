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
"""Finetune a trained model on a dataset.

Similar to train.py, but loads the model and trainer from checkpoint
and uses finetune_set instead of train_set from the data.pkl
"""

import os
import sys
import shutil
import pprint
import logging
import argparse
import tempfile
from os.path import join as path_join

import toml
import torch
import wandb

import new_semantic_parsing as nsp
import new_semantic_parsing.dataclasses

from new_semantic_parsing import utils, cli_utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("wandb.sdk.internal.internal").setLevel(logging.WARNING)


def parse_args(args=None):
    """Parses cli arguments.

    This function is shared between retrain.py and retrain_simple.py
    """
    parser = argparse.ArgumentParser()

    # fmt: off

    # data
    parser.add_argument("--data-dir", required=True,
                        help="Path to preprocess.py --save-dir containing tokenizer, "
                             "data.pkl, and args.toml")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store checkpoints and other output files")
    parser.add_argument("--eval-data-amount", default=1., type=float,
                        help="amount of validation set to use when training. "
                             "The final evaluation will use the full dataset.")
    parser.add_argument("--new-classes", default=None,
                        help="names of classes to track")

    parser.add_argument("--new-data-amount", default=1., type=float,
                        help="Amount of new data (finetune_set) to train on, 0 < amount <= 1")
    parser.add_argument("--old-data-amount", default=0., type=float,
                        help="Amount of old data (train_set) to train on, only values from {0, 1} are supported")
    parser.add_argument("--old-data-sampling-method", default="merge_subset",
                        help="how to sample from old data")
    parser.add_argument("--average-checkpoints", default=False, action="store_true")
    parser.add_argument("--new-model-weight", default=0.5, type=float)

    # model
    parser.add_argument("--model-dir", required=True,
                        help="Model directory containing 1) checkpoint loadable via "
                             "EncoderDecoderWPointerModel.from_pretrained and "
                             "2) tokenizer directory")

    # training
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--min-epochs", default=1, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--min-steps", default=None, type=int)
    parser.add_argument("--early-stopping", default=None, type=int,
                        help="Early stopping patience. No early stopping by default.")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="By default, checkpoint lr is used.")
    parser.add_argument("--encoder-lr", default=None, type=float,
                        help="Encoder learning rate, overrides --lr")
    parser.add_argument("--decoder-lr", default=None, type=float,
                        help="Decoder learning rate, overrides --lr")

    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--dropout", default=None, type=float,
                        help="Dropout amount for the encoder and decoder, by defalut checkpoint value is used")
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--label-smoothing", default=None, type=float)

    # --- retrain-specific
    parser.add_argument("--move-norm", default=None, type=float,
                        help="Regularization coefficient for the distance between the initial and current network")
    parser.add_argument("--move-norm-p", default=2, type=int,
                        help="Parameter p of the L-p norm used in move-norm regularization")
    parser.add_argument("--no-opt-state", default=False, action="store_true",
                        help="Initialize optimizer state randomly instead of loading it from the trainer checkpoint")
    parser.add_argument("--no-lr-scheduler", default=False, action="store_true",
                        help="Keep learning rate constant instead of scheduling it. Only works with retrain_simple.")
    parser.add_argument("--weight-consolidation", default=None, type=float,
                        help="Weight consolidation regularization strength.")

    # --- freezing
    parser.add_argument("--freeze-encoder", default=None, type=int,
                        help="Step to freeze encoder")
    parser.add_argument("--unfreeze-encoder", default=None, type=int,
                        help="Step to unfreeze encoder")
    parser.add_argument("--freeze-decoder", default=None, type=int,
                        help="Step to freeze decoder")
    parser.add_argument("--unfreeze-decoder", default=None, type=int,
                        help="Step to unfreeze decoder")
    parser.add_argument("--freeze-head", default=None, type=int,
                        help="Step to freeze head")
    parser.add_argument("--unfreeze-head", default=None, type=int,
                        help="Step to unfreeze head")

    # misc
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--disable-wandb", default=False, action="store_true",
                        help="do not use wandb, mainly used for testing")
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--device", default=None,
                        help="Device to train the model on, cuda if available by default")
    parser.add_argument("--clean-output", default=False, action="store_true")
    parser.add_argument("--split-amount-finetune", default=None, type=float,
                        help="Only used for logging, amount of data that was removed from the training set")
    parser.add_argument("--cl-iteration", default=0, type=int,
                        help="used to log the cl batch number, useful for plotting learning curves")

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields

    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    if args.encoder_lr is None and args.lr is not None:
        args.encoder_lr = args.lr
        args.decoder_lr = args.lr

    if args.lr is None and args.encoder_lr is not None:
        args.lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tags = args.tags.split(",") if args.tags else []  # list is required by wandb interface
    args.new_classes = args.new_classes.split(",") if args.new_classes else []

    if args.split_amount_finetune is not None:
        args.split_amount_train = 1.0 - args.split_amount_finetune

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    if not (0 < args.new_data_amount <= 1):
        raise ValueError(f"--new-data-amount should be between 0 and 1 (exclusive)")

    if not (0 <= args.old_data_amount <= 1):
        raise ValueError(f"--old-data-amount should be between 0 and 1 (inclusive)")

    if not hasattr(nsp.dataclasses.SamplingMethods, args.old_data_sampling_method):
        raise ValueError(args.old_data_sampling_method)

    return args


def check_args(args):
    if args.lr is not None:
        raise ValueError(
            "Learning rate specification is not supported, use retrain_simple.py instead."
        )
    if args.no_lr_scheduler:
        raise ValueError("--no-lr-scheduler is not supported, use retrain_simple.py instead.")


def load_tokenizer(model_dir, data_dir):
    tokenizer_path1 = model_dir
    tokenizer_path2 = path_join(data_dir, "tokenizer")

    if os.path.exists(path_join(tokenizer_path1, "schema_vocab.txt")):
        return nsp.TopSchemaTokenizer.load(tokenizer_path1)

    if os.path.exists(tokenizer_path2):
        return nsp.TopSchemaTokenizer.load(tokenizer_path2)

    raise ValueError("Tokenizer is not found in both --model-dir and --data-dir.")


def load_data(path, new_data_amount, old_data_amount, old_data_sampling_method):
    datasets = torch.load(path)
    finetune_dataset: nsp.PointerDataset = datasets["finetune_dataset"]
    if finetune_dataset is None:
        raise RuntimeError("Datafile provided does not contain finetune_dataset.")

    eval_dataset: nsp.PointerDataset = datasets["valid_dataset"]

    train_subset = finetune_dataset
    if new_data_amount is not None and new_data_amount < 1.0:
        train_subset = utils.make_subset(train_subset, new_data_amount)

    wandb.config.update({"num_new_data": len(train_subset)})

    if old_data_amount > 0:
        if old_data_sampling_method == nsp.dataclasses.SamplingMethods.merge_subset:
            old_data_subset = utils.make_subset(datasets["train_dataset"], old_data_amount)
            train_subset = torch.utils.data.ConcatDataset([train_subset, old_data_subset])

        elif old_data_sampling_method == nsp.dataclasses.SamplingMethods.sample:
            train_subset = nsp.data.SampleConcatSubset(
                train_subset, datasets["train_dataset"], old_data_amount
            )

        else:
            raise ValueError(old_data_sampling_method)

    return train_subset, eval_dataset


def load_model(
    model_dir,
    dropout=None,
    move_norm=None,
    move_norm_p=None,
    label_smoothing=None,
    weight_consolidation=None,
):
    """Load a trained model and override some model properties if specified."""
    model_config = nsp.EncoderDecoderWPointerConfig.from_pretrained(model_dir)
    model_config.track_grad_square = False

    if dropout is not None:
        model_config.set_dropout(dropout)
    if move_norm is not None:
        model_config.move_norm = move_norm
    if move_norm_p is not None:
        model_config.move_norm_p = move_norm_p
    if label_smoothing is not None:
        model_config.label_smoothing = label_smoothing
    if weight_consolidation is not None:
        model_config.weight_consolidation = weight_consolidation

    model = nsp.EncoderDecoderWPointerModel.from_pretrained(model_dir, config=model_config)
    model.reset_initial_params()
    return model


def average_checkpoints(new_model, args, save_to):
    old_model = load_model(
        args.model_dir, args.dropout, args.move_norm, args.move_norm_p, args.label_smoothing
    )
    new_model = cli_utils.average_models(
        old_model=old_model,
        new_model=new_model,
        new_model_weight=args.new_model_weight,
    )
    new_model.save_pretrained(save_to)


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    mode = "disabled" if args.disable_wandb else None
    wandb.init(project=args.wandb_project, tags=args.tags, config=args, mode=mode)

    logger.info(f"Starting finetuning with args: \n{pprint.pformat(vars(args))}")

    logger.info("Loading tokenizers")
    schema_tokenizer = load_tokenizer(args.model_dir, args.data_dir)

    logger.info("Loading data")
    train_dataset, eval_dataset = load_data(
        path=path_join(args.data_dir, "data.pkl"),
        new_data_amount=args.new_data_amount,
        old_data_amount=args.old_data_amount,
        old_data_sampling_method=args.old_data_sampling_method,
    )
    train_args = cli_utils.load_saved_args(path_join(args.model_dir, "args.toml"))

    # NOTE: do not log metrics as hyperparameters
    wandb.config.update({"pretrain_" + k: v for k, v in train_args.items() if k != "metrics"})
    wandb.config.update({"num_total_data": len(train_dataset)})

    logger.info("Loading model")
    model = load_model(
        model_dir=args.model_dir,
        dropout=args.dropout,
        move_norm=args.move_norm,
        move_norm_p=args.move_norm_p,
        label_smoothing=args.label_smoothing,
        weight_consolidation=args.weight_consolidation,
    )

    logger.info("Preparing for training")

    max_tgt_len = train_args.get("max_tgt_len", 68)  # 68 is max_tgt_len for TOP
    # this is not a lightning module anymore
    freezing_schedule = nsp.dataclasses.EncDecFreezingSchedule.from_args(args)

    lightning_module = nsp.PointerModule(
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        lr=args.lr or train_args["lr"],
        batch_size=args.batch_size or train_args["batch_size"],
        warmup_steps=args.warmup_steps or train_args["warmup_steps"],
        weight_decay=args.weight_decay or train_args["weight_decay"],
        log_every=args.log_every or train_args["log_every"],
        monitor_classes=args.new_classes or train_args["new_classes"],
        freezing_schedule=freezing_schedule,
        max_tgt_len=max_tgt_len,
        no_lr_scheduler=getattr(args, "no_lr_scheduler", False),
    )

    trainer = nsp.Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        device=args.device,
        gradient_clip_val=args.max_grad_norm,
        early_stopping_metric="eval_exact_match",
        patience=args.early_stopping,
        maximize_early_stopping_metric=True,
        limit_val_batches=args.eval_data_amount,
        save_dir=args.output_dir,
    )

    lightning_module = lightning_module.to(
        args.device
    )  # required to load optimizer state to the correct device
    optimizer_and_scheduler = lightning_module.configure_optimizers()
    optimizer_and_scheduler = trainer.load_optimizer_and_scheduler_states(
        optimizer_and_scheduler, args.model_dir
    )

    # get evaluation metrics of the initial model
    pretrain_metrics = train_args["metrics"]
    _first_step_metrics = {
        "epoch": 0,
        "global_step": 0,
        **pretrain_metrics["means"],
        **pretrain_metrics["stdevs"],
    }
    wandb.log(_first_step_metrics, step=0)
    wandb.watch(lightning_module, log="all", log_freq=lightning_module.log_every)

    # --- FIT

    cli_utils.check_config(lightning_module, trainer, args, strict=True)

    if model.initial_params is not None:
        assert torch.allclose(model.get_move_norm(), torch.zeros(1, device=model.device))

    assert lightning_module.model.config.track_grad_square is False, "you should not update grad_square during fine-tuning"
    trainer.fit(lightning_module, optimizer_and_scheduler)

    cli_utils.check_config(lightning_module, trainer, args, strict=True)

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": nsp.SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    logger.info("Training finished!")

    if args.average_checkpoints:
        raise NotImplementedError("Average checkpoints is not supported anymore")
        # average_checkpoints(model, args, save_to=os.path.dirname(best_model_checkpoint))

    final_metrics, description = cli_utils.evaluate_model_n_rounds(
        trainer.model.model,
        schema_tokenizer,
        trainer.valid_dataloader,
        prefix="eval",
        max_len=max_tgt_len,
    )

    logger.info(description)
    wandb.log(
        {**final_metrics["means"], **final_metrics["stdevs"]}, step=trainer.model.global_step
    )

    # Compute RI and RD
    class_weights = eval_dataset.get_class_frequencies(schema_tokenizer)
    class_weights = {f"cls/eval_{cls}_tree_path_f1": p for cls, p in class_weights.items()}

    finetuning_metrics = cli_utils.evaluate_finetuning_procedure(
        pretrain_metrics, final_metrics, class_weights
    )
    wandb.log(finetuning_metrics, step=trainer.model.global_step)

    # Compute RI and RD with very small outliers stuff
    finetuning_metrics0 = cli_utils.evaluate_finetuning_procedure(
        pretrain_metrics, final_metrics, class_weights, sigma=0.0
    )
    finetuning_metrics0 = {k + "_0.0": v for k, v in finetuning_metrics0.items()}
    wandb.log(finetuning_metrics0, step=trainer.model.global_step)

    if args.clean_output:
        shutil.rmtree(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    main(args)
