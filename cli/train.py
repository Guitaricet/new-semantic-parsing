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
"""Train the model using preprocessed (binary) data and save the model and tokenizer into a directory."""

import argparse
import logging
import pprint
import os
import sys
import tempfile
from datetime import datetime
from os.path import join as path_join

import pandas as pd
import toml
import torch
import transformers
import wandb

import new_semantic_parsing as nsp
import new_semantic_parsing.dataclasses
import new_semantic_parsing.optimization

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
    parser = argparse.ArgumentParser()

    # fmt: off

    # data
    parser.add_argument("--data-dir", required=True,
                        help="Path to preprocess.py --save-dir containing tokenizer, "
                             "data.pkl, and args.toml")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store checkpoints and other output files")
    parser.add_argument("--eval-data-amount", default=1.0, type=float,
                        help="Amount of validation set to use when training. "
                             "The final evaluation will use the full dataset.")
    parser.add_argument("--new-classes", default=None,
                        help="names of classes to track")
    parser.add_argument("--max-src-len", default=None, type=int, help="Use 63 for TOP dataset")
    parser.add_argument("--max-tgt-len", default=None, type=int, help="Use 98 for TOP dataset")

    # model
    parser.add_argument("--encoder-model", default=None,
                        help="Pretrained model name, e.g. bert-base-uncased")
    parser.add_argument("--layers", default=None, type=int,
                        help="Number of layers in the encoder. "
                             "Only used if --encoder-model is not provided.")
    parser.add_argument("--hidden", default=None, type=int,
                        help="Hidden size of the encoder. "
                             "Only used if --encoder-model is not provided.")
    parser.add_argument("--heads", default=None, type=int,
                        help="Hidden size of the encoder. "
                             "Only used if --encoder-model is not provided.")
    parser.add_argument("--decoder-layers", default=None, type=int,
                        help="Number of layers in the decoder. "
                             "Equal to the number of the encoder layers by default")
    parser.add_argument("--decoder-hidden", default=None, type=int,
                        help="Hidden size of the decoder. "
                             "Equal to the hidden side of the encoder by default")
    parser.add_argument("--decoder-heads", default=None, type=int,
                        help="Hidden size of the decoder. "
                             "Equal to the number of the encoder heads by default")

    # model architecture changes
    parser.add_argument("--use-pointer-bias", default=False, action="store_true",
                        help="Use bias in pointer network")

    # training
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--min-epochs", default=1, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--min-steps", default=None, type=int)
    parser.add_argument("--early-stopping", default=None, type=int,
                        help="Early stopping patience. No early stopping by default.")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--encoder-lr", default=None, type=float,
                        help="Encoder learning rate, overrides --lr")
    parser.add_argument("--decoder-lr", default=None, type=float,
                        help="Decoder learning rate, overrides --lr")

    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout amount for the encoder, decoder and head, default value 0.1 is from Transformers")
    parser.add_argument("--warmup-steps", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--max-param-importance", default=0.001, type=float)
    parser.add_argument("--new-param-importance-scale", default=0.00001, type=float)

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
    parser.add_argument("--split-amount-finetune", default=None, type=float,
                        help="Only used for logging, amount of data that was removed from the training set")
    parser.add_argument("--alert-em", default=0.7, type=float,
                        help="use wandb alert if the final exact match is below this value")
    parser.add_argument("--aggregation-file", default="final_metrics.csv",
                        help="append the final metrics to this file, used for plotting")
    parser.add_argument("--skip-training", default=False, action="store_true",
                        help="Only used for debug")
    parser.add_argument("--use-synaptic-intelligence", default=False, action="store_true",
                        help="Enables AdamSI optimizer that tracks the path integral needed for Synaptic Intelligence")

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields
    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    if args.encoder_lr is not None:
        args.lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    args.decoder_layers = args.decoder_layers or args.layers
    args.decoder_hidden = args.decoder_hidden or args.hidden
    args.decoder_heads = args.decoder_heads or args.heads
    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tags = args.tags.split(",") if args.tags else []  # list is required by wandb interface
    args.new_classes = args.new_classes.split(",") if args.new_classes else []

    if args.split_amount_finetune is not None:
        args.split_amount_train = 1.0 - args.split_amount_finetune

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    if args.label_smoothing > 0:
        raise ValueError("Label smoothing is broken or requires significantly different hyperparameters")

    return args


def make_model(schema_tokenizer, max_src_len, args, preprocess_args=None, use_synaptic_intelligence=False):
    """Initialize a model.
    If args.encoder_model is specified, use it to load a pretrained encoder model.

    Args:
        schema_tokenizer: TopSchemaTokenizer
        max_src_len: int, maximum length of the source sequence in BPE tokens
        args: argparse object from parse_args()
        preprocess_args: (optional) dict, used to check the tokenizer used for preprocessing

    Returns:
        EncoderDecoderWPointerModel
    """
    if not args.encoder_model:
        # initialize the model from scratch
        model = nsp.EncoderDecoderWPointerModel.from_parameters(
            layers=args.layers,
            hidden=args.hidden,
            heads=args.heads,
            decoder_layers=args.decoder_layers,
            decoder_hidden=args.decoder_hidden,
            decoder_heads=args.decoder_heads,
            src_vocab_size=schema_tokenizer.src_tokenizer.vocab_size,
            tgt_vocab_size=schema_tokenizer.vocab_size,
            encoder_pad_token_id=schema_tokenizer.src_tokenizer.pad_token_id,
            decoder_pad_token_id=schema_tokenizer.pad_token_id,
            max_src_len=max_src_len,
            dropout=args.dropout,
            model_args=args,
        )
        if use_synaptic_intelligence:
            model.reset_initial_params()

        return model

    # use a pretrained model as an encoder

    if preprocess_args is not None and preprocess_args["text_tokenizer"] != args.encoder_model:
        logger.warning("Data may have been preprocessed with a different tokenizer")
        logger.warning(f'Preprocessing tokenizer     : {preprocess_args["text_tokenizer"]}')
        logger.warning(f"Pretrained encoder tokenizer: {args.encoder_model}")

    encoder_config = transformers.AutoConfig.from_pretrained(args.encoder_model)
    encoder_config.hidden_dropout_prob = args.dropout
    encoder_config.attention_probs_dropout_prob = args.dropout

    encoder = transformers.AutoModel.from_pretrained(args.encoder_model, config=encoder_config)

    if encoder.config.vocab_size != schema_tokenizer.src_tokenizer.vocab_size:
        raise ValueError("Preprocessing tokenizer and model tokenizer are not compatible")

    ffn_hidden = 4 * args.decoder_hidden if args.decoder_hidden is not None else None

    decoder_config = transformers.BertConfig(
        is_decoder=True,
        vocab_size=schema_tokenizer.vocab_size + max_src_len,
        hidden_size=args.decoder_hidden or encoder.config.hidden_size,
        intermediate_size=ffn_hidden or encoder.config.intermediate_size,
        num_hidden_layers=args.decoder_layers or encoder.config.num_hidden_layers,
        num_attention_heads=args.decoder_heads or encoder.config.num_attention_heads,
        pad_token_id=schema_tokenizer.pad_token_id,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )
    decoder = transformers.BertModel(decoder_config)

    model = nsp.EncoderDecoderWPointerModel(
        encoder=encoder,
        decoder=decoder,
        max_src_len=max_src_len,
        dropout=args.dropout,
        model_args=args,
    )

    if use_synaptic_intelligence:
        model.reset_initial_params()

    return model


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    mode = "disabled" if args.disable_wandb else None
    wandb.init(project=args.wandb_project, tags=args.tags, config=args, mode=mode)
    wandb.config.update({"new_classes": " ".join(args.new_classes)}, allow_val_change=True)

    logger.info(f"Starting training with args: \n{pprint.pformat(vars(args))}")

    logger.info("Loading tokenizers")
    schema_tokenizer = nsp.TopSchemaTokenizer.load(path_join(args.data_dir, "tokenizer"))

    logger.info("Loading data")
    datasets = torch.load(path_join(args.data_dir, "data.pkl"))
    train_dataset: nsp.PointerDataset = datasets["train_dataset"]
    eval_dataset: nsp.PointerDataset = datasets["valid_dataset"]

    wandb.config.update({
        "num_data": len(train_dataset),
        "batch_number": 0,
        "num_total_data": len(train_dataset),
    })

    max_src_len, max_tgt_len = train_dataset.get_max_len()
    logger.info(f"Maximum training set length, src: {max_src_len} tgt: {max_tgt_len}")

    if args.max_src_len is not None and args.max_src_len < max_src_len:
        logger.warning(f"--max-src-len is smaller than the training dataset max len = {max_src_len}")

    eval_max_src_len, eval_max_tgt_len = eval_dataset.get_max_len()
    logger.info(f"Maximum eval set length, src: {eval_max_src_len} tgt: {eval_max_tgt_len}")

    if args.max_src_len is not None and args.max_src_len < eval_max_src_len:
        logger.warning(f"--max-src-len is smaller than the eval dataset max len = {max_src_len}")

    max_src_len = args.max_src_len or max_src_len
    max_tgt_len = args.max_tgt_len or max_tgt_len

    try:
        preprocess_args = cli_utils.load_saved_args(path_join(args.data_dir, "args.toml"))
        wandb.config.update({"preprocess_" + k: v for k, v in preprocess_args.items()})

    except FileNotFoundError:
        preprocess_args = None

    logger.info("Creating a model")
    model = make_model(
        schema_tokenizer=schema_tokenizer,
        max_src_len=max_src_len,
        args=args,
        preprocess_args=preprocess_args,
        use_synaptic_intelligence=args.use_synaptic_intelligence,
    )

    logger.info("Preparing for training")

    freezing_schedule = nsp.dataclasses.EncDecFreezingSchedule.from_args(args)
    # NOTE: this is not a lightning module object, but it follows its interface
    lightning_module = nsp.PointerModule(
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        monitor_classes=args.new_classes,
        freezing_schedule=freezing_schedule,
        max_tgt_len=max_tgt_len,
        no_lr_scheduler=getattr(args, "no_lr_scheduler", False),
    )

    wandb.watch(lightning_module, log="all", log_freq=args.log_every)

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

    # --- FIT
    cli_utils.check_config(lightning_module, trainer, args)

    if args.skip_training:
        logger.info(f"--skip-training has been specified, saving initialized model to {args.output_dir}")
        trainer.setup_trainer(lightning_module)
        trainer.save()

        with open(path_join(args.output_dir, "args.toml"), "w") as f:
            logger.info("Saving the metrics to the args.toml file")
            args_dict = {
                "version": nsp.SAVE_FORMAT_VERSION,
                "max_src_len": max_src_len,
                "max_tgt_len": max_tgt_len,
                **vars(args),
            }
            toml.dump(args_dict, f)

        logger.info("Script finished successfully")
        return

    trainer.fit(model=lightning_module, use_synaptic_intelligence=args.use_synaptic_intelligence)

    # Save weight importance
    if isinstance(trainer.optimizer, nsp.optimization.AdamSI):
        adam_si: nsp.optimization.AdamSI = trainer.optimizer
        model.set_new_param_importance(adam_si.omega, scale=args.new_param_importance_scale)
        model.save_pretrained(args.output_dir)

    logger.info("Training finished!")

    logger.info(f"Evaluating the best model")
    final_metrics, description = cli_utils.evaluate_model_n_rounds(
        trainer.model.model,
        schema_tokenizer,
        trainer.valid_dataloader,
        prefix="eval",
        max_len=max_tgt_len,
    )
    logger.info("Finished evaluation")

    final_em = final_metrics["means"]["eval_exact_match"]
    if final_em < 0.7:
        wandb.alert(title="low EM", text=f"eval_exact_match {final_em}")

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        logger.info("Saving the metrics to the args.toml file")
        args_dict = {
            "version": nsp.SAVE_FORMAT_VERSION,
            "metrics": final_metrics,
            "max_src_len": max_src_len,
            "max_tgt_len": max_tgt_len,
            **vars(args),
        }
        toml.dump(args_dict, f)

    wandb.log({**final_metrics["means"], **final_metrics["stdevs"]}, step=trainer.model.global_step)

    metrics_to_plot = pd.Series({
        "cl_iteration": 0,
        "eval_exact_match": final_metrics["means"]["eval_exact_match"],
        "eval_tree_path_f1": final_metrics["means"]["eval_tree_path_f1"],
        "datetime": datetime.now(),
    })

    logger.info(f"Saving the metrics to aggregation file {args.aggregation_file}")
    if os.path.exists(args.aggregation_file):
        agg_df = pd.read_csv(args.aggregation_file)
        try:
            agg_df = agg_df.append(metrics_to_plot, ignore_index=True)
        except Exception as e:
            logger.error(e)
            logger.info("Continuing script")
    else:
        agg_df = pd.DataFrame([metrics_to_plot])

    agg_df.to_csv(args.aggregation_file, index=False)

    logger.info("Finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)
