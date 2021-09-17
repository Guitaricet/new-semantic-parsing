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
"""Preprocess text data and save binary Dataset objects along with tokenizers to a directory."""

import os
import sys
import logging
import argparse
from functools import reduce
from os.path import join as path_join
from random import shuffle

import toml
import torch
import pandas as pd

import transformers

import new_semantic_parsing as nsp
from new_semantic_parsing import utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--data", required=True,
                        help="path to TOP dataset directory")
    parser.add_argument("--text-tokenizer", required=True,
                        help="pratrained tokenizer name or path to a saved tokenizer")
    parser.add_argument("--output-dir", required=True,
                        help="directory to save preprocessed data")
    parser.add_argument("--seed", default=34)

    # splitting parameters
    parser.add_argument("--split-class", default=None,
                        help="remove --split-ratio of the class from the training dataset and make a finetune_data; "
                             "do not perform split by default")
    parser.add_argument("--split-amount", default=None, type=float,
                        help="0 < --split-amount < 1, amount of data to remove from the training dataset")
    # fmt: on

    args = parser.parse_args(args)

    if args.split_amount is not None:
        if not 0.0 < args.split_amount < 1.0:
            raise ValueError("--split-amount should be between 0. and 1.")

    if args.split_class is not None:
        if args.split_amount is None:
            raise ValueError("--split-amount should be specified if --split-class is provided")

    return args


def train_finetune_split(train_data, schema_vocab, split_amount, split_class=None):
    """Split train_data into train and finetune parts with ratio split_amount.

    Train part should contain all classses from the original train_data.
    If split_class is provided, split across examples containing this class.
    E.i. split_amount of data with split_class goes to finetune set.

    Args:
        train_data: pd.DataFrame
        schema_vocab: set of tokens
        split_amount: float
        split_class: if provided, split across the specified class
    """
    # Get a small set of examples that contains all classes from schema_vocab
    required_example_ids = utils.get_required_example_ids(schema_vocab, train_data)

    ids = set(range(len(train_data)))
    if split_class is not None:
        ids = set(train_data.index[train_data.schema.str.contains(split_class)])
        logger.info(f"Moving {100 * split_amount}% of {split_class} into a finetuning subset")

        _take = int(len(ids) * split_amount)
        _leave = len(ids) - _take
        logger.info(
            f"Take {_take} class examples to finetuning set and leave {_leave} class examles in"
            " training set."
        )
        if len(ids) == 0:
            raise RuntimeError(f"Cannot find specified class {split_class} in the data.")

    split_ids = list(ids - required_example_ids)

    take = int(len(split_ids) * split_amount)
    leave = len(train_data) - take

    assert take > 0

    logger.info(f"Taking {take} examples and leaving {leave} examples")

    shuffle(split_ids)
    subset_ids = split_ids[:take]

    subset_ids_set = set(subset_ids)
    all_ids = set(range(len(train_data)))

    assert len(subset_ids_set.intersection(required_example_ids)) == 0
    train_data_ids = list(all_ids - subset_ids_set | required_example_ids)

    finetune_data = train_data.iloc[subset_ids]
    train_data = train_data.iloc[train_data_ids]

    return train_data, finetune_data


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    # File structure:
    # that's text\tthat 's text\t[IN:UNSUPPORTED that 's text]
    train_path = path_join(path_join(args.data, "train.tsv"))
    train_data = pd.read_table(train_path, names=["text", "tokens", "schema"])
    full_train_data_size = len(train_data)  # used to check the train/finetune split
    finetune_data, finetune_path = None, None

    schema_vocab = reduce(set.union, map(utils.get_vocab_top_schema, train_data.schema))

    if args.split_amount is not None:
        # finetune part is not used by train script, but used by retrain script
        logger.info("Splitting the training dataset")
        train_data, finetune_data = train_finetune_split(
            train_data, schema_vocab, args.split_amount, args.split_class
        )

        os.makedirs(args.output_dir)

        finetune_path = path_join(args.output_dir, "finetune.tsv")
        logger.info(f"Saving the finetune_data to {finetune_path}")
        finetune_data.to_csv(finetune_path, sep="\t", index=False, header=False)

        train_path = path_join(args.output_dir, "train.tsv")
        logger.info(f"Saving the modified training set to {train_path}")
        train_data.to_csv(train_path, sep="\t", index=False, header=False)

    logger.info("Getting schema vocabulary")

    if args.split_amount is not None:
        finetune_schema_vocab = reduce(
            set.union, map(utils.get_vocab_top_schema, finetune_data.schema)
        )
        vocab_delta = finetune_schema_vocab - schema_vocab
        if len(vocab_delta) > 0:
            logger.warning(
                f"Finetuning subset contains vocabulary elements not from the training subset"
            )
            logger.warning(f"New elements: {', '.join(vocab_delta)}")

    logger.info(f"Schema vocabulary size: {len(schema_vocab)}")

    logger.info("Building tokenizers")
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_tokenizer, use_fast=True)
    max_src_len = max([len(text_tokenizer.encode(t)) for t in train_data.tokens])

    logger.info(f"Maximum source text length is {max_src_len}")
    if max_src_len > 512:
        logger.warning(f"max_src_len is {max_src_len}, you might be in trouble")

    schema_tokenizer = nsp.TopSchemaTokenizer(schema_vocab, text_tokenizer, max_pointer_len=max_src_len)

    logger.info("Tokenizing train dataset")
    train_dataset = nsp.data.make_dataset(train_path, schema_tokenizer)

    logger.info("Tokenizing validation and test datasets")
    valid_dataset = nsp.data.make_dataset(path_join(args.data, "eval.tsv"), schema_tokenizer)
    test_dataset = nsp.data.make_dataset(path_join(args.data, "test.tsv"), schema_tokenizer)

    finetune_dataset = None
    if args.split_amount is not None:
        logger.info("Tokenizing finetune set")
        finetune_dataset = nsp.data.make_dataset(finetune_path, schema_tokenizer)

        logger.info(f"Original train set size: {full_train_data_size}")
        logger.info(f"Reduced  train set size: {len(train_dataset)}")
        logger.info(f"Finetune       set size: {len(finetune_dataset)}")

        train_finetune_data_size = len(train_dataset) + len(finetune_dataset)
        if train_finetune_data_size != full_train_data_size:
            raise RuntimeError(f"{train_finetune_data_size} != {full_train_data_size}")

    logger.info(f"Saving config, data and tokenizer to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": nsp.SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    # text tokenizer is saved along with schema_tokenizer
    model_type = None
    if not os.path.exists(args.text_tokenizer):
        model_type = utils.get_model_type(args.text_tokenizer)

    schema_tokenizer.save(path_join(args.output_dir, "tokenizer"), encoder_model_type=model_type)

    data_state = {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "finetune_dataset": finetune_dataset,
        "version": nsp.SAVE_FORMAT_VERSION,
    }

    torch.save(data_state, path_join(args.output_dir, "data.pkl"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
