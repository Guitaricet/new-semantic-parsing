# Copyright 2021 Vladislav Lialin
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
from os.path import join as path_join

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
                        help="path to a directory containing train.csv, eval.csv, test.csv and vocab.txt")
    parser.add_argument("--text-tokenizer", required=True,
                        help="pratrained tokenizer name or path to a saved tokenizer")
    parser.add_argument("--output-dir", required=True,
                        help="directory to save preprocessed data")
    parser.add_argument("--seed", default=34)

    # fmt: on

    args = parser.parse_args(args)

    return args


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    # File structure:
    # that's text\tthat 's text\t[IN:UNSUPPORTED that 's text]
    train_path = path_join(args.data, "train.tsv")

    logger.info("Getting schema vocabulary")
    with open(path_join(args.data, "vocab.txt")) as f:
        schema_vocab = f.read().split("\n")

    logger.info("Finished splitting data, pre-processing each dataset")
    logger.info("Building tokenizers")
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_tokenizer, use_fast=True)
    schema_tokenizer = nsp.TopSchemaTokenizer(schema_vocab, text_tokenizer)

    logger.info("Tokenizing train dataset")
    train_dataset = nsp.data.make_dataset(train_path, schema_tokenizer)

    logger.info("Tokenizing validation and test datasets")
    valid_dataset = nsp.data.make_dataset(path_join(args.data, "eval.tsv"), schema_tokenizer)
    test_dataset = nsp.data.make_dataset(path_join(args.data, "test.tsv"), schema_tokenizer)

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
        "split_type": "class_incremental_multiple_batches",
        "version": nsp.SAVE_FORMAT_VERSION,
    }

    torch.save(data_state, path_join(args.output_dir, "data.pkl"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
