# Copyright 2020 Google LLC
# Copyright 2020 The HuggingFace Inc. team.
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
"""Utilities used across code.

Include fixing random seeds, metrics computation, learning rate selection, model loading, and prediction.
"""
import random
import re
import json

import numpy as np
import pandas as pd
import torch
import transformers


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_vocab_top_schema(text):
    schema_tokens = {"[", "]", "IN:", "SL:"}

    text = text.replace("[", "")
    text = text.replace("]", "")

    for token in text.split(" "):
        if token[:3] in ["IN:", "SL:"]:
            schema_tokens.add(token[3:])
    return schema_tokens


def get_src_pointer_mask(input_ids, tokenizer: transformers.PreTrainedTokenizer):
    """Computes mask which ignores padding and special tokens"""
    mask = np.ones(len(input_ids))
    if input_ids[0] == tokenizer.cls_token_id:
        mask[0] = 0
    for i, token_id in enumerate(input_ids):
        if token_id in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            mask[i] = 0
    return mask


def get_model_type(model_name):
    """Searches for a largest substring from transformers.CONFIG_MAPPING"""
    candidate = ""

    for name in transformers.CONFIG_MAPPING:
        if name in model_name and len(name) > len(candidate):
            candidate = name

    if len(candidate) == 0:
        raise ValueError(f"{model_name} is not found in transformers.CONFIG_MAPPING")

    return candidate


def make_subset(dataset, subset_size):
    """Makes torch Subset by randomly sampling indices from dataset

    Args:
        dataset: torch Dataset
        subset_size: float, 0 < subset_size < 1
    """
    if subset_size == 1:
        return dataset

    if not (0 < subset_size < 1):
        raise ValueError(subset_size)

    _subset_size = int(subset_size * len(dataset))
    _subset_ids = np.random.permutation(len(dataset))[:_subset_size]

    _subset = torch.utils.data.Subset(dataset, indices=_subset_ids)
    return _subset


def get_required_example_ids(schema_vocab, train_data):
    """Finds a subset of train_data that contains all schema_vocab tokens.

    Args:
        schema_vocab: set of str, required schema tokens
        train_data: pd.DataFrame with field "schema"

    Returns:
        a set of train_data ids
    """
    required_schema_vocab = set()
    required_example_ids = set()

    for i, row in train_data.iterrows():
        add_this = False
        tokens_not_present = [w for w in schema_vocab if w not in required_schema_vocab]

        # Add the example id to required_example_ids if the example
        # contains a schema token not present in the required_schema_vocab
        for token in tokens_not_present:
            if token in row.schema:
                add_this = True
                required_schema_vocab.add(token)

        if add_this:
            required_example_ids.add(i)

        if sorted(required_schema_vocab) == sorted(schema_vocab):
            break
    else:
        raise RuntimeError(f"Full vocabulary was not found in the training set. "
                           f"tokens_not_present: {tokens_not_present}")

    return required_example_ids


def matches_pattern(string, pattern):
    if pattern is None:
        return True

    return re.match(pattern, string) is not None


def snips2top(snips_example, intent):
    """Converts Snips format to TOP format

    Args:
        snips_example: list, one example following snips format
        intent: str

    Returns:
        query_text, top_format_schema
    """
    query_text = ""
    top_format_str = f"[IN:{intent.upper()}"

    for text_chunk in snips_example:
        text = text_chunk["text"].strip(" ")

        if "entity" in text_chunk:
            entity_name = text_chunk["entity"].upper()
            top_format_str += f" [SL:{entity_name} {text} ]"

        else:
            top_format_str += " " + text

        query_text += " " + text

    query_text = query_text.strip(" ")
    top_format_str += " ]"

    return query_text, top_format_str


def make_snips_df(snips_files):
    snips_data = []
    for train_file in snips_files:
        with open(train_file, encoding="latin-1") as f:
            data = json.load(f)

        assert len(data.keys()) == 1, data.keys()
        intent = list(data.keys())[0]

        for example in data[intent]:
            assert len(example.keys()) == 1
            text, schema = snips2top(example["data"], intent)
            snips_data.append([text, text, schema])

    snips_df = pd.DataFrame(snips_data, columns=["text", "tokens", "schema"])
    return snips_df


@torch.no_grad()
def get_dynamic_ewc_weight(loss_value, ewc_reg_value):
    return torch.log(loss_value / ewc_reg_value)
