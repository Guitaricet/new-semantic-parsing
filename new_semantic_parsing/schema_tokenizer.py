# Copyright 2020 Google LLC
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenizer analogous to transformers.PreTrainedTokenizer which handles tokenization and numericalization.

Mostly follows PreTrainedTokenizer interface with exceptions of task-specific methods such as
source text encoding and pair (both source and target texts) encoding.
Also handles postprocessing generated outputs into TOP format.
"""

import os
import json
from os.path import join as path_join

import transformers

from new_semantic_parsing import utils
from new_semantic_parsing.dataclasses import SchemaItem, PairItem
from new_semantic_parsing.configuration_encoder_decoder_wpointer import (
    EncoderDecoderWPointerConfig,
)


class TopSchemaTokenizer:
    """
    Handles tokenization of TOP schema.

    Encodes schema into token_ids from schema_vocab and words into position-based ids.
    Used for *both* source sentence and schema.

    word_id = tokenizer.vocab_size + position

    [CLS] token is ignored for position calculation

    Usage:
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
        schema_vocab = reduce(set.union, map(utils.get_vocab_top_schema, schema_examples))
        schema_tokenizer = TopSchemaTokenizer(schema_vocab, tokenizer)

        # encode pair
        source_str = 'Go to Mountain View'
        schema_str = '[IN:GET_DIRECTIONS Go to [SL:Location Mountain View]]'

        pair: PairItem = schema_tokenizer.encode_pair(source_str, schema_str)

        # Alternatively:
        # 1. encode source
        source_ids: List = schema_tokenizer.encode_source(source_str)

        # 2. encode schema
        source_ids: List = schema_tokenizer.encode(schema_str, source_ids)

    """

    def __init__(self, schema_vocab, src_text_tokenizer: transformers.PreTrainedTokenizer):
        """
        :param schema_vocab: list with all schema tokens (not source text tokens)
        :param src_text_tokenizer: transformers.PreTrainedTokenizer object
        """
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        if not isinstance(schema_vocab, list):
            raise ValueError("Class-incremental setup requires schema vocab order ordered as in vocab.txt")

        self.vocab = set(schema_vocab)
        self._itos = [self.pad_token, self.bos_token, self.eos_token] + schema_vocab
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self.src_tokenizer = src_text_tokenizer

    @property
    def vocab_size(self):
        return len(self._itos)

    @property
    def pad_token_id(self):
        return self._stoi[self.pad_token]

    @property
    def bos_token_id(self):
        return self._stoi[self.bos_token]

    @property
    def eos_token_id(self):
        return self._stoi[self.eos_token]

    @property
    def special_tokens(self):
        return [self.pad_token, self.bos_token, self.eos_token]

    @property
    def special_ids(self):
        return [self.pad_token_id, self.bos_token_id, self.eos_token_id]

    def index2token(self, i):
        if i < self.vocab_size:
            return self._itos[i]
        else:
            # returns ptr@{i-self.vocab_size}
            return self.decode([i])[0]

    def encode(self, schema_text, source_ids, max_length=None, pad_to_max_length=False):
        return self.encode_plus(schema_text, source_ids, max_length, pad_to_max_length).ids

    def encode_plus(
        self, schema_text, source_ids, max_length=None, pad_to_max_length=False
    ) -> SchemaItem:
        # NOTE: this method should do the same things as .batch_encode_plus
        schema_tokens = self.tokenize(schema_text)

        if max_length is not None:
            schema_tokens = schema_tokens[: max_length - 2]  # minus BOS and EOS

        schema_tokens = [self.bos_token] + schema_tokens + [self.eos_token]

        if pad_to_max_length:
            delta = max_length - len(schema_tokens)
            if delta > 0:
                schema_tokens += [self.pad_token] * delta

        item = self.convert_tokens_to_ids(schema_tokens, source_ids)

        return item

    def convert_tokens_to_ids(self, schema_tokens, src_token_ids) -> SchemaItem:
        """
        :param schema_tokens: string
        :param src_token_ids: list or numpy array of integers
        :return: list of integers - a mix of token ids and position ids
            position id = position + vocab_size
        """
        schema_ids = []
        pointer_mask = []

        # points to a first token corresponding to a word
        has_cls = (
            self.src_tokenizer.cls_token is not None
            and self.src_tokenizer.cls_token_id in src_token_ids
        )
        src_tokens_pointer = int(has_cls)

        for i, token in enumerate(schema_tokens):
            token_follows_schema = token in {
                "[",
                "]",
                "IN:",
                "SL:",
                *self.special_tokens,
            } or schema_tokens[i - 1] in {"IN:", "SL:"}
            if token in self._stoi and token_follows_schema:
                # The reason for second condition are cases when a word from a text exacly equal to the schema word
                # e.g. "IS THERE A PATH"
                # PATH is in a schema vocabulary, but not a schema word

                pointer_mask.append(0)
                schema_ids.append(self._stoi[token])
                continue

            subtokens = self.src_tokenizer.encode(token, add_special_tokens=False)

            for subtoken in subtokens:
                assert subtoken == src_token_ids[src_tokens_pointer]
                pointer_mask.append(1)
                schema_ids.append(self.vocab_size + src_tokens_pointer)
                src_tokens_pointer += 1

        return SchemaItem(schema_ids, pointer_mask)

    def decode(self, ids, source_ids=None, skip_special_tokens=True, return_tokens=False):
        schema = []
        # we combine text into chunks to that it would be easier to merge bpe tokens into words
        text_chunk_ids = []

        for i in ids:
            if i < self.vocab_size:
                if text_chunk_ids and source_ids is not None:
                    schema.append(self.src_tokenizer.decode(text_chunk_ids))
                    text_chunk_ids = []

                if skip_special_tokens and i in self.special_ids:
                    continue
                schema.append(self._itos[i])
            else:
                position = i - self.vocab_size
                if source_ids is not None:
                    text_chunk_ids.append(source_ids[position])
                else:
                    schema.append(f"@ptr{position}")

        if return_tokens:
            return schema

        schema = self.detokenize(schema)
        return schema

    def encode_pair(self, schema_text, source_text) -> PairItem:
        source_ids = self.encode_source(source_text)
        schema_item = self.encode_plus(schema_text, source_ids)
        schema_ids = schema_item.ids

        source_pointer_mask = utils.get_src_pointer_mask(source_ids, self.src_tokenizer)
        schema_pointer_mask = schema_item.pointer_mask

        pair = PairItem(source_ids, source_pointer_mask, schema_ids, schema_pointer_mask)
        return pair

    def encode_source(self, source_text):
        return self.src_tokenizer.encode(source_text)

    def is_schema_token_id(self, i):
        """Checks if i correspond to the schema token or to the pointer token."""
        is_special_token = i in [self.pad_token_id, self.bos_token_id, self.eos_token_id]
        return i < self.vocab_size and not is_special_token

    def save(self, path, encoder_model_type=None):
        """
        Save schema tokenizer and text tokenizer
        Needs pre-trained encoder model type - this is a workaround for Transformers #4197
        """
        os.makedirs(path, exist_ok=True)

        with open(path_join(path, "schema_vocab.txt"), "w") as f:
            f.write("\n".join(self._itos[3:]))  # the first three are the special tokens

        self.src_tokenizer.save_pretrained(path)

        if encoder_model_type is None:
            return

        with open(path_join(path, "config.json"), "w") as f:
            json.dump({"model_type": encoder_model_type}, f)

    @classmethod
    def load(cls, path: str):
        with open(path_join(path, "schema_vocab.txt")) as f:
            schema_vocab = f.read().strip("\n").split("\n")

        if not os.path.exists(path_join(path, "config.json")):
            raise RuntimeError(
                f"{path_join(path, 'config.json')} is required for tokenizer loading"
            )

        try:
            config = transformers.AutoConfig.from_pretrained(path)
        except KeyError:
            # config with custom model_type raises KeyError
            config = EncoderDecoderWPointerConfig.from_pretrained(path)
            config = config.encoder

        text_tokenizer = transformers.AutoTokenizer.from_pretrained(path, config=config)

        return cls(schema_vocab, text_tokenizer)

    @staticmethod
    def tokenize(text):
        # TODO: make a faster regex version
        tokenized = ""
        for char in text:
            if char in ["[", "]"]:
                char = " " + char + " "
            if char in [":"]:
                char = char + " "
            tokenized += char
        tokens = tokenized.strip(" ").split(" ")
        tokens = [t for t in tokens if t != ""]
        return tokens

    @staticmethod
    def detokenize(tokens):
        merge_vocab = {"[", "IN:", "SL:"}
        text = ""

        for token in tokens:
            if token in merge_vocab:
                text += token
            else:
                text += token + " "

        return text.strip(" ")

    @staticmethod
    def postprocess(text):
        """TOP format expects tokenized words and punctuation"""
        if len(text) == 0:
            return ""

        stripped_symbols = [".", ",", "?", "!", ";"]
        postprocessed = text[0]

        is_abbr = False

        for i in range(1, len(text)):
            # always just append the last symbol, as it is ]
            if i >= len(text) - 1:
                postprocessed += text[i]
                continue

            # do not strip dots for capital latters
            # e.g. D.C.
            if text[i - 1].isupper() and text[i] == ".":
                is_abbr = True
                postprocessed += text[i]
                continue

            # do not strip dots for capital latters
            # e.g. D. C . -> D.C.
            # NOTE: it should be "D.C ." to match the TOP format
            if is_abbr and text[i - 1] == "." and text[i] == " " and text[i + 1].isupper():
                continue

            # all abbreviations should be hadled upper
            is_abbr = False

            # strip punctuation
            if text[i - 1] != " " and text[i] in stripped_symbols:
                postprocessed += " " + text[i]
                continue

            if text[i - 1] in stripped_symbols and text[i] != " ":
                postprocessed += " " + text[i]
                continue

            # strip apostrophe for posessive nouns
            if text[i - 1] != " " and text[i : i + 2] == "'s":
                postprocessed += " " + text[i]
                continue

            # merge apostrophe with the next symbol
            # used when posessive noun is a slot value
            # e.g. "[SL:CONTACT Vlad] ' s"
            if text[i - 1] == "'" and text[i] == " " and text[i + 1] == "s":
                continue

            postprocessed += text[i]

        # time
        postprocessed = postprocessed.replace("a . m", "a.m")
        postprocessed = postprocessed.replace("p . m", "p.m")

        return postprocessed
