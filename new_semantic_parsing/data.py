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
"""Elements of data feeding pipeline - torch Dataset and collator"""

from collections import Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from new_semantic_parsing.dataclasses import InputDataClass, List, Tensor, PairItem
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.utils import get_src_pointer_mask, make_subset


class PointerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_tensors,
        target_tensors=None,
        source_pointer_masks=None,
        target_pointer_masks=None,
    ):
        """Stores tensors and makes labels as shifted target_tensors.

        Args:
            source_tensors: list of tensors, input ids
            target_tensors: list of tensors, labels
            source_pointer_masks: list of tensors, mask for the pointer network
                does not allow to point to padding, cls and sep
            target_pointer_masks: list of tensors, mask showing pointer locations in labels
        """
        self.source_tensors = source_tensors
        self.target_tensors = target_tensors
        self.source_pointer_masks = source_pointer_masks
        self.target_pointer_masks = target_pointer_masks

        if source_tensors is None:
            raise ValueError("source_tensors cannot be None")

        self.torchified = all(
            map(
                self._check_torchified,
                [source_tensors, source_pointer_masks, target_tensors, target_pointer_masks,],
            )
        )

    @classmethod
    def from_pair_items(cls, items: List[PairItem]):
        source_tensors = []
        target_tensors = []
        source_pointer_masks = []
        target_pointer_masks = []

        for item in items:
            source_tensors.append(item.src_ids)
            target_tensors.append(item.tgt_ids)
            source_pointer_masks.append(item.src_pointer_mask)
            target_pointer_masks.append(item.tgt_pointer_mask)

        return cls(source_tensors, target_tensors, source_pointer_masks, target_pointer_masks)

    def __len__(self):
        return len(self.source_tensors)

    def __getitem__(self, item) -> InputDataClass:
        source_pointer_mask = None
        if self.source_pointer_masks is not None:
            source_pointer_mask = self.source_pointer_masks[item]

        if self.target_tensors is None:
            return InputDataClass(
                input_ids=self.source_tensors[item],
                pointer_mask=source_pointer_mask,
            )

        target_pointer_mask = None
        if self.target_pointer_masks is not None:
            target_pointer_mask = self.target_pointer_masks[item][:-1]

        return InputDataClass(
            input_ids=self.source_tensors[item],
            pointer_mask=source_pointer_mask,
            decoder_input_ids=self.target_tensors[item][:-1],
            decoder_pointer_mask=target_pointer_mask,
            labels=self.target_tensors[item][1:],
        )

    @staticmethod
    def _check_torchified(x: List[Tensor]):
        if x is None:
            return True
        return isinstance(x[0], torch.Tensor)

    def torchify(self):
        """Makes all tensors torch.Tensor"""
        if self.torchified:
            return

        self.source_tensors = [torch.LongTensor(t) for t in self.source_tensors]
        if self.source_pointer_masks is not None:
            self.source_pointer_masks = [torch.FloatTensor(t) for t in self.source_pointer_masks]

        if self.target_tensors is not None:
            self.target_tensors = [torch.LongTensor(t) for t in self.target_tensors]
        if self.target_pointer_masks is not None:
            self.target_pointer_masks = [torch.FloatTensor(t) for t in self.target_pointer_masks]

        self.torchified = True

    def get_max_len(self):
        """Gets maximum length of source sequences and target sequences in the dataset
        Returns a tuple (source_max_len, target_max_len)
        if target_tensors is None, target_max_len is also None
        """
        source_max_len = max(len(t) for t in self.source_tensors)
        if self.target_tensors is None:
            return source_max_len, None

        target_max_len = max(len(t) for t in self.target_tensors)
        return source_max_len, target_max_len

    def get_class_frequencies(self, schema_tokenizer: TopSchemaTokenizer):
        """Gets frequencies of each schema token"""
        if self.target_tensors is None:
            raise RuntimeError(".get_class_frequencies was called on the dataset with no labels")

        counts = Counter()
        for example in self.target_tensors:
            schema_tokens = [i for i in example.tolist() if schema_tokenizer.is_schema_token_id(i)]
            counts.update(schema_tokens)

        total_count = sum(counts.values())
        frequencies = {
            schema_tokenizer.index2token(i): count / total_count for i, count in counts.items()
        }

        return frequencies


class Seq2SeqDataCollator:
    """Pads tensors to the maximum length in batch.
    Length is different for encoder and decoder inputs.
    Also makes padding masks.

    Decoder inputs should have prefix `decoder_`
    `labels` considered a decoder field too
    All other tensors considered encoder inputs

    All values in the input DataClasses should be torch.Tensor or shape (seq_len, *)
    or None, None values are ignored

    All values corresponsing to the keys ending with `mask` are padded with zeroes
    """

    def __init__(self, pad_id, decoder_pad_id=None):
        self.encoder_pad_id = pad_id
        self.decoder_pad_id = decoder_pad_id or pad_id

    def collate_batch(self, examples):
        """
        :param examples: list of InputDataClass
        :return: dict with the InputDataClass fields
        """
        batch = dict()
        batch_size = len(examples)

        self._encoder_max_len = None
        self._decoder_max_len = None

        # iterate ofer the first example to get shapes
        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            is_decoder = self._is_decoder_field(k)

            maxlen = max(getattr(ex, k).shape[0] for ex in examples)
            self._shape_check(maxlen, is_decoder, k)

            batched_shape = (batch_size, maxlen, *v.shape[1:])
            batch[k] = torch.zeros(batched_shape, dtype=v.dtype, device=v.device)

            if k.endswith("mask"):
                continue

            batch[k].fill_(self.decoder_pad_id if is_decoder else self.encoder_pad_id)

        for i, example in enumerate(examples):
            for k, tensor in vars(example).items():
                if tensor is None:
                    continue
                batch[k][i, : len(tensor)] = tensor

            if example.attention_mask is None:
                batch["attention_mask"] = batch["input_ids"] != self.encoder_pad_id

            has_decoder_inputs = example.decoder_input_ids is not None
            has_decoder_mask = example.decoder_attention_mask is not None
            if has_decoder_inputs and not has_decoder_mask:
                batch["decoder_attention_mask"] = batch["decoder_input_ids"] != self.decoder_pad_id

        return batch

    @staticmethod
    def _is_decoder_field(field_name):
        return field_name.startswith("decoder_") or field_name == "labels"

    def _shape_check(self, maxlen, is_decoder, key):
        """Performs data shape validation"""
        if is_decoder:
            if self._decoder_max_len is not None and self._decoder_max_len != maxlen:
                raise ValueError(f"decoder input tensors have different lengths ({key})")
        else:
            if self._encoder_max_len is not None and self._encoder_max_len != maxlen:
                raise ValueError(f"encoder input tensors have different lengths({key})")


def make_dataset(filepath, schema_tokenizer: TopSchemaTokenizer):
    data = pd.read_table(filepath, names=["text", "tokens", "schema"])

    pairs = [
        schema_tokenizer.encode_pair(schema, text)
        for text, schema in tqdm(zip(data.tokens, data.schema), total=len(data.tokens))
    ]

    dataset = PointerDataset.from_pair_items(pairs)
    dataset.torchify()

    return dataset


def make_test_dataset(filepath, schema_tokenizer: TopSchemaTokenizer, max_len=None):
    data = pd.read_table(filepath, names=["text", "tokens", "schema"])

    text_ids: List[list] = [
        schema_tokenizer.encode_source(text) for text in tqdm(data.tokens, desc="tokenization")
    ]
    if max_len is not None:
        text_ids = [t[:max_len] for t in text_ids]

    text_pointer_masks: List[np.ndarray] = [
        get_src_pointer_mask(t, schema_tokenizer.src_tokenizer) for t in text_ids
    ]

    dataset = PointerDataset(source_tensors=text_ids, source_pointer_masks=text_pointer_masks)
    dataset.torchify()

    return dataset


class SampleConcatSubset(torch.utils.data.Dataset):
    def __init__(self, concat_dataset, sample_dataset, sample_probability):
        """Implements concatenation of concat_dataset and a subset of sample_dataset.

        The elements of the subset are sampled with sample_probability.

        Args:
            concat_dataset: torch Dataset
            sample_dataset: torch Dataset
            sample_probability: float, 0 < sample_probability < 1
        """

        self._concat_dataset = concat_dataset
        self._sample_dataset = sample_dataset
        self._sample_probability = sample_probability

        self._data = None
        self.resample()

    def __len__(self):
        if self._data is None:
            raise RuntimeError("Call .resample first")
        return len(self._data)

    def resample(self, sample_probability=None):
        """Resamples from sample_dataset, inplace

        Args:
            sample_probability: optional, float, 0 < sample_probability < 1,
                equal to self.sample_probability by default

        Returns:
            torch Dataset
        """
        sample_probability = sample_probability or self._sample_probability

        subset = make_subset(self._sample_dataset, sample_probability)
        self._data = torch.utils.data.ConcatDataset([self._concat_dataset, subset])
        return self._data

    def __getitem__(self, item):
        return self._data[item]
