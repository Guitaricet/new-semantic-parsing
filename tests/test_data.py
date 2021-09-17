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
import random
import unittest

import torch

import new_semantic_parsing as nsp
from new_semantic_parsing import utils
from new_semantic_parsing.dataclasses import InputDataClass


class Seq2SeqDataCollatorDatasetTest(unittest.TestCase):
    def test_collate_batch_shapes(self):
        utils.set_seed(29)

        bs = 3
        e_pad = 0
        d_pad = 1
        d_mask = 2
        src_tensors = [
            torch.tensor([2, 5, 4, 4, 2]),
            torch.tensor([1, 8, 2, 8, 5, 4, 2, 2, 5, 7]),
            torch.tensor([4, 6, 4, 2, 1, 2]),
        ]
        tgt_tensors = [
            torch.tensor([6, 7, 8, 7, 2, 2, 4, 8, 5]),
            torch.tensor([5, 2, 2, 8, 7, 3, 5, 4, 2, 2, 1]),
            torch.tensor([8, 2, 2, 3, 5, 2, 2, 2, 3, 8, 4, 6, 7, 8]),
        ]
        tgt_masks = [(tgt_tensors[i] == d_mask).type(torch.FloatTensor) for i in range(3)]

        expected_input_ids = torch.tensor(
            [
                [2, 5, 4, 4, 2, 0, 0, 0, 0, 0],
                [1, 8, 2, 8, 5, 4, 2, 2, 5, 7],
                [4, 6, 4, 2, 1, 2, 0, 0, 0, 0],
            ]
        )
        expected_decoder_input_ids = torch.tensor(
            [
                [6, 7, 8, 7, 2, 2, 4, 8, 5, 1, 1, 1, 1, 1],
                [5, 2, 2, 8, 7, 3, 5, 4, 2, 2, 1, 1, 1, 1],
                [8, 2, 2, 3, 5, 2, 2, 2, 3, 8, 4, 6, 7, 8],
            ]
        )
        expected_decoder_pointer_mask = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )

        assert d_mask in tgt_tensors[0]

        examples = [
            InputDataClass(
                input_ids=src_tensors[i],
                decoder_input_ids=tgt_tensors[i],
                decoder_pointer_mask=tgt_masks[i],
                labels=tgt_tensors[i],
            )
            for i in range(bs)
        ]

        collator = nsp.data.Seq2SeqDataCollator(e_pad, d_pad)

        batch = collator.collate_batch(examples)

        self.assertEqual(batch["input_ids"].shape, (bs, 10))
        self.assertIsInstance(batch["input_ids"], torch.LongTensor)
        self.assertEqual(batch["input_ids"][0, -1], e_pad)
        self.assertTrue(torch.all(batch["input_ids"] == expected_input_ids))

        self.assertEqual(batch["decoder_input_ids"].shape, (bs, 14))
        self.assertIsInstance(batch["decoder_input_ids"], torch.LongTensor)
        self.assertEqual(batch["decoder_input_ids"][0, -1], d_pad)
        self.assertTrue(torch.all(batch["decoder_input_ids"] == expected_decoder_input_ids))

        self.assertEqual(batch["labels"].shape, (bs, 14))
        self.assertIsInstance(batch["labels"], torch.LongTensor)

        self.assertEqual(batch["decoder_pointer_mask"].shape, (bs, 14))
        self.assertIsInstance(batch["decoder_pointer_mask"], torch.FloatTensor)
        _mask = batch["decoder_pointer_mask"]
        self.assertTrue(((_mask == 0) | (_mask == 1)).all())
        self.assertTrue(torch.all(batch["decoder_pointer_mask"] == expected_decoder_pointer_mask))


class PointerDatasetTest(unittest.TestCase):
    def test_getitem(self):
        utils.set_seed(29)

        d_mask = 2
        src_tensors = [
            torch.tensor([2, 5, 4, 4, 2]),
            torch.tensor([1, 8, 2, 8, 5, 4, 2, 2, 5, 7]),
            torch.tensor([4, 6, 4, 2, 1, 2]),
        ]
        tgt_tensors = [
            torch.tensor([6, 7, 8, 7, 2, 2, 4, 8, 5]),
            torch.tensor([5, 2, 2, 8, 7, 3, 5, 4, 2, 2, 1]),
            torch.tensor([8, 2, 2, 3, 5, 2, 2, 2, 3, 8, 4, 6, 7, 8]),
        ]
        tgt_masks = [(tgt_tensors[i] == d_mask).type(torch.FloatTensor) for i in range(3)]

        expected_decoder_input_ids = [
            torch.tensor([6, 7, 8, 7, 2, 2, 4, 8]),
            torch.tensor([5, 2, 2, 8, 7, 3, 5, 4, 2, 2]),
            torch.tensor([8, 2, 2, 3, 5, 2, 2, 2, 3, 8, 4, 6, 7]),
        ]
        expected_decoder_pointer_mask = [
            torch.tensor([0, 0, 0, 0, 1, 1, 0, 0]),
            torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 1, 1]),
            torch.tensor([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]),
        ]

        dataset = nsp.data.PointerDataset(src_tensors, tgt_tensors, target_pointer_masks=tgt_masks)

        item: InputDataClass = dataset[0]

        self.assertIsInstance(item, InputDataClass)
        self.assertIsInstance(item.input_ids, torch.LongTensor)
        self.assertIsInstance(item.decoder_input_ids, torch.LongTensor)
        self.assertIsInstance(item.labels, torch.LongTensor)

        self.assertTrue(torch.all(item.input_ids == src_tensors[0]))
        self.assertTrue(torch.all(item.decoder_input_ids == expected_decoder_input_ids[0]))
        self.assertTrue(torch.all(item.decoder_pointer_mask == expected_decoder_pointer_mask[0]))

    def test_getitem_inference(self):
        utils.set_seed(29)

        src_tensors = [
            torch.randint(0, 10, size=(random.randint(5, 13),), dtype=torch.int64)
            for _ in range(10)
        ]

        dataset = nsp.data.PointerDataset(src_tensors)

        item = dataset[0]

        self.assertIsInstance(item, InputDataClass)
        self.assertIsInstance(item.input_ids, torch.LongTensor)

        self.assertIsNone(item.decoder_input_ids)
        self.assertIsNone(item.labels)

    def test_len(self):
        dataset = nsp.data.PointerDataset([None, None], [None, None], None)
        self.assertEqual(len(dataset), 2)

    def test_get_frequencies(self):
        vocab = ["3", "4", "5"]  # + PAD, BOS, EOS
        max_pointer_len = 10
        tokenizer = nsp.TopSchemaTokenizer(vocab, None, max_pointer_len=max_pointer_len)

        # The test relies on this
        # assert tokenizer.pad_token_id == 10
        # assert tokenizer.bos_token_id == 11
        # assert tokenizer.eos_token_id == 12

        # all numbers below max_pointer_len should be ignored
        tgt_tensors = [[10, 12, 13, 14, 6], [10, 1, 12, 13], [15, 0, 0], [13, 12]]
        expected_frequencies = {
            "3": 3 / 5.0,
            "4": 1 / 5.0,
            "5": 1 / 5.0,
        }
        dataset = nsp.PointerDataset(tgt_tensors, tgt_tensors)
        dataset.torchify()

        frequencies = dataset.get_class_frequencies(tokenizer)

        self.assertDictEqual(frequencies, expected_frequencies)
