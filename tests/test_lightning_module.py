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
import unittest

import torch
import transformers

from new_semantic_parsing import utils
from new_semantic_parsing.data import PointerDataset, Seq2SeqDataCollator
from new_semantic_parsing.lightning_module import PointerModule
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.modeling_encoder_decoder_wpointer import EncoderDecoderWPointerModel


class LightningModuleTest(unittest.TestCase):
    def setUp(self):
        utils.set_seed(3)
        src_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
        vocab = [
            "[",
            "]",
            "IN:",
            "SL:",
            "GET_DIRECTIONS",
            "DESTINATION",
            "DATE_TIME_DEPARTURE",
            "GET_ESTIMATED_ARRIVAL",
        ]
        self.schema_tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        self.model = EncoderDecoderWPointerModel.from_parameters(
            layers=2,
            hidden=32,
            heads=2,
            src_vocab_size=src_tokenizer.vocab_size,
            tgt_vocab_size=self.schema_tokenizer.vocab_size,
            max_src_len=17,
            dropout=0.1,
        )

        source_texts = [
            "Directions to Lowell",
            "Get directions to Mountain View",
        ]
        target_texts = [
            "[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]",
            "[IN:GET_DIRECTIONS Get directions to [SL:DESTINATION Mountain View]]",
        ]

        pairs = [
            self.schema_tokenizer.encode_pair(t, s) for t, s in zip(target_texts, source_texts)
        ]

        self.dataset = PointerDataset.from_pair_items(pairs)
        self.dataset.torchify()

        collator = Seq2SeqDataCollator(pad_id=self.schema_tokenizer.pad_token_id)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2, collate_fn=collator.collate_batch
        )

        self.test_batch = next(iter(dataloader))

        self.module = PointerModule(
            model=self.model,
            schema_tokenizer=self.schema_tokenizer,
            train_dataset=self.dataset,
            valid_dataset=self.dataset,
            lr=1e-3,
        )

    def test_training_step(self):
        out = self.module.training_step(batch=self.test_batch, batch_idx=0)

        loss = out["loss"]
        self.assertIsInstance(loss, torch.FloatTensor)

    def test_validation_step(self):
        self.module.eval()
        assert self.model.config.dropout > 0

        out1 = self.module.validation_step(batch=self.test_batch, batch_idx=0)
        out2 = self.module.validation_step(batch=self.test_batch, batch_idx=0)

        self.assertTrue(torch.allclose(out1["eval_exact_match"], out2["eval_exact_match"]))
        self.assertIsInstance(out1["eval_exact_match"], torch.FloatTensor)

    def test_validation_epoch_end(self):
        validation_step_outputs = 3 * [
            {
                "eval_loss": torch.tensor(1.0),
                "eval_accuracy": torch.tensor(0.7),
                "eval_exact_match": torch.tensor(0.01),
            }
        ]

        out = self.module.validation_epoch_end(validation_step_outputs)

        logs = out["log"]

        self.assertTrue(torch.isclose(logs["eval_loss"], torch.tensor(1.0)))
        self.assertTrue(torch.isclose(logs["eval_accuracy"], torch.tensor(0.7)))
        self.assertTrue(torch.isclose(logs["eval_exact_match"], torch.tensor(0.01)))

    def test_expand_output_vocab(self):
        src_vocab_size = 23
        tgt_vocab_size = 17
        max_src_len = 7
        expand_by = 3

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=max_src_len,
            dropout=0,
            track_grad_square=True,
        )

        model.expand_output_vocab(expand_by=expand_by, reuse_weights=True, init_type="zeros")

        self.assertTrue(model.output_vocab_size == tgt_vocab_size + expand_by)
        self.assertTrue(model.decoder.embeddings.word_embeddings.num_embeddings == tgt_vocab_size + max_src_len + expand_by)

        all_named_parameters = [k for k, v in model.named_parameters()]
        self.assertTrue("decoder.embeddings.word_embeddings.weight" in all_named_parameters)
        self.assertTrue("lm_head.decoder.weight" in all_named_parameters)
        self.assertTrue("lm_head.bias" in all_named_parameters)
        self.assertTrue("lm_head.decoder.bias" not in all_named_parameters)

        decoder_emb_matrix = model.decoder.embeddings.word_embeddings.weight
        self.assertTrue(not torch.allclose(decoder_emb_matrix[-1], torch.zeros(32)),
                        "the last embeddings should be equal to the pointer embeddings before .expand_output_dim")
        self.assertTrue(torch.allclose(decoder_emb_matrix[tgt_vocab_size + 1], torch.zeros(32)),
                        "vocabulary should be extended with zero embeddings")
