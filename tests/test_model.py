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
import os
import shutil
import tempfile
import unittest
from copy import deepcopy

import torch
import transformers

from new_semantic_parsing import EncoderDecoderWPointerModel
from new_semantic_parsing.utils import set_seed


MODELS = ["bert-base-cased"]


class EncoderDecoderWPointerTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_shape_on_random_data(self):
        set_seed(42)

        bs = 3
        src_len = 5
        tgt_len = 7

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=17,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        # decoder accepts vocabulary of schema vocab + pointer embeddings
        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=23,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        # logits are projected into schema vocab and combined with pointer scores
        max_pointer = src_len + 3
        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_pointer
        )

        x_enc = torch.randint(0, encoder_config.vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, decoder_config.vocab_size, size=(bs, tgt_len))

        out = model(input_ids=x_enc, decoder_input_ids=x_dec)

        # different encoders return different number of outputs
        # e.g. BERT returns two, but DistillBERT only one
        self.assertGreaterEqual(len(out), 4)

        schema_vocab = decoder_config.vocab_size - max_pointer

        combined_logits = out[0]
        expected_shape = (bs, tgt_len, schema_vocab + src_len)
        self.assertEqual(combined_logits.shape, expected_shape)

        decoder_hidden = out[1]
        expected_shape = (bs, tgt_len, decoder_config.hidden_size)
        self.assertEqual(decoder_hidden.shape, expected_shape)

        combined_logits = out[2]
        expected_shape = (bs, decoder_config.hidden_size)
        self.assertEqual(combined_logits.shape, expected_shape)

        encoder_hidden = out[3]
        expected_shape = (bs, src_len, encoder_config.hidden_size)
        self.assertEqual(encoder_hidden.shape, expected_shape)

    def test_shape_on_real_data(self):
        set_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23
        max_position = 5

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_position
        )

        # similar to real data
        # e.g. '[CLS] Directions to Lowell [SEP]'
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2]])
        # e.g. '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]'
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq, decoder_input_ids=tgt_seq, pointer_mask=mask)[0]

        expected_shape = (1, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_shape_on_real_data_batched(self):
        set_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23
        max_position = 7

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_position
        )

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0], [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor(
            [
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7, 0, 0],
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7],
            ]
        )
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq, decoder_input_ids=tgt_seq, pointer_mask=mask)[0]

        expected_shape = (2, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_loss_computation(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        max_position = 7
        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder=encoder, decoder=decoder, max_src_len=7)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0], [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor(
            [
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7, 0, 0],
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7],
            ]
        )
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]])

        loss = model(
            input_ids=src_seq,
            decoder_input_ids=tgt_seq,
            pointer_mask=mask,
            labels=tgt_seq,
        )[0]

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss, 0)

    def test_save_load(self):
        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
        )

        input_ids = torch.randint(src_vocab_size, size=(3, 7))
        tgt_sequence = torch.randint(tgt_vocab_size, size=(3, 11))
        decoder_input_ids = tgt_sequence[:, :-1].contiguous()
        labels = tgt_sequence[:, 1:].contiguous()

        expected_output = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels
        )

        os.mkdir(self.output_dir)
        model.save_pretrained(self.output_dir)

        loaded_model = EncoderDecoderWPointerModel.from_pretrained(self.output_dir)
        self.assertDictEqual(model.config.to_dict(), loaded_model.config.to_dict())
        for i, (p1, p2) in enumerate(zip(model.parameters(), loaded_model.parameters())):
            self.assertTrue(torch.allclose(p1, p2))

        output = loaded_model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels
        )

        self.assertEqual(len(output), len(expected_output))
        self.assertTrue(torch.allclose(expected_output[0], output[0]))  # loss
        self.assertTrue(torch.allclose(expected_output[1], output[1]))  # logits

    def test_freeze(self):
        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
        )

        # check that all parameters are trainable
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, msg=name)

        model.freeze_encoder()
        model.freeze_decoder()
        model.freeze_head()

        # check that all parameters are frozen
        for name, param in model.named_parameters():
            self.assertFalse(param.requires_grad, msg=name)

        model.freeze_encoder(freeze=False)
        model.freeze_decoder(freeze=False)
        model.freeze_head(freeze=False)

        # check that all parameters are trainable again
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, msg=name)

        # check that initial optimizer state does not interfere with the freezing
        bs, src_len, tgt_len = 3, 5, 7
        x_enc = torch.randint(0, src_vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, tgt_vocab_size, size=(bs, tgt_len))

        dec_inp = x_dec[:, :-1].contiguous()
        labels = x_dec[:, 1:].contiguous()

        for opt_class in [torch.optim.SGD, torch.optim.Adam]:
            with self.subTest(repr(opt_class)):
                optimizer = opt_class(model.parameters(), lr=1e-3)

                for _ in range(5):
                    out = model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)

                    loss = out[0]
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                model.freeze_encoder(freeze=True)
                model.freeze_decoder(freeze=True)

                model_copy = deepcopy(model)

                # do multiple optimizer updates to ensure that ADAM betas do not interfere with the freezing
                for _ in range(5):
                    optimizer.zero_grad()

                    out = model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)

                    loss = out[0]
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                for (n1, p1), (n2, p2) in zip(
                    model.encoder.named_parameters(), model_copy.encoder.named_parameters()
                ):
                    assert n1 == n2
                    self.assertFalse(p1.requires_grad)
                    self.assertTrue(torch.allclose(p1, p2), msg=f"Optimizer state changed {n1}")

                for (n1, p1), (n2, p2) in zip(
                    model.decoder.named_parameters(), model_copy.decoder.named_parameters()
                ):
                    assert n1 == n2
                    self.assertFalse(p1.requires_grad)
                    self.assertTrue(torch.allclose(p1, p2), msg=f"Optimizer state changed {n1}")

    def test_move_norm(self):
        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
            move_norm=0.1,
        )

        self.assertTrue(model.initial_params is not None)
        # check that model parameters do not include initial_params
        self.assertEqual(len(list(model.parameters())), len(model.initial_params))

        # check that model updates do not change initial_params
        bs, src_len, tgt_len = 3, 5, 7
        x_enc = torch.randint(0, src_vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, tgt_vocab_size, size=(bs, tgt_len))

        dec_inp = x_dec[:, :-1].contiguous()
        labels = x_dec[:, 1:].contiguous()

        optimizer = torch.optim.SGD(model.parameters(), 1e-3)

        out = model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)

        loss = out[0]
        loss.backward()

        optimizer.step()

        for n, p1 in model.named_parameters():
            if "pooler" in n:
                # we do not use pooler weights
                continue

            p2 = model.initial_params[n]
            self.assertTrue(torch.any(p2 != p1), msg=n)

        # check norm computation
        norm = model.get_move_norm()
        self.assertGreater(norm, 0)

    def test_move_norm_update(self):
        """Test that move norm affects optimization"""

        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
            move_norm=100,
        )

        model_copy = deepcopy(model)
        model_copy.config.move_norm = None
        del model_copy.initial_params

        model_copy2 = deepcopy(model)
        model_copy2.config.move_norm = None
        del model_copy2.initial_params

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_copy.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

        # check that model updates do not change initial_params
        bs, src_len, tgt_len = 3, 5, 7
        x_enc = torch.randint(0, src_vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, tgt_vocab_size, size=(bs, tgt_len))

        dec_inp = x_dec[:, :-1].contiguous()
        labels = x_dec[:, 1:].contiguous()

        losses = []
        for _model in [model, model_copy, model_copy2]:
            for _ in range(2):
                # at the first update move_norm = 0 as model == initial
                optimizer = torch.optim.SGD(_model.parameters(), 1e-3)

                out = _model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)

                loss = out[0]
                loss.backward()

                optimizer.step()

            losses.append(loss.detach())

        self.assertTrue(torch.allclose(losses[1], losses[2]), msg="test is not deterministic")
        self.assertFalse(torch.allclose(losses[0], losses[1]))

        for (n1, p1), (n2, p2), (n3, p3) in zip(
            model.named_parameters(), model_copy.named_parameters(), model_copy2.named_parameters()
        ):
            assert n1 == n2 == n3
            if "pooler" in n1:
                # we do not use pooler weights
                continue

            self.assertTrue(torch.allclose(p2, p3), msg=f"test is not deterministic")
            self.assertFalse(torch.allclose(p1, p2), msg=n1)

    def test_get_weight_consolidation(self):
        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
            track_grad_square=True,
        )

        bs, src_len, tgt_len = 3, 5, 7
        x_enc = torch.randint(0, src_vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, tgt_vocab_size, size=(bs, tgt_len))

        dec_inp = x_dec[:, :-1].contiguous()
        labels = x_dec[:, 1:].contiguous()

        optimizer = torch.optim.Adam(model.parameters())

        for _ in range(3):
            optimizer.zero_grad()

            out = model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)
            loss = out[0]

            loss.backward()
            optimizer.step()

            model.update_grad_squared()

        state_dict = model.state_dict()

        new_model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
            weight_consolidation=100,
        )

        new_model.load_state_dict(state_dict)

        reg = new_model._get_weight_consolidation()

        # before finetuning reg is 0
        self.assertIsNotNone(reg)
        self.assertFalse(torch.isinf(reg))
        self.assertTrue(reg == 0)

        optimizer = torch.optim.Adam(new_model.parameters())

        for _ in range(3):
            optimizer.zero_grad()

            out = new_model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)
            loss = out[0]

            loss.backward()
            optimizer.step()

        reg = new_model._get_weight_consolidation()
        self.assertTrue(reg > 0)

    def test_expand_out_and_move_norm_update(self):
        """Test that move norm affects optimization"""

        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            dropout=0,
            move_norm=100,
        )

        model.reset_initial_params()
        model.expand_output_vocab(2)

        model_copy = deepcopy(model)
        model_copy.config.move_norm = None
        del model_copy.initial_params

        model_copy2 = deepcopy(model)
        model_copy2.config.move_norm = None
        del model_copy2.initial_params

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_copy.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

        # check that model updates do not change initial_params
        bs, src_len, tgt_len = 3, 5, 7
        x_enc = torch.randint(0, src_vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, tgt_vocab_size, size=(bs, tgt_len))

        dec_inp = x_dec[:, :-1].contiguous()
        labels = x_dec[:, 1:].contiguous()

        losses = []
        for i, _model in enumerate([model, model_copy, model_copy2]):
            print(i)
            for _ in range(2):
                # at the first update move_norm = 0 as model == initial
                optimizer = torch.optim.SGD(_model.parameters(), 1e-3)

                out = _model(input_ids=x_enc, decoder_input_ids=dec_inp, labels=labels)

                loss = out[0]
                loss.backward()

                optimizer.step()

            losses.append(loss.detach())

        self.assertTrue(torch.allclose(losses[1], losses[2]), msg="test is not deterministic")
        self.assertFalse(torch.allclose(losses[0], losses[1]))

        for (n1, p1), (n2, p2), (n3, p3) in zip(
            model.named_parameters(), model_copy.named_parameters(), model_copy2.named_parameters()
        ):
            # model weights should change
            # model copy weights should stay the same
            assert n1 == n2 == n3
            if "pooler" in n1:
                # we do not use pooler weights
                continue

            self.assertTrue(torch.allclose(p2, p3), msg=f"test is not deterministic")
            if n1 == "lm_head.bias": continue
            self.assertFalse(torch.allclose(p1, p2), msg=n1)
