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
import random
import unittest
from pprint import pprint

import torch
import transformers
import numpy as np

from new_semantic_parsing import EncoderDecoderWPointerModel
from new_semantic_parsing.dataclasses import InputDataClass
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.utils import compute_metrics
from new_semantic_parsing.data import PaddedDataCollator


class EncoderDecoderWPointerTest(unittest.TestCase):
    def test_shape_on_random_data(self):
        torch.manual_seed(42)

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

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=23,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        x_enc = torch.randint(0, encoder_config.vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, decoder_config.vocab_size, size=(bs, tgt_len))

        out = model(input_ids=x_enc, decoder_input_ids=x_dec)

        # different encoders return different number of outputs
        # e.g. BERT returns two, but DistillBERT only one
        self.assertGreaterEqual(len(out), 4)

        combined_logits = out[0]
        expected_shape = (bs, tgt_len, decoder_config.vocab_size - encoder_config.vocab_size + src_len)
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

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        # e.g. '[CLS] Directions to Lowell [SEP]'
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2]])
        # e.g. '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]'
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_mask=mask)[0]

        expected_shape = (1, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_shape_on_real_data_batched(self):
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

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_mask=mask)[0]

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

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        loss = model(input_ids=src_seq,
                     decoder_input_ids=tgt_seq,
                     pointer_mask=mask,
                     labels=tgt_seq)[0]

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss, 0)


class ModelOverfitTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists('runs'):
            shutil.rmtree('runs')

    def test_overfit(self):
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        # NOTE: the test is still not deterministic
        # NOTE: very long test, takes about ~10 seconds

        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')

        vocab = {'[', ']', 'IN:', 'SL:', 'GET_DIRECTIONS', 'DESTINATION',
                 'DATE_TIME_DEPARTURE', 'GET_ESTIMATED_ARRIVAL'}
        schema_tokenizer = TopSchemaTokenizer(vocab, tokenizer)

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=3, hidden=128, heads=2,
            src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=schema_tokenizer.vocab_size
        )

        source_texts = [
            'Directions to Lowell',
            'Get directions to Mountain View',
        ]
        schema_texts = [
            '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]',
            '[IN:GET_DIRECTIONS Get directions to [SL:DESTINATION Mountain View]]'
        ]

        source_ids = tokenizer.batch_encode_plus(source_texts, pad_to_max_length=True)['input_ids']

        schema_batch = schema_tokenizer.batch_encode_plus(
            schema_texts, source_ids, pad_to_max_length=True, return_tensors='pt'
        )

        source_ids = torch.LongTensor(source_ids)
        source_ids_mask = ((source_ids != tokenizer.pad_token_id) &
                           (source_ids != tokenizer.cls_token_id) &
                           (source_ids != tokenizer.sep_token_id)).type(torch.FloatTensor)

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self): return 2

            def __getitem__(self, i):
                return InputDataClass(**{
                    'input_ids': source_ids[i],
                    'attention_mask': source_ids_mask[i],
                    'decoder_input_ids': schema_batch.input_ids[i],
                    'decoder_attention_mask': schema_batch.attention_mask[i],
                    'labels': schema_batch.input_ids[i],
                })

        train_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            num_train_epochs=55,
            seed=42,
        )

        trainer = transformers.Trainer(
            model,
            train_args,
            train_dataset=MockDataset(),
            data_collator=PaddedDataCollator(),
            eval_dataset=MockDataset(),
            compute_metrics=compute_metrics,
        )
        # a trick to reduce the amount of logging
        trainer.is_local_master = lambda: False

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        train_out = trainer.train()
        eval_out = trainer.evaluate()

        pprint('Training output')
        pprint(train_out)
        pprint('Evaluation output')
        pprint(eval_out)

        # accuracy should be 1.0 and eval loss should be around 0.9
        self.assertGreater(eval_out['eval_accuracy'], 0.99)


# this compute_metrics is not compatible with the new_semantic_parsing.Trainer and only used here
def compute_metrics(eval_prediction: transformers.EvalPrediction):
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    accuracy = np.mean(predictions.reshape(-1) == eval_prediction.label_ids.reshape(-1))
    exact_match = np.mean(np.all(predictions == eval_prediction.label_ids, axis=1))

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
    }
