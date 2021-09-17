# Provided under the CC-BY-SA license. Please cite the accompanying paper when using TOP dataset -
# @ARTICLE {
#     author  = "Sonal Gupta and Rushin Shah and Mrinal Mohit and Anuj Kumar and Michael Lewis",
#     title   = "Semantic Parsing for Task Oriented Dialog using Hierarchical Representations",
#     journal = "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
#     year    = "2018",
#     month   = "Oct"
# }
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import transformers
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer


class TransformersTokenizerMock:
    cls_token = "[CLS]"
    cls_token_id = 101

    def encode(self, x, add_special_tokens=False):
        subtokens = x.split(",")
        return [int(t[3:]) for t in subtokens]

    def decode(self, x):
        return " ".join([f"tok{i}" for i in x])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def save_pretrained(self, path):
        pass


class TopSchemaTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.tmpdirname):
            shutil.rmtree(self.tmpdirname)

    def test_tokenize(self):
        # Test cases are examples from TOP dataset arxiv.org/abs/1810.07942
        schema_str = "[IN:INTENT1 tok1 tok2 tok3 [SL:SLOT1 tok4 tok5 ] ]"
        schema_tok = "[ IN: INTENT1 tok1 tok2 tok3 [ SL: SLOT1 tok4 tok5 ] ]".split(" ")

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = (
            "[IN:GET_EVENT Any [SL:CATEGORY_EVENT festivals ] [SL:DATE_TIME this weekend ] ]"
        )

        schema_tok = (
            "[ IN: GET_EVENT Any [ SL: CATEGORY_EVENT festivals ] [ SL: DATE_TIME this weekend ] ]"
        ).split(" ")

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = (
            "[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
            "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
            "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
            "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]"
        )

        schema_tok = (
            "[ IN: GET_ESTIMATED_ARRIVAL What time will I arrive at "
            "[ SL: DESTINATION [ IN: GET_LOCATION_HOME [ SL: CONTACT_RELATED "
            "my ] [ SL: TYPE_RELATION Mom ] 's house ] ] if I leave "
            "[ SL: DATE_TIME_DEPARTURE in five minutes ] ? ]"
        )
        schema_tok = schema_tok.split(" ")

        tokens = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(tokens, schema_tok)

    def test_encode_nocls(self):
        vocab = {"[", "]", "IN:", "INTENT1", "SL:", "SLOT1"}
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        schema_str = "[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]"
        source_tokens = [6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # note that the vocabulary is sorted
        # fmt: off
        expected_ids = [tokenizer.bos_token_id, 7, 3, 4, 9, 10, 11, 7, 5, 6, 12, 13, 8, 8, tokenizer.eos_token_id]
        # fmt: on
        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_encode_cls(self):
        vocab = ["[", "]", "IN:", "INTENT1", "SL:", "SLOT1"]
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        schema_str = "[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]"
        source_tokens = [TransformersTokenizerMock.cls_token_id, 6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # fmt: off
        expected_ids = [tokenizer.bos_token_id, 7, 3, 4, 10, 11, 12, 7, 5, 6, 13, 14, 8, 8, tokenizer.eos_token_id]
        # fmt: on
        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_keywords_in_text(self):
        vocab = ["[", "]", "IN:", "INTENT1", "SL:", "SLOT1", "SLT1"]
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        # i.e. SLOT1 after tok2 is just a token which is written exactly like a schema word
        schema_str = "[IN:INTENT1 tok6 tok2 SLT1 tok31 [SL:SLOT1 tok42 tok5 ] ]"
        source_tokens = [6, 2, 1, 31, 42, 5]
        # fmt: off
        expected_ids = [tokenizer.bos_token_id, 8, 3, 4, 10, 11, 12, 13, 8, 5, 6, 14, 15, 9, 9, tokenizer.eos_token_id]
        # fmt: on
        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_save_load(self):
        vocab = ["[", "]", "IN:", "INTENT1", "SL:", "SLOT1"]
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        schema_str = "[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]"
        source_tokens = [6, 2, 31, 42, 5]

        ids = tokenizer.encode(schema_str, source_tokens)

        tokenizer.save(self.tmpdirname, encoder_model_type="test_type")

        patch_tok_load = patch(
            "new_semantic_parsing.schema_tokenizer.transformers.AutoTokenizer.from_pretrained",
            MagicMock(return_value=TransformersTokenizerMock()),
        )
        patch_config_load = patch(
            "new_semantic_parsing.schema_tokenizer.transformers.AutoConfig.from_pretrained",
            MagicMock(return_value=None),
        )
        with patch_tok_load, patch_config_load:
            loaded_tokenizer = TopSchemaTokenizer.load(self.tmpdirname)

        self.assertSetEqual(set(loaded_tokenizer.vocab), set(tokenizer.vocab))

        new_ids = loaded_tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, new_ids)

    def test_decode(self):
        vocab = {"[", "]", "IN:", "INTENT1", "SL:", "SLOT1"}
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        schema_str = "[IN:INTENT1 tok6 tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]"
        source_tokens = [6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # note that the vocabulary is sorted
        expected_ids = [7, 3, 4, 9, 10, 11, 7, 5, 6, 12, 13, 8, 8]

        schema_decoded = tokenizer.decode(expected_ids, source_tokens)

        self.assertEqual(schema_str, schema_decoded)

    def test_decode_wpointers(self):
        vocab = {"[", "]", "IN:", "INTENT1", "SL:", "SLOT1"}
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer, max_pointer_len=17)

        schema_str = "[IN:INTENT1 @ptr0 @ptr1 @ptr2 [SL:SLOT1 @ptr3 @ptr4 ] ]"
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # note that the vocabulary is sorted
        ids = [7, 3, 4, 9, 10, 11, 7, 5, 6, 12, 13, 8, 8]

        schema_decoded = tokenizer.decode(ids)

        self.assertEqual(schema_str, schema_decoded)

    def test_postprocess_punct(self):
        text = "[What is this?]"
        expected = "[What is this ?]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "[This is nothing ! ]"
        expected = "[This is nothing ! ]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "7;45"
        expected = "7 ; 45"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

    def test_postprocess_apostrophe(self):
        text = "[What's]"
        expected = "[What 's]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "[I didn't do this.]"
        expected = "[I didn't do this .]"
        postprocessed = TopSchemaTokenizer.postprocess(text)

        self.assertSequenceEqual(expected, postprocessed)

        text = "[[Your ] ' s]"
        expected = "[[Your ] 's]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)
