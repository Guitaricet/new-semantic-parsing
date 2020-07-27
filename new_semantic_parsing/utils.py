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
import torch
import numpy as np
import transformers

from tqdm.auto import tqdm


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
    """Compute mask which ignores padding and special tokens"""
    mask = np.ones(len(input_ids))
    if input_ids[0] == tokenizer.cls_token_id:
        mask[0] = 0
    for i, token_id in enumerate(input_ids):
        if token_id in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            mask[i] = 0
    return mask


def get_lr(model: transformers.PreTrainedModel):
    """Get optimal learning rate according to the Scaling Laws.

    https://arxiv.org/abs/2001.08361
    lr ~= 0.003239 - 0.0001395 log(n_non_embedding_params)
    """

    if hasattr(model, "embeddings"):
        n_embedding_params = get_n_embed_params_trainable(model.embeddings)
    elif hasattr(model, "encoder") and hasattr(model, "decoder"):
        n_embed_encoder = get_n_embed_params_trainable(model.encoder.embeddings)
        n_embed_decoder = get_n_embed_params_trainable(model.decoder.embeddings)
        n_embedding_params = n_embed_encoder + n_embed_decoder
    else:
        raise ValueError(
            "Model object should have .embeddings or.encoder.embeddings and .decoder.embeddings"
        )

    n_non_embedding_params = model.num_parameters(only_trainable=True) - n_embedding_params
    return 0.003239 - 0.0001395 * np.log(n_non_embedding_params)


def get_n_embed_params_trainable(embeddings):
    """
    :param embeddings: BertEmbeddings
    :returns: number of trainable embedding parameters
    """
    n_params = 0

    tok = embeddings.word_embeddings
    if tok.training:
        n_params += tok.num_embeddings * tok.embedding_dim

    pos = embeddings.position_embeddings
    if pos.training:
        n_params += pos.num_embeddings * pos.embedding_dim

    typ = embeddings.token_type_embeddings
    if typ.training:
        n_params += typ.num_embeddings * typ.embedding_dim

    return n_params


def get_model_type(model_name):
    """Search for a largest substring from transformers.CONFIG_MAPPING"""
    candidate = ""

    for name in transformers.CONFIG_MAPPING:
        if name in model_name and len(name) > len(candidate):
            candidate = name

    if len(candidate) == 0:
        raise ValueError(f"{model_name} is not found in transformers.CONFIG_MAPPING")

    return candidate


def iterative_prediction(model, dataloader, schema_tokenizer, max_len, num_beams, device="cpu"):
    model = model.to(device)

    predictions_ids = []
    predictions_str = []
    text_tokenizer = schema_tokenizer.src_tokenizer

    for batch in tqdm(dataloader, desc="generation"):
        prediction_batch: torch.LongTensor = model.generate(
            input_ids=batch["input_ids"].to(device),
            pointer_mask=batch["pointer_mask"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_length=max_len,
            num_beams=num_beams,
            pad_token_id=text_tokenizer.pad_token_id,
            bos_token_id=schema_tokenizer.bos_token_id,
            eos_token_id=schema_tokenizer.eos_token_id,
        )

        for i, prediction in enumerate(prediction_batch):
            prediction = [
                p for p in prediction.cpu().numpy() if p not in schema_tokenizer.special_ids
            ]
            predictions_ids.append(prediction)

            prediction_str: str = schema_tokenizer.decode(
                prediction, batch["input_ids"][i], skip_special_tokens=True,
            )
            predictions_str.append(prediction_str)

    return predictions_ids, predictions_str


def make_subset(dataset, subset_size):
    """Make torch Subset by randomly sampling indices from dataset

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
    required_schema_vocab = set()
    required_example_ids = set()

    for i, row in train_data.iterrows():
        add_this = False
        for token in schema_vocab.difference(required_schema_vocab):
            if token in row.schema:
                add_this = True
                required_schema_vocab.add(token)

        if add_this:
            required_example_ids.add(i)

        if required_schema_vocab == schema_vocab:
            break
    else:
        raise RuntimeError("Full vocabulary was not found in the training set")

    return required_example_ids


def check_config(pointer_module, trainer, args, strict=False):
    """Check that both module and trainer comply with args"""
    _cfg = pointer_module.model.config
    if args.dropout is not None:
        assert _cfg.dropout == args.dropout
        assert _cfg.encoder.hidden_dropout_prob == args.dropout
        assert _cfg.decoder.hidden_dropout_prob == args.dropout
        assert _cfg.encoder.attention_probs_dropout_prob == args.dropout
        assert _cfg.decoder.attention_probs_dropout_prob == args.dropout

    if getattr(args, "move_norm", None) is not None or strict:
        assert _cfg.move_norm == args.move_norm

    if args.label_smoothing is not None:
        assert _cfg.label_smoothing == args.label_smoothing

    if args.weight_decay is not None and (trainer.optimizers is not None or strict):
        for param_group in trainer.optimizers[0].param_groups:
            if not param_group["use_weight_decay"]:
                continue
            assert param_group["weight_decay"] == args.weight_decay
