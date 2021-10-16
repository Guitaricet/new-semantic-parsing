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
"""Transformers Config to store configuration and handle model loading from checkpoint."""


from transformers import EncoderDecoderConfig


class EncoderDecoderWPointerConfig(EncoderDecoderConfig):
    """Class to store the configuration of a `EncoderDecoderWPointerModel".

    Transformers use config objects to get the hyperparameters for model loading.
    """

    model_type = "encoder_decoder_wpointer"

    def __init__(self, max_src_len, dropout=None, model_args=None, **kwargs):
        """
        Args:
            max_src_len: int, maximum source sequence length in BPE tokens
            dropout: float, dropout prob for head and pointer network,
                to change dropout values for the encoder and decoder, modify encoder and decoder configs
            model_args: argparse args (probably, from tre train script)
        """
        super().__init__(**kwargs)

        self.max_src_len = max_src_len

        self.move_norm = kwargs.get("move_norm", None)
        self.move_norm_p = kwargs.get("move_norm_p", 2)
        self.use_pointer_bias = kwargs.get("use_pointer_bias", False)
        self.label_smoothing = kwargs.get("label_smoothing", 0)
        self.weight_consolidation = kwargs.get("weight_consolidation", None)
        self.max_param_importance = kwargs.get("max_param_importance", None)

        if model_args is not None:
            self.move_norm = getattr(model_args, "move_norm", None)
            self.move_norm_p = getattr(model_args, "move_norm_p", 2)
            self.use_pointer_bias = getattr(model_args, "use_pointer_bias", False)
            self.label_smoothing = getattr(model_args, "label_smoothing", 0)
            self.weight_consolidation = getattr(model_args, "weight_consolidation", None)
            self.max_param_importance = getattr(model_args, "max_param_importance", None)

        self.model_type = self.model_type
        self.dropout = dropout or 0

    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config, decoder_config, max_src_len, model_args):
        return cls(
            encoder=encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            max_src_len=max_src_len,
            model_args=model_args,
        )

    def set_dropout(self, dropout_prob):
        self.dropout = dropout_prob
        self.encoder.hidden_dropout_prob = dropout_prob
        self.encoder.attention_probs_dropout_prob = dropout_prob
        self.decoder.hidden_dropout_prob = dropout_prob
        self.decoder.attention_probs_dropout_prob = dropout_prob
