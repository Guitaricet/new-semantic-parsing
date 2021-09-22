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
"""The objects that allow for convenient storing and manipulation of dictionaries of tensors as torch buffers."""

import torch.nn as nn


class ParamsBufferHolder(nn.Module):
    """
    Stores named_params copies **as buffers** in a convenient way.

    Buffers behave just like the model parameters, but they are not updated during optimizer.step().

    Args:
        named_params: an iterator returning (name, tensor) or an equivalent dict
        buffer_name_prefix: prefix for the buffer name in .register_buffer

    Usage:
        ParamsBufferHolder(my_module.named_parameters())

    """

    def __init__(self, named_params, buffer_name_prefix=""):
        super().__init__()

        self.buffer_name_prefix = buffer_name_prefix
        self._param_names = set()
        self._buffer_names = set()

        named_params_iterator = named_params
        if isinstance(named_params, dict):
            named_params_iterator = named_params.items()

        for n, p in named_params_iterator:
            buffer_name = self._param_name_to_buffer_name(n)
            buffer_value = p.detach().clone()
            self.register_buffer(buffer_name, buffer_value)

            self._param_names.add(n)
            self._buffer_names.add(buffer_name)

    def __getitem__(self, item):
        return self.get(item)

    def __len__(self):
        return len(self._buffer_names)

    def __repr__(self):
        return f"ParamBufferHolder with parameters {self._param_names}"

    def __contains__(self, item):
        return item in self._param_names

    def items(self):
        for n in self._param_names:
            yield n, self.get(n)

    def get(self, param_name):
        """Returns a buffer that corresponds to the param_name

        Usage:
            self.initial_params = ParamsBufferHolder(self.named_parameters())

            for n, p in self.named_parameters():
                buffer = self.initial_params.get(n)
        """
        buffer_name = self._param_name_to_buffer_name(param_name)
        if buffer_name not in self._buffer_names:
            raise ValueError(f"Buffer corresponding to the name {param_name} is not found.")

        return self.__getattr__(buffer_name)

    def set(self, param_name, param_value):
        """Sets a buffer value corresponding to the param_name

        Usage:
            self.initial_params = ParamsBufferHolder(self.named_parameters())

            for n, p in self.named_parameters():
                self.initial_params.set(n, torch.zeros_like(p))
        """
        buffer_name = self._param_name_to_buffer_name(param_name)
        if buffer_name not in self._buffer_names:
            raise ValueError(f"Buffer corresponding to the name {param_name} is not found.")

        self.__setattr__(buffer_name, param_value)

    def _param_name_to_buffer_name(self, param_name):
        return self.buffer_name_prefix + param_name.replace(".", "_")
