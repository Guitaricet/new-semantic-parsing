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
"""Utils to get optimizer and scheduler.

Optimizers have different param gropus for encoder and decoder to support
gradual unfreezing and different learning rates.
"""

import math
from typing import List
from itertools import chain

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR


class KeyIndexableList:
    """This data structure looks like both list and dict.

    However, unlike Dict, it is not designed to be quickly mutable,
    which it ok for the case of optimizer where we use it.
    """
    def __init__(self, list_of_kv_tuples):
        self.list_of_kv_tuples = list_of_kv_tuples

        self._str_index = {k: i for i, (k, _) in enumerate(list_of_kv_tuples)}

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.get_index_by_key(item)

        k, v = self.list_of_kv_tuples[item]
        return v

    def __iter__(self):
        for _, v in self.list_of_kv_tuples:
            yield v

    def __len__(self):
        return len(self.list_of_kv_tuples)

    def get_index_by_key(self, key):
        return self._str_index[key]


def get_optimizers(model, learning_rate, weight_decay=0, adam_eps=1e-9, use_synaptic_intelligence=False):
    """Setups the optimizer and the learning rate scheduler.

    Creates optimizer which can update encoder and decoder with different learning rates
    and scheduler which increases lr for warmup_steps and does not update encoder
    for num_frozen_encoder_steps.

    Args:
        model: EncoderDecoderWPointerModel.
        learning_rate: either float or dict with keys 'encoder_lr' and 'decoder_lr'.
        weight_decay: optimizer weight_decay.
        adam_eps: ADAM epsilon value
        use_synaptic_intelligence:

    Returns:
        A tuple with two values: torch Optimizer and torch LambdaLR scheduler.
    """

    lr = learning_rate
    if isinstance(lr, float):
        encoder_lr = decoder_lr = lr
    elif isinstance(lr, dict):
        encoder_lr = lr.get("encoder_lr", 0)
        decoder_lr = lr.get("decoder_lr", 0)
    else:
        raise ValueError("learning_rate should be either float or dict")

    # decoder parameters include prediction head and pointer network
    # optionally, they also include the module which projects encoder representations
    # into decoder-sized dimension
    to_chain = [
        model.decoder.named_parameters(),
        model.lm_head.named_parameters(),
        model.decoder_q_proj.named_parameters(),
    ]
    if model.enc_dec_proj is not None:
        to_chain.append(model.enc_dec_proj.named_parameters())

    decoder_parameters = chain(*to_chain)

    no_decay = ["bias", "LayerNorm.weight"]
    # fmt: off
    optimizer_grouped_parameters = [
        {
            "params": KeyIndexableList(
                [(n, p) for n, p in decoder_parameters if not any(nd in n for nd in no_decay)]
            ),
            "initial_lr": decoder_lr,
            "lr": decoder_lr,
            "use_weight_decay": True,
            "weight_decay": weight_decay,
            "group_type": "decoder_params",
        },
        {
            "params": KeyIndexableList(
                [(n, p) for n, p in decoder_parameters if any(nd in n for nd in no_decay)]
            ),
            "initial_lr": decoder_lr,
            "lr": decoder_lr,
            "use_weight_decay": False,
            "weight_decay": 0.0,
            "group_type": "decoder_params_no_decay",
        },
    ]

    if encoder_lr > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": KeyIndexableList(
                    [(n, p) for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
                ),
                "initial_lr": encoder_lr,
                "lr": encoder_lr,
                "use_weight_decay": True,
                "weight_decay": weight_decay,
                "group_type": "encoder_params",
            },
            {
                "params": KeyIndexableList(
                    [(n, p) for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)]
                ),
                "initial_lr": encoder_lr,
                "lr": encoder_lr,
                "use_weight_decay": False,
                "weight_decay": 0.0,
                "group_type": "encoder_params_no_decay",
            },
        ])
    # fmt: on

    if not use_synaptic_intelligence:
        return torch.optim.Adam(optimizer_grouped_parameters, eps=adam_eps, betas=(0.9, 0.98))

    omega_list = [
        {
            "omega": KeyIndexableList(
                [(n, torch.zeros_like(p)) for n, p in decoder_parameters if not any(nd in n for nd in no_decay)]
            ),
            "group_type": "decoder_params",
        },
        {
            "omega": KeyIndexableList(
                [(n, torch.zeros_like(p)) for n, p in decoder_parameters if any(nd in n for nd in no_decay)]
            ),
            "group_type": "decoder_params_no_decay",
        },
    ]

    if encoder_lr > 0:
        omega_list.extend([
            {
                "omega": KeyIndexableList(
                    [(n, torch.zeros_like(p)) for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
                ),
                "group_type": "encoder_params",
            },
            {
                "omega": KeyIndexableList(
                    [(n, torch.zeros_like(p)) for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)]
                ),
                "group_type": "encoder_params_no_decay",
            },
        ])

    optimizer = AdamSI(optimizer_grouped_parameters, omega=omega_list, eps=adam_eps, betas=(0.9, 0.98))
    return optimizer


class NoamSchedulePolicy:
    def __init__(self, num_warmup_steps, model_size):
        self.num_warmup_steps = num_warmup_steps
        self.model_size = model_size

    def __call__(self, current_step):
        current_step = max(current_step, 1)
        _num_warmup_steps = max(self.num_warmup_steps, 1)

        scale = self.model_size ** -0.5 * min(
            current_step ** (-0.5), current_step * _num_warmup_steps ** (-1.5)
        )
        return scale


def get_noam_schedule(optimizer, num_warmup_steps, model_size, last_epoch=1):
    """Creates a Noam (inverse square root) scheduler with linear warmup and encoder gradual unfreezing.

    :param optimizer: torch Optimizer where some param groups have 'group_type' key
        if group_type starts with 'encoder_' it will be frozen for `num_frozen_encoder_steps`
    :param num_warmup_steps: number of steps for linear warmup from 0 to optimizer.lr
    :param model_size: hidden size of the model (d_model)
    :param last_epoch: The index of last epoch. Default: 1.

    :return: LambdaLR scheduler
    """
    policy = NoamSchedulePolicy(num_warmup_steps, model_size)
    return LambdaLR(optimizer, policy, last_epoch)


def set_encoder_requires_grad(param_groups, value: bool):
    for param_group in param_groups:
        group_type = param_group.get("group_type", "")
        if not group_type.startswith("encoder"):
            continue

        for param in param_group["params"]:
            if param.requires_grad is value:
                # if value has already been set
                return
            param.requires_grad = value


def set_weight_decay(param_groups, weight_decay):
    for param_group in param_groups:
        if param_group["use_weight_decay"]:
            param_group["weight_decay"] = weight_decay


class AdamSI(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        """ADAM optimizer that additionally tracks Synaptic Intelligence integral sum(weight update * grad)

        Args:
            params: same as ADAM first argument, but instead of iterable, params should be KeyIndexableList
            omega: a list of tensors representing SI integral for every parameter in `params`
                should follow the same order as tensors in params
        """
        if isinstance(params, KeyIndexableList):
            self.omega = KeyIndexableList([(k, torch.zeros_like(v)) for k, v in params.list_of_kv_tuples])
            self.omega_groups = [self.omega]

        elif isinstance(params, list) and isinstance(params[0], KeyIndexableList):
            self.omega = [
                {
                    "omega": KeyIndexableList([(k, v) for k, v in group_dict["params"]]),
                    "group_type": group_dict["group_type"],
                }
                for group_dict in params
            ]
            self.omega_groups = self.omega

        super().__init__(params, **kwargs)


    @torch.no_grad()
    def step(self, closure=None):
        """A copy of torch.optim.Adam.step, with some modifications to update self.omega

        All modifications are highlighted with a comment "Modification by <NAME>"
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, omega_group in zip(self.param_groups, self.omega_groups):  # Modification by Vlad Lialin
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self._step_fn(params_with_grad,  # Modification by Vlad Lialin
                   omega_group["omega"],  # Modification by Vlad Lialin
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps'],
                   )
        return loss

    def _step_fn(self,
                 params: List[Tensor],
                 omegas: KeyIndexableList,  # Modification by Vlad Lialin
                 grads: List[Tensor],
                 exp_avgs: List[Tensor],
                 exp_avg_sqs: List[Tensor],
                 max_exp_avg_sqs: List[Tensor],
                 state_steps: List[int],
                 amsgrad: bool,
                 beta1: float,
                 beta2: float,
                 lr: float,
                 weight_decay: float,
                 eps: float):
        """A copy of torch.optim.functional.adam with added omega update

        All modifications are highlighted with a comment "Modification by <NAME>"
        """

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

            omegas[i].addcdiv_(exp_avg * grad, denom, value=-step_size)  # Modification by Vlad Lialin
