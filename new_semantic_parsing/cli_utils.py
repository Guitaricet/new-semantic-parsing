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
"""Utils only used in cli scripts"""
import os
import sys
import logging

import toml
import torch

import numpy as np

import new_semantic_parsing as nsp
import new_semantic_parsing.metrics
import new_semantic_parsing.utils

from new_semantic_parsing import config


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("wandb.sdk.internal.internal").setLevel(logging.WARNING)


def evaluate_model_n_rounds(
    model: nsp.EncoderDecoderWPointerModel,
    schema_tokenizer: nsp.TopSchemaTokenizer,
    eval_dataloader,
    prefix,
    max_len,
    n_rounds=5,
    num_beams=1,
):
    """Computes metrics for each class n_rounds times and average results

    To reduce the amount of noise in the metrics,
    in every round evaluate on a random subset of eval_dataset.
    Final results contain both mean and standard deviation.

    Args:
        checkpoint_path: str, path to the pytorch lightning checkpoint
        schema_tokenizer: TopSchemaTokenizer
        eval_dataloader: torch Dataloader to evaluate on
        prefix: str, prefix for all metrics
        n_rounds: int, number of times to evaluate on eval_dataset subset
        subset_size: float, size of the subset for every round

    Returns:
        tuple (final_metrics, description)
            final_metrics is a dict with keys means, stdevs
            description is a string describing main metircs
    """

    model.eval()

    _, pred_tokens = iterative_prediction(
        model,
        eval_dataloader,
        schema_tokenizer,
        max_len=max_len,
        num_beams=num_beams,
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_tokens=True,
    )

    true_tokens = _get_true_tokens_from_dataset(eval_dataloader.dataset, schema_tokenizer)

    all_final_metrics = []
    folds = _get_kfold_subsets(pred_tokens, true_tokens, n_rounds)

    for fold, (predictions_subset, labels_subset) in enumerate(folds):
        logger.info(f"Computing metrics for the fold {fold}")
        _final_metrics = nsp.metrics.get_metrics(
            predictions_subset,
            labels_subset,
            monitor_classes=schema_tokenizer.vocab,
            compute_macro=True,
            prefix=prefix,
            schema_tokenizer=schema_tokenizer,
            do_each=True,
            verbose=False,
        )
        all_final_metrics.append(_final_metrics)

    metrics_statistic = _get_metrics_staistic(all_final_metrics)
    description = _get_final_metrics_description(metrics_statistic)
    return metrics_statistic, description


def _get_true_tokens_from_dataset(dataset, schema_tokenizer):
    true_ids = dataset.target_tensors
    true_tokens = [
        schema_tokenizer.decode(t, source_ids=s, return_tokens=True, skip_special_tokens=True)
        for t, s in zip(true_ids, dataset.source_tensors)
    ]
    return true_tokens


def _get_kfold_subsets(pred_tokens, true_tokens, k_folds):
    """Gets cross-validation like splits.

    Returns:
        generator with k_fold elements,
        each element is a tuple of (pred_tokens_subset, true_tokens_subset)
    """

    n_examples = len(pred_tokens)
    permutation = np.random.permutation(n_examples)

    fold_size = n_examples // k_folds
    fold_edges = [fold_size * i for i in range(1, k_folds)]

    folds = np.split(permutation, fold_edges)

    assert len(folds) == k_folds

    for fold_idx in range(k_folds):
        subset_ids = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])

        _pred_tokens = [pred_tokens[i] for i in subset_ids]
        _true_tokens = [true_tokens[i] for i in subset_ids]

        yield _pred_tokens, _true_tokens


def _get_metrics_staistic(metrics):
    metrics_statistics = {
        "means": {},
        "stdevs": {},
    }

    metric_names = metrics[0].keys()
    for metric_name in metric_names:
        metric = [m[metric_name] for m in metrics]
        metrics_statistics["means"][metric_name] = np.mean(metric)
        metrics_statistics["stdevs"][metric_name + "_std"] = np.std(metric)

    return metrics_statistics


def _get_final_metrics_description(final_metrics):
    metric_names = final_metrics["means"].keys()

    description = "\n"
    max_name_len = max(map(len, metric_names))
    for k in metric_names:
        if not k.endswith("_f1"):
            continue
        mean = round(float(final_metrics["means"][k]), 3)
        stdev = round(float(final_metrics["stdevs"][k + "_std"]), 3)

        space = " " * (max_name_len - len(k))
        description += f"{k} {space}: {mean} +- {2 * stdev}\n"

    return description


def check_config(pointer_module, trainer, args, strict=False):
    """Checks that both module and trainer comply with args"""
    _cfg = pointer_module.model.config
    if args.dropout is not None:
        assert _cfg.dropout == args.dropout
        assert _cfg.encoder.hidden_dropout_prob == args.dropout
        assert _cfg.decoder.hidden_dropout_prob == args.dropout
        assert _cfg.encoder.attention_probs_dropout_prob == args.dropout
        assert _cfg.decoder.attention_probs_dropout_prob == args.dropout

    if getattr(args, "move_norm", None) is not None or strict:
        assert _cfg.move_norm == args.move_norm
        assert _cfg.move_norm_p == args.move_norm_p

    if args.label_smoothing is not None:
        assert _cfg.label_smoothing == args.label_smoothing

    if args.weight_decay is not None and (trainer.optimizer is not None or strict):
        for param_group in trainer.optimizers[0].param_groups:
            if not param_group["use_weight_decay"]:
                continue
            assert param_group["weight_decay"] == args.weight_decay


def iterative_prediction(
    model: nsp.EncoderDecoderWPointerModel,
    dataloader,
    schema_tokenizer: nsp.TopSchemaTokenizer,
    max_len,
    num_beams,
    device="cpu",
    return_tokens=False,
):
    """Executes inference-time prediction loop.

    Returns:
        A tuple of two elements (predictions_ids, predictions_str)
            predictions_ids: list of np.arrays
            predictions_str: list of strings if return_tokens is False
                or list of lists of strings if return_tokens is True
    """
    model = model.to(device)

    predictions_ids = []
    predictions_str = []
    text_tokenizer = schema_tokenizer.src_tokenizer

    assert dataloader.num_workers == config.NUM_WORKERS

    logger.info("Generating predictions sequentially")
    for batch in dataloader:
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
                prediction,
                batch["input_ids"][i],
                skip_special_tokens=True,
                return_tokens=return_tokens,
            )
            predictions_str.append(prediction_str)

        logger.info(f"Generation example: \n{predictions_str[0]}")

    return predictions_ids, predictions_str


def get_outliers(pretrain_metrics, final_metrics, filter_by_name=None, sigma=1.0):
    """Gets the names of metrics that are worse and better than the initial value with a high probability (0.97)

    97% confidence comes from the probability that both metrics will exceed/fall behind their standard deviation
    e.g. m1 > m1 + std1 with prob 1/6 and m2 < m2 - std2 with prob 1/6, then m2 < m1 with prob 1/6**2 ~= 0.03

    Args:
        pretrain_metrics: dict with keys "means" and "stdevs",
            "means" value is a subdictionary with metric names as keys,
            "stdevs" value is a subdictionary with metric_name + "_std" as keys
        final_metrics: dict with the structure as initial_metrics
        filter_by_name: regex pattern that metric name needs to match
        sigma: value that controls how far the outliers should be,
            if zero, any changed value will be an outlier
    """
    negative_outliers, positive_outliers = [], []

    for metric_name, pretrain_mean in pretrain_metrics["means"].items():
        if not nsp.utils.matches_pattern(metric_name, filter_by_name):
            continue

        pretrain_std = pretrain_metrics["stdevs"][metric_name + "_std"]
        final_mean = final_metrics["means"][metric_name]
        final_std = final_metrics["stdevs"][metric_name + "_std"]

        if pretrain_mean - final_mean > sigma * (pretrain_std + final_std):
            negative_outliers.append(metric_name)

        if pretrain_mean - final_mean < -sigma * (pretrain_std + final_std):
            positive_outliers.append(metric_name)

    return negative_outliers, positive_outliers


def load_saved_args(path):
    with open(path) as f:
        train_args = toml.load(f)
        if train_args["version"] != nsp.SAVE_FORMAT_VERSION:
            logger.warning(
                "Binary data version differs from the current version. "
                "This may cause failing and unexpected behavior."
            )

        if "metrics" in train_args:
            # for some reason means and stdevs are saved as strings
            train_args["metrics"]["means"] = {
                k: float(v) for k, v in train_args["metrics"]["means"].items()
            }
            train_args["metrics"]["stdevs"] = {
                k: float(v) for k, v in train_args["metrics"]["stdevs"].items()
            }

    return train_args


def evaluate_finetuning_procedure(pretrain_metrics, final_metrics, metric_weights, sigma=1.0):
    """Identifies the classes that degraded for sure and improved for sure, compute RI and RD.

    RI and RD

    Args:
        pretrain_metrics: dict metric_name:metric_value
        final_metrics: dict metric_name:metric_value
        metric_weights: dict metric_name:metric_weight

    Returns:
        dict with keys
            absolute_improvement
            relative_improvement
            absolute_degradation
            relative_degradation
            n_positive_outliers
            n_negative_outliers
            positive_outliers
            negative_outliers
            delta/cls/{class_name}_tree_path_f1
    """

    metric_pattern = "cls/.*_tree_path_f1$"
    deltas = get_metrics_delta(pretrain_metrics, final_metrics, metric_pattern)

    negative_outliers, positive_outliers = get_outliers(
        pretrain_metrics,
        final_metrics,
        metric_pattern,
        sigma=sigma,
    )

    default_metrics = {
        "absolute_improvement": 0,
        "relative_improvement": 0,
        "absolute_degradation": 0,
        "relative_degradation": 0,
    }
    positive_outliers_metrics, negative_outliers_metrics = default_metrics, default_metrics

    if len(negative_outliers):
        logger.info(f"{len(negative_outliers)} classes degraded: {negative_outliers}")
        negative_outliers_metrics = nsp.metrics.get_outliers_metrics(
            negative_outliers,
            pretrain_metrics,
            final_metrics,
            prefix="negative",
            suffix="degradation",
            metric_weights=metric_weights,
        )

    if len(positive_outliers):
        logger.info(f"{len(positive_outliers)} classes improved: {positive_outliers}")
        positive_outliers_metrics = nsp.metrics.get_outliers_metrics(
            positive_outliers,
            pretrain_metrics,
            final_metrics,
            prefix="positive",
            suffix="improvement",
            metric_weights=metric_weights,
        )

    metrics = {
        **deltas,
        **positive_outliers_metrics,
        **negative_outliers_metrics,
        "n_negative_outliers": len(negative_outliers),
        "n_positive_outliers": len(positive_outliers),
    }
    return metrics


def get_metrics_delta(pretrain_metrics, final_metrics, filter_by_name=None):
    deltas = {}

    for metric_name, pretrain_mean in pretrain_metrics["means"].items():
        if metric_name.endswith("_std"):
            continue
        if not nsp.utils.matches_pattern(metric_name, filter_by_name):
            continue

        deltas["delta/" + metric_name] = final_metrics["means"][metric_name] - pretrain_mean

    return deltas


def average_models(
    old_model: nsp.EncoderDecoderWPointerModel,
    new_model: nsp.EncoderDecoderWPointerModel,
    new_model_weight: float,
    inplace=False,
):
    """Averages new_model and old_model parameters.

    Parameters are averaged with weights new_model_weight and (1 - new_model_weight).
    If inplace, mutates new_model object.
    """
    if not (0 <= new_model_weight <= 1):
        raise ValueError(
            f"new_model_weight should be between 0 and 1, got {new_model_weight} instead."
        )
    if not inplace:
        new_model_tmp = nsp.EncoderDecoderWPointerModel(new_model.config)
        new_model_tmp.load_state_dict(new_model.state_dict())
        new_model = new_model_tmp

    old_named_params = old_model.state_dict()
    new_named_params = new_model.state_dict()

    if old_named_params.keys() != new_named_params.keys():
        raise RuntimeError("Models should have the same parameters")

    for name, old_param in old_named_params.items():
        new_named_params[name] = (
            new_model_weight * new_named_params[name] + (1 - new_model_weight) * old_param
        )

    new_model.load_state_dict(new_named_params, strict=True)
    return new_model


def make_lightning_module(
    model,
    schema_tokenizer,
    train_dataset,
    eval_dataset,
    max_tgt_len,
    args,
):
    freezing_schedule = nsp.dataclasses.EncDecFreezingSchedule.from_args(args)

    no_lr_scheduler = getattr(args, "no_lr_scheduler", False)

    lightning_module = nsp.PointerModule(
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        monitor_classes=args.new_classes,
        freezing_schedule=freezing_schedule,
        max_tgt_len=max_tgt_len,
        no_lr_scheduler=no_lr_scheduler,
    )

    return lightning_module
