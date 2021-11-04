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
"""Metrics computation."""

import sys
import logging
from collections import Counter
from functools import reduce
from operator import add
from pprint import pformat
from typing import List

import torch
import wandb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


LBR = "["
RBR = "]"
IN = "IN:"
SL = "SL:"


class Tree:
    """TOP format tree object."""

    def __init__(self, entity, subtrees: List = None):
        """Makes a tree node with value entity and subtrees.

        if subtrees is None, .subtrees attribute will be empty list

        Args:
            entity: intent/slot value, e.g. IN:INTENT
            subtrees: list of Tree objects
        """
        self.entity = entity
        self.subtrees = subtrees
        if subtrees is None:
            self.subtrees = []

        # for per-class metrics
        self._counts = Counter([entity])
        self._len = 1

        if len(self.subtrees) > 0:
            self._len += sum(map(len, self.subtrees))
            self._counts += reduce(add, (s._counts for s in self.subtrees))

        # for __repr__ and __eq__
        self._dict_repr = {self.entity: [s._dict_repr for s in self.subtrees]}

    def __repr__(self):
        return pformat(self._dict_repr)

    def __eq__(self, other):
        if isinstance(other, dict):
            return self._dict_repr == other
        if isinstance(other, Tree):
            return self._dict_repr == other._dict_repr
        raise ValueError(type(other))

    def __len__(self):
        return self._len

    def get_size(self, classes=None):
        """Gets the number of nodes with node.entity in classes.

        Args:
            classes: optional, a iterable of classes to consider when computing the size

        Returns:
            if classes argument is not specified, returns the total number of nodes
            (same as __len__)
            if classes argument is specified, returns only the count of nodes corresponding to classes
        """
        if classes is None:
            return self.__len__()

        _size = sum(self._counts.get(c, 0) for c in classes)
        return _size

    @property
    def counts(self):
        return self._counts

    @classmethod
    def from_tokens(cls, tokens, return_index=False, inside_slot=False):
        """Builds a parsing tree for labeled bracketing score computation.

        The tree is build until the last ] symbol, everything after it is ignored

        Args:
            tokens: list of tokens
            return_index: used in recursion to provide token index

        Returns:
            Tree object, if return_index == False
            tuple (Tree, index), if return_index == True

        Raises:
            ValueError, if tokens do not represent a valid tree
        """
        # every tree should start with
        # [ ENTITY_TYPE: ENTITY
        if len(tokens) < 3 or tokens[0] != LBR:
            raise ValueError(f"Tree starts with {tokens[:4]}")

        entity_type = tokens[1]

        # ignore invalid subtrees
        if entity_type not in [IN, SL]:
            raise ValueError(f"Tree starts with {tokens[:4]}")

        entity = entity_type + tokens[2]  # e.g. IN:INTENT

        subtrees = []
        slot_value_tokens = []

        safety = 1000
        i = 3
        inside_slot = inside_slot or entity_type == SL
        while i < len(tokens):
            if safety <= 0:
                raise RuntimeError(
                    "The sequence is too long (>1000 tokens) or "
                    "there is a bug in the tree building function."
                )
            safety -= 1
            token = tokens[i]

            # ignore non-slot values
            # e.g. ignore "go stuff in" [IN:STUFF Do stuff]
            if not inside_slot and token not in [LBR, RBR]:
                i += 1
                continue

            # LBR starts a new subtree
            if token == LBR:
                subtree, j = cls.from_tokens(
                    tokens[i:], return_index=True, inside_slot=inside_slot
                )

                if slot_value_tokens:
                    subtrees.append(Tree(" ".join(slot_value_tokens)))
                    slot_value_tokens = []

                subtrees.append(subtree)
                i += j

                continue

            # RBR ends the tree, merge slot values into a single leaf if any
            # e.g. "stuff value" becomes a single leaf in [IN:GET_STUFF [SL:STUF_VALUE stuff value]]
            if token == RBR:
                if slot_value_tokens:
                    subtrees.append(Tree(" ".join(slot_value_tokens)))
                    slot_value_tokens = []

                i += 1
                break

            # if the token is not a special symbol and inside SL: bracket (probably, nested)
            slot_value_tokens.append(token)
            i += 1

        tree = Tree(entity, subtrees)

        if return_index:
            return tree, i

        return tree

    def to_tokens(self):
        if not self.subtrees:
            return self.entity

        return f"[{self.entity} {self.subtrees_to_tokens()}]"

    def subtrees_to_tokens(self):
        return " ".join([s.to_tokens() for s in self.subtrees])


# Main function


def get_metrics(
    pred_tokens,
    true_tokens,
    monitor_classes,
    prefix,
    schema_tokenizer,
    do_each=False,
    verbose=False,
    compute_macro=False,
):
    """Computes exact_match and tree-based metrics

    Apply prefix to all keys.
    The main purpuse of this function is to unify evaluation in PointerModule and cli_utils.evaluate_model()

    Args:
        pred_tokens: List[List[str]]
        true_tokens: List[List[str]]
        monitor_classes: List[str]
        prefix: str, will be appended to all return dict keys
        schema_tokenizer: TopSchemaTokenizer
        do_each: bool, if False compute tree path metrics only for monitor_classes[0] and overall
            if True compute tree path metrics for all monitor_classes and overall
        compute_macro: adds macro-averaged metrics to the return dictionary

    Returns:
        dictionary with keys
            {prefix}_{score_name}
            {prefix}_new_classes_{score_name}
            cls/{prefix}_{monitor_classes[i]}_{score_name}; if do_each=False then only i == 0, else for each class
        for each score_name - key from get_tree_path_scores output dictionary
    """
    if verbose:
        logger.info(f"Getting metrics for classes {monitor_classes}")

    exact_match = sum(int(str(p) == str(l)) for p, l in zip(pred_tokens, true_tokens))
    exact_match /= len(true_tokens)

    tree_metrics = get_tree_path_metrics(
        pred_tokens,
        true_tokens,
        monitor_classes,
        prefix,
        do_each,
        verbose=verbose,
        compute_macro=compute_macro,
    )

    pred_strs = [schema_tokenizer.detokenize(p) for p in pred_tokens]
    true_strs = [schema_tokenizer.detokenize(p) for p in true_tokens]

    exact_match_str = sum(int(p == t) for p, t in zip(pred_strs, true_strs)) / len(true_strs)

    log_dict = {
        f"{prefix}_exact_match": exact_match,
        f"{prefix}_exact_match_str": exact_match_str,
        **tree_metrics,
    }

    return log_dict


# Tree path scores


def get_tree_path_metrics(
    pred_tokens, true_tokens, monitor_classes, prefix, do_each=False, verbose=False, compute_macro=False,
):
    """Gets metrics for all classes, for monitor classes and for monitor_classes[0].

    Apply prefix to all keys.

    Args:
        pred_tokens: List[List[str]]
        true_tokens: List[List[str]]
        monitor_classes: List[str]
        prefix: str, will be appended to all return dict keys
        do_each: bool, if False compute tree path metrics only for monitor_classes[0] and overall
            if True compute tree path metrics for all monitor_classes and overall
        compute_macro: adds macro-averaged metrics to the return dictionary

    Returns:
        dictionary with keys
            {prefix}_{score_name}
            {prefix}_new_classes_{score_name}
            cls/{prefix}_{monitor_classes[i]}_{score_name}, if do_each=False then i is only == 0
        for each score_name - key from get_tree_path_scores output dictionary
    """

    if verbose:
        logger.info("Computing tree path scores.")

    tree_path_scores = get_tree_path_scores(
        pred_tokens=pred_tokens, true_tokens=true_tokens, verbose=verbose
    )
    tree_path_scores = {f"{prefix}_{k}": v for k, v in tree_path_scores.items()}

    if verbose:
        logger.info(
            "Finished computing tree path scores overall. Starting to compute per-class scores."
        )

    if monitor_classes is not None:
        if verbose:
            logger.info(f"Computing scores for classes {monitor_classes}")

        _new_classes_scores = get_tree_path_scores(
            pred_tokens=pred_tokens, true_tokens=true_tokens, classes=monitor_classes
        )
        _new_classes_scores = {
            f"{prefix}_new_classes_{k}": v for k, v in _new_classes_scores.items()
        }
        tree_path_scores.update(_new_classes_scores)

        sum_p, sum_r, sum_f1 = 0, 0, 0
        for i, class_ in enumerate(monitor_classes):
            if verbose:
                logger.info(f"Computing scores for the class {class_}")

            if i > 0 and not do_each:
                break

            _class_score = get_tree_path_scores(
                pred_tokens=pred_tokens, true_tokens=true_tokens, classes=[class_]
            )
            _class_score_prefixed = {f"cls/{prefix}_{class_}_{k}": v for k, v in _class_score.items()}
            tree_path_scores.update(_class_score_prefixed)

            sum_p += _class_score["tree_path_precision"]
            sum_r += _class_score["tree_path_recall"]
            sum_f1 += _class_score["tree_path_f1"]

        if compute_macro:
            p = sum_p / len(monitor_classes)
            r = sum_r / len(monitor_classes)
            f1 = sum_f1 / len(monitor_classes)

            tree_path_scores[f"{prefix}_precision_macro"] = p
            tree_path_scores[f"{prefix}_recall_macro"] = r
            tree_path_scores[f"{prefix}_f1_macro"] = f1

    return tree_path_scores


def get_tree_path_scores(pred_tokens, true_tokens, classes=None, verbose=False):
    """
    Args:
        pred_tokens: list of lists of tokens
        true_tokens: list of lists of tokens

    Returns:
        dictionary with keys
            predicted_paths
            expected_paths
            tree_path_precision
            tree_path_recall
            tree_path_f1
    """
    if verbose:
        logger.info("Computing tree path scores")

    pred_paths_lst, true_paths_lst = [], []

    for pred, true in zip(pred_tokens, true_tokens):
        if verbose:
            logger.info(f"(get_tree_path_scores) Predicted: {pred}")
            logger.info(f"(get_tree_path_scores) Expected:  {true}")

        try:
            pred_paths = _get_paths_with_values(Tree.from_tokens(pred))
        except ValueError:
            pred_paths = dict()

        # we need to build true tree even if pred tree is not valid to compute correct n_expected
        true_paths = _get_paths_with_values(Tree.from_tokens(true))

        pred_paths_lst.append(pred_paths)
        true_paths_lst.append(true_paths)

    true_positives = 0
    n_predicted = 0
    n_expected = 0

    for pred_paths, true_paths in zip(pred_paths_lst, true_paths_lst):
        if classes is None:
            n_expected += len(true_paths)
            n_predicted += len(pred_paths)
        else:
            n_expected += len([p for p in true_paths.keys() if any(class_in_path(c, p) for c in classes)])
            n_predicted += len([p for p in pred_paths.keys() if any(class_in_path(c, p) for c in classes)])

        true_positives += _get_tree_path_matches(pred_paths, true_paths, classes)

    precision = 0
    if n_predicted > 0:
        precision = true_positives / n_predicted

    recall = 0
    if n_expected > 0:
        recall = true_positives / n_expected

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tree_path_precision": precision,
        "tree_path_recall": recall,
        "tree_path_f1": f1,
    }


def _get_paths_with_values(tree, max_depth=10) -> dict:
    """Goes over the tree and return all slot values with the slot names.

    Slot names include paths to this slot. E.g. IN1.SL1: sl1_value.
    Intents and slot without values have None as a value.

    Args:
        tree: Tree object
        max_depth: max recursion depth, returns empty dict if max_depth<=0

    Returns:
        dict(str -> str), maps paths to slot values
        e.g.
            node1.node2.node3: value1
    """
    # check tree for backward branches

    if max_depth <= 0:
        return {}

    is_special = SL in tree.entity or IN in tree.entity
    if not is_special:
        return {}

    if tree.subtrees is None:
        return {tree.entity: None}

    is_slot = SL in tree.entity
    paths = {}

    if is_slot:
        paths[tree.entity] = tree.subtrees_to_tokens()

    for subtree in tree.subtrees:
        subpaths = _get_paths_with_values(subtree, max_depth=max_depth - 1)
        for path, value in subpaths.items():
            paths[tree.entity + "." + path] = value

    return paths


def _get_tree_path_matches(pred_tree_paths, true_tree_paths, classes=None):
    n_matches = 0
    for true_path, true_slot in true_tree_paths.items():
        path_value_match = true_path in pred_tree_paths and pred_tree_paths[true_path] == true_slot
        class_match = classes is None or any(c in true_path for c in classes)

        if path_value_match and class_match:
            n_matches += 1

    return n_matches


# Non-tree metrics: EM, Accuracy, first intent precision


def compute_metrics_from_batch(predictions, labels, masks, stop_tokens):
    """Computes accuracy, exact match and first intent precision
     where all predictions, labels and masks are torch.tensor
     """
    device = predictions.device

    # correct tokens which are not masked (masked tokens have mask == 0)
    n_correct = ((predictions == labels) & masks.bool()).sum()
    n_total = masks.sum()
    accuracy = n_correct / n_total.float()

    # trauncate until EOS token
    # for exact match we consider all tokens until EOS/PAD
    # this is closer to inference setup when generation stops after EOS/PAD

    def truncate(pred):
        i = 0
        for i, idx in enumerate(pred):
            if idx in stop_tokens:
                break
        return pred[:i]

    truncated_preds = [truncate(p) for p in predictions.detach().unbind()]
    truncated_labels = [truncate(l) for l in labels.detach().unbind()]

    exact_match = sum(
        (p.shape == l.shape and torch.all(p == l))
        for p, l in zip(truncated_preds, truncated_labels)
    )
    if exact_match == 0:
        exact_match = torch.tensor(0.0, device=device)

    exact_match = exact_match.float() / len(truncated_preds)

    # intent is the third token in the sequence
    # [ IN: INTENT ...
    # 0 1   2
    intent_tokens_preds = predictions[:, 2]
    intent_tokens_labels = labels[:, 2]

    first_intent_precision = torch.sum(intent_tokens_preds == intent_tokens_labels) / float(
        intent_tokens_preds.shape[0]
    )

    return {
        "accuracy": accuracy,
        "exact_match": exact_match,
        "first_intent_precision": first_intent_precision,
    }


# Metrics used to evaluate finetuning procedure


def get_outliers_metrics(
    metric_names, initial_metrics, final_metrics, prefix, suffix, metric_weights
):
    """Gets aggregate metrics evaluating the change of initial and final performance.

    Metrics are summed with metric_weights for relative metrics and without weights for absolute metrics.
    E.g., if metric_names is a list of positive outliers, this function computes Relative Improvemen
    and the total metric increase on these classes.

    Args:
        metric_names: list of metrics to compute change on
        initial_metrics: dict metric_name:metric_value
        final_metrics: dict metric_name:metric_value
        prefix: string used for naming, see Returns
        suffix: string used for naming, see Returns
        metric_weights: weights used to aggregate relative change

    Returns:
        dict with keys
            {prefix}_outliers
            absolute_{suffix}
            relative_{suffix}
    """
    table = wandb.Table(
        columns=[
            "metric_name",
            "pretrain_mean",
            "pretrain_stdev",
            "final_mean",
            "final_stdev",
            "absolute_improvement",
            "relative_improvement",
            "class_frequency",
        ]
    )

    abs_deltas = {}
    rel_deltas = {}
    digits = 4

    for name in metric_names:
        abs_delta = final_metrics["means"][name] - initial_metrics["means"][name]
        rel_delta = abs_delta / max(initial_metrics["means"][name], 0.001)
        abs_deltas[name] = abs_delta
        rel_deltas[name] = rel_delta

        table.add_data(
            name,
            round(initial_metrics["means"][name], digits),
            round(initial_metrics["stdevs"][name + "_std"], digits),
            round(final_metrics["means"][name], digits),
            round(final_metrics["stdevs"][name + "_std"], digits),
            round(abs_delta, digits),
            round(rel_delta, digits),
            round(metric_weights[name], digits),
        )

    abs_delta_overall = 0
    rel_delta_overall = 0

    if len(abs_deltas) > 0:
        abs_delta_overall = sum(abs_deltas.values())
        rel_delta_overall = sum(metric_weights[name] * v for name, v in rel_deltas.items())

        abs_delta_overall = round(abs_delta_overall, digits)
        rel_delta_overall = rel_delta_overall

    res = {
        f"{prefix}_outliers": table,
        f"absolute_{suffix}": abs_delta_overall,
        f"relative_{suffix}": rel_delta_overall,
    }
    return res


def class_in_path(class_name, path_str):
    """
    Checks if the class of interest is in the tree path

    Note that the path is supposed to have the nodes separated by "."-symbol.
    The main issue this function addresses is ambiguous cases, e.g.
    SL:PATH vs SL:PATH_AVOID

    Args:
        class_name: str, e.g. SL:PATH
        path_str: str, e.g. IN:GET_ESTIMATED_DEPARTURE.SL:SOURCE

    Returns:
        True if the class in the tree path, else otherwise
    """
    # "DEPARTURE." in "IN:DEPARTURE.SL:SOURCE" case
    class_in_the_middle = (class_name + ".") in path_str

    # "IN:DEPARTURE.SL:SOURCE".endswith("SOURCE") case
    class_in_the_end = path_str.endswith(class_name)

    return class_in_the_middle or class_in_the_end
