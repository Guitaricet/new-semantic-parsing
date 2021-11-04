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
import unittest

from cli import preprocess, train
import cli.retrain_incremental as retrain


DATA = "../data/top-dataset-semantic-parsing-1000"
DATA_SNIPS = "../data/snips/top_format"
DATA_BIN = "test_cli/toy_preprocessed"
DATA_BIN_SNIPS = "test_cli/snips_preprocessed"
MODEL_DIR = "test_cli/train_output"
OUTPUT_DIR = "test_cli/retrain_output"
EPOCHS = 2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_exists = os.path.exists(DATA)
skip_msg = "Ignoring CLI tests as the data is not present"


class TestPreprocessCLI(unittest.TestCase):
    @unittest.skipUnless(data_exists, skip_msg)
    def test01_preprocess(self):
        shutil.rmtree(DATA_BIN, ignore_errors=True)

        args = [
            "--data",
            DATA,
            "--text-tokenizer",
            "bert-base-cased",
            "--output-dir",
            DATA_BIN,
        ]

        args = preprocess.parse_args(args)
        preprocess.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test01_preprocess_split(self):
        shutil.rmtree(DATA_BIN, ignore_errors=True)

        args = [
            "--data",
            DATA,
            "--text-tokenizer",
            "bert-base-cased",
            "--output-dir",
            DATA_BIN,
            "--split-amount",
            "0.1",
        ]

        args = preprocess.parse_args(args)
        preprocess.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test01_preprocess_snips(self):
        shutil.rmtree(DATA_BIN_SNIPS, ignore_errors=True)

        args = [
            "--data",
            DATA_SNIPS,
            "--text-tokenizer",
            "bert-base-cased",
            "--output-dir",
            DATA_BIN_SNIPS,
        ]

        args = preprocess.parse_args(args)
        preprocess.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test01_preprocess_snips_split(self):
        shutil.rmtree(DATA_BIN_SNIPS, ignore_errors=True)

        args = [
            "--data",
            DATA_SNIPS,
            "--text-tokenizer",
            "bert-base-cased",
            "--output-dir",
            DATA_BIN_SNIPS,
            "--split-amount",
            "0.1",
            "--split-class",
            "GETWEATHER",
        ]

        args = preprocess.parse_args(args)
        preprocess.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test02_preprocess_split_class(self):
        shutil.rmtree(DATA_BIN, ignore_errors=True)

        args = [
            "--data",
            DATA,
            "--text-tokenizer",
            "bert-base-cased",
            "--output-dir",
            DATA_BIN,
            "--split-class",
            "IN:GET_DIRECTIONS",
            "--split-amount",
            "0.1",
        ]

        args = preprocess.parse_args(args)
        preprocess.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test03_train(self):
        shutil.rmtree(MODEL_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            MODEL_DIR,
            "--layers",
            "1",
            "--hidden",
            "16",
            "--heads",
            "2",
            "--epochs",
            f"{EPOCHS}",
            "--early-stopping",
            "6",
            "--decoder-lr",
            "0.1",
            "--encoder-lr",
            "0.01",
            "--dropout",
            "0",
            "--log-every",
            "10",
            "--tags",
            "CLI_tests",
            "--eval-data-amount",
            "1.0",
            "--split-amount-finetune",
            "0.314",
            "--min-epochs",
            "1",
        ]

        args = train.parse_args(args)
        train.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test04_retrain(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--lr",
            "0.3",
            "--epochs",
            "1",
            "--log-every",
            "10",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--disable-wandb",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test05_retrain_noargs(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            f"{EPOCHS}",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--disable-wandb",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test06_retrain_old_data001_merge(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--old-data-amount",
            "0.01",
            "--early-stopping",
            "16",
            "--old-data-sampling-method",
            "merge_subset",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test07_retrain_old_data001_sample(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--old-data-amount",
            "0.01",
            "--old-data-sampling-method",
            "sample",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test08_retrain_move_norm(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--move-norm",
            "0.1",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test08_retrain_move_norm_p(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--move-norm",
            "0.1",
            "--move-norm-p",
            "1",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test09_retrain_no_opt(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--no-opt-state",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test10_retrain_dropout(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--dropout",
            "0.9",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test11_retrain_weightdecay(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--weight-decay",
            "0.2",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test12_retrain_labelsmoothing(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--label-smoothing",
            "0.18",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test13_retrain_limit_iters(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--min-steps",
            "3",
            "--max-steps",
            "10",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--label-smoothing",
            "0.18",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test14_retrain_limit_iters(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--epochs",
            "1",
            "--min-steps",
            "3",
            "--max-steps",
            "10",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--label-smoothing",
            "0.18",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test16_retrain_average(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--lr",
            "0.3",
            "--epochs",
            "5",
            "--log-every",
            "10",
            "--new-classes",
            "IN:GET_LOCATION,IN:GET_LOCATION_HOME,SL:POINT_ON_MAP,SL:CATEGORY_LOCATION",
            "--batch-size",
            "16",
            "--no-lr-scheduler",
            "--average-checkpoints",
            "--new-model-weight",
            "1",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test18_retrain_weight_consolidation(self):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--model-dir",
            MODEL_DIR,
            "--tags",
            "CLI_tests",
            "--lr",
            "0.3",
            "--epochs",
            "5",
            "--log-every",
            "10",
            "--new-classes",
            "IN:GET_LOCATION",
            "--batch-size",
            "16",
            "--weight-consolidation",
            "0.1",
        ]

        args = retrain.parse_args(args)
        retrain.main(args)

    @unittest.skipUnless(data_exists, skip_msg)
    def test_19_train_synaptic_intelligence(self):
        # python cli/train.py --data-dir tests/test_cli/toy_preprocessed --output-dir tests/test_cli/train_output --layers 1 --hidden 16 --heads 2 --epochs 2 --use-synaptic-intelligence --lr 1e-3
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        args = [
            "--data-dir",
            DATA_BIN,
            "--output-dir",
            OUTPUT_DIR,
            "--layers",
            "1",
            "--hidden",
            "16",
            "--heads",
            "2",
            "--epochs",
            f"{EPOCHS}",
            "--use-synaptic-intelligence",
            "--lr",
            "1e-3",
        ]

        args = train.parse_args(args)
        train.main(args)
