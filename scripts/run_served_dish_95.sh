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

set -e
cd ..


# Train

SET_NAME=served_dish_90
DATE=Jan14
CLASSES=SL:SERVED_DISH

DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"
FINETUNED="$MODEL"_finetuned
BATCH_SIZE=112
SPLIT=0.90

export TOKENIZERS_PARALLELISM=false


python cli/preprocess.py \
  --data data/snips/top_format \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class $CLASSES \
  --split-amount $SPLIT \


TAG="$SET_NAME"_"$DATE"

python cli/train.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size $BATCH_SIZE \
  --layers 4 \
  --hidden 256 \
  --dropout 0.2 \
  --heads 4 \
  --label-smoothing 0.0 \
  --epochs 100 \
  --early-stopping 10 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --new-classes $CLASSES \
  --track-grad-square \
  --seed 1 \


#TAG="$SET_NAME"_"$DATE"_replay
#
#for old_data_amount in 0.0 0.05 0.1 0.2 0.5 0.7 1.0
#do
#
#    rm -rf $FINETUNED
#
#    python cli/retrain.py \
#      --data-dir $DATA \
#      --model-dir $MODEL \
#      --batch-size $BATCH_SIZE \
#      --dropout 0.2 \
#      --epochs 50 \
#      --early-stopping 10 \
#      --log-every 100 \
#      --new-data-amount 1.0 \
#      --old-data-amount $old_data_amount \
#      --new-classes $CLASSES \
#      --tags finetune,$TAG \
#      --output-dir $FINETUNED \
#      --old-data-sampling-method merge_subset \
#
#done
#
#TAG="$SET_NAME"_"$DATE"_sample
#
#for old_data_amount in 0.0 0.05 0.1 0.2 0.5 0.7 1.0
#do
#
#    rm -rf $FINETUNED
#
#    python cli/retrain.py \
#      --data-dir $DATA \
#      --model-dir $MODEL \
#      --batch-size $BATCH_SIZE \
#      --dropout 0.2 \
#      --epochs 50 \
#      --early-stopping 10 \
#      --log-every 100 \
#      --new-data-amount 1.0 \
#      --old-data-amount $old_data_amount \
#      --new-classes $CLASSES \
#      --tags finetune,$TAG \
#      --output-dir $FINETUNED \
#      --old-data-sampling-method sample \
#
#done
#
#
#TAG="$SET_NAME"_"$DATE"_ewc_replay
#
#for old_data_amount in 0.0 0.05 0.1 0.2 0.5 0.7 1.0
#do
#
#    ewc=100
#
#    rm -rf $FINETUNED
#
#    python cli/retrain.py \
#      --data-dir $DATA \
#      --model-dir $MODEL \
#      --batch-size $BATCH_SIZE \
#      --dropout 0.2 \
#      --epochs 50 \
#      --early-stopping 10 \
#      --log-every 100 \
#      --new-data-amount 1.0 \
#      --old-data-amount $old_data_amount \
#      --weight-consolidation $ewc \
#      --new-classes $CLASSES \
#      --tags finetune,$TAG,ewc_"$ewc" \
#      --output-dir $FINETUNED \
#      --old-data-sampling-method merge_subset \
#
#done


TAG="$SET_NAME"_"$DATE"_ewc_sample-runner2

for old_data_amount in 0.0 0.05 0.1 0.2 0.5 0.7 1.0
do

    ewc=100

    rm -rf $FINETUNED

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --weight-consolidation $ewc \
      --new-classes $CLASSES \
      --tags finetune,$TAG,ewc_"$ewc" \
      --output-dir $FINETUNED \
      --old-data-sampling-method sample \

done


TAG="$SET_NAME"_"$DATE"_move_norm-runner2

for old_data_amount in 0.0 0.05 0.1 0.2 0.5 0.7 1.0
do

for move_norm in 0.05 0.1
do

    rm -rf $FINETUNED

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --move-norm $move_norm \
      --new-classes $CLASSES \
      --tags finetune,$TAG,move_norm_"$move_norm" \
      --output-dir $FINETUNED \
      --old-data-sampling-method merge_subset \

done
done
