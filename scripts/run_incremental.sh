set -e
cd ..

export TOKENIZERS_PARALLELISM=false


SET_NAME=top
DATE=20Sep2021

DATA=data-bin/top_incremental/batch_0
MODEL=output_dir/top_incremental_"$DATE"
BATCH_SIZE=112

TAG="$SET_NAME"_"$DATE"_incremental


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
  --epochs 100 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --seed 1 \

#
#for data_dir in data/top-incremental/*
#do
#      python cli/retrain_incremental.py \
#      --data-dir $data_dir \
#      --model-dir $MODEL \
#      --batch-size $BATCH_SIZE \
#      --dropout 0.2 \
#      --epochs 50 \
#      --early-stopping 10 \
#      --log-every 100 \
#      --new-data-amount 1.0 \
#      --old-data-amount 0.2 \
#      --tags finetune,$TAG \
#      --output-dir output_dir/finetuned \
#
#done