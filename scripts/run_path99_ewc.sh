set -e
cd ..


SET_NAME=path_99
DATE=Jan8
CLASSES=SL:PATH

DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"_bert_run
BATCH_SIZE=128


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class $CLASSES \
  --split-amount 0.99 \


TAG="$SET_NAME"_"$DATE"_bert_run

python cli/train_lightning.py \
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
  --new-classes $CLASSES \
  --track-grad-square \
  --seed 1 \


TAG="$SET_NAME"_"$DATE"_bert_run_ewc_find
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

for ewc in 0 0.1 10
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --weight-consolidation $ewc \
      --new-classes $CLASSES \
      --tags finetune,$TAG,ewc_"$ewc" \
      --output-dir output_dir/finetuned \
      --old-data-sampling-method sample \


done
done
