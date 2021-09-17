set -e
cd ..


SET_NAME=road_95
DATE=Dec2
CLASSES=GET_INFO_ROAD_CONDITION

DATA=data-bin/"$SET_NAME"_"Oct27"
MODEL=output_dir/"$SET_NAME"_"$DATE"_bert_run
BATCH_SIZE=128
ACCUM_STEPS=1
SPLIT=0.95


#python cli/preprocess.py \
#  --data data/top-dataset-semantic-parsing \
#  --text-tokenizer bert-base-cased \
#  --output-dir $DATA \
#  --split-class $CLASSES \
#  --split-amount $SPLIT \


python cli/train.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $ACCUM_STEPS \
  --layers 4 \
  --hidden 256 \
  --dropout 0.2 \
  --heads 4 \
  --label-smoothing 0.1 \
  --epochs 100 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --new-classes $CLASSES \
  --seed 1 \


TAG="$SET_NAME"_"$DATE"_bert_run_ewc_find


for old_data_amount in 0.0 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

for ewc in 1000000000 1000000 1000 0
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
