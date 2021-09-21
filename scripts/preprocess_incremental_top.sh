set -e
cd ..

for data_dir in data/top-incremental/*
do

echo "Processing ($data_dir)"
python cli/preprocess_class_incremental.py \
  --data $data_dir \
  --text-tokenizer bert-base-cased \
  --output-dir "data-bin/top_incremental/$(basename $data_dir)"

done
