#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS-info

rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
    --load_ckpt=checkpoints/epoch=2-v0_server.ckpt \
    --dataset=KAIROS \
    --eval_only \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3 \
    --knowledge-pair-gen

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test_info.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
