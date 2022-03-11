#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS-info-simtr-adv

rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
    --load_ckpt=checkpoints/gen-KAIROS-info-simtr/epoch=3.ckpt \
    --dataset=KAIROS \
    --eval_only \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info_adv.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3 \
    --coref_dir=data/wikievents/coref \
    --sim_train \
    --adv \
    # --knowledge-pair-gen \


python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test_info_adv.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test_adv.jsonlines \
--head-only \
--coref 

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
#     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=2-v0.ckpt \
#     --dataset=KAIROS \
#     --eval_only \
#     --train_file=data/wikievents/train_info.jsonl \
#     --val_file=data/wikievents/dev_info.jsonl \
#     --test_file=data/wikievents/test_info.jsonl \
#     --train_batch_size=4 \
#     --eval_batch_size=1 \
#     --learning_rate=3e-5 \
#     --accumulate_grad_batches=4 \
#     --num_train_epochs=3 \
#     --coref_dir=data/wikievents/coref \
#     --sim_train \
#     # --knowledge-pair-gen

# python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
# --test-file=data/wikievents/test_info.jsonl \
# --dataset=KAIROS \
# --coref-file=data/wikievents/coref/test.jsonlines \
# --head-only \
# --coref 

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
#     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=3.ckpt \
#     --dataset=KAIROS \
#     --eval_only \
#     --train_file=data/wikievents/train_info.jsonl \
#     --val_file=data/wikievents/dev_info.jsonl \
#     --test_file=data/wikievents/test_info.jsonl \
#     --train_batch_size=4 \
#     --eval_batch_size=1 \
#     --learning_rate=3e-5 \
#     --accumulate_grad_batches=4 \
#     --num_train_epochs=3 \
#     --coref_dir=data/wikievents/coref \
#     --sim_train \
#     # --knowledge-pair-gen

# python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
# --test-file=data/wikievents/test_info.jsonl \
# --dataset=KAIROS \
# --coref-file=data/wikievents/coref/test.jsonlines \
# --head-only \
# --coref 

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
#     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=3-v0.ckpt \
#     --dataset=KAIROS \
#     --eval_only \
#     --train_file=data/wikievents/train_info.jsonl \
#     --val_file=data/wikievents/dev_info.jsonl \
#     --test_file=data/wikievents/test_info.jsonl \
#     --train_batch_size=4 \
#     --eval_batch_size=1 \
#     --learning_rate=3e-5 \
#     --accumulate_grad_batches=4 \
#     --num_train_epochs=3 \
#     --coref_dir=data/wikievents/coref \
#     --sim_train \
#     # --knowledge-pair-gen

# python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
# --test-file=data/wikievents/test_info.jsonl \
# --dataset=KAIROS \
# --coref-file=data/wikievents/coref/test.jsonlines \
# --head-only \
# --coref 

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
#     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=4.ckpt \
#     --dataset=KAIROS \
#     --eval_only \
#     --train_file=data/wikievents/train_info.jsonl \
#     --val_file=data/wikievents/dev_info.jsonl \
#     --test_file=data/wikievents/test_info.jsonl \
#     --train_batch_size=4 \
#     --eval_batch_size=1 \
#     --learning_rate=3e-5 \
#     --accumulate_grad_batches=4 \
#     --num_train_epochs=3 \
#     --coref_dir=data/wikievents/coref \
#     --sim_train \
#     # --knowledge-pair-gen

# python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
# --test-file=data/wikievents/test_info.jsonl \
# --dataset=KAIROS \
# --coref-file=data/wikievents/coref/test.jsonlines \
# --head-only \
# --coref 

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
#     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=4-v0.ckpt \
#     --dataset=KAIROS \
#     --eval_only \
#     --train_file=data/wikievents/train_info.jsonl \
#     --val_file=data/wikievents/dev_info.jsonl \
#     --test_file=data/wikievents/test_info.jsonl \
#     --train_batch_size=4 \
#     --eval_batch_size=1 \
#     --learning_rate=3e-5 \
#     --accumulate_grad_batches=4 \
#     --num_train_epochs=3 \
#     --coref_dir=data/wikievents/coref \
#     --sim_train \
#     # --knowledge-pair-gen

# python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
# --test-file=data/wikievents/test_info.jsonl \
# --dataset=KAIROS \
# --coref-file=data/wikievents/coref/test.jsonlines \
# --head-only \
# --coref 

# --load_ckpt=checkpoints/${CKPT_NAME}/epoch=3.ckpt \
# --load_ckpt=checkpoints/epoch=2-v0_server.ckpt \