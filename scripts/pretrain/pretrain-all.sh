#!/usr/bin/env bash

#rm checkpoints/pretrain_all -rf

mkdir -p checkpoints/pretrain_all

TOTAL_UPDATES=305000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0001         # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=32          # Increase the batch size 32x

CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/pretrain_all \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode none --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format json --log-interval 10 \
    --no-epoch-checkpoints --save-dir checkpoints/pretrain_all/ \
    --memory-efficient-fp16 \
    --mask-prob 0.2 --random-token-prob 0.5 \
    | tee -a result/pretrain_all
