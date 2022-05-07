#!/usr/bin/env bash

MAX_EPOCHS=30
WARMUP_UPDATES=500
LR=1e-05         # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3    # S - start of instruction, B - instruction body, - not instruction
MAX_SENTENCES=32 # Batch size.

INSTBOUND_PATH=checkpoints/finetune_instbound_elf
mkdir -p $INSTBOUND_PATH
rm -f $INSTBOUND_PATH/checkpoint_best.pt

# finetune on pretrained weights
cp checkpoints/pretrain_gcc_clang/checkpoint_best.pt $INSTBOUND_PATH/

CUDA_VISIBLE_DEVICES=0 python train.py data-bin/finetune_instbound_elf/ \
  --max-positions 512 \
  --max-sentences $MAX_SENTENCES \
  --user-dir finetune_tasks \
  --task instbound \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --arch roberta_base \
  --criterion instbound \
  --num-classes $NUM_CLASSES \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $LR --max-epoch $MAX_EPOCHS --warmup-updates $WARMUP_UPDATES \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --find-unused-parameters \
  --no-epoch-checkpoints --log-format=json --log-interval 10 \
  --save-dir $INSTBOUND_PATH \
  --restore-file $INSTBOUND_PATH/checkpoint_best.pt \
  --memory-efficient-fp16 |
  tee result/finetune_instbound_elf
