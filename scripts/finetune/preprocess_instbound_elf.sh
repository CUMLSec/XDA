#!/usr/bin/env bash

fairseq-preprocess \
  --only-source \
  --srcdict data-bin/pretrain_gcc_clang/dict.txt \
  --trainpref data-src/finetune_instbound_elf/train.data \
  --validpref data-src/finetune_instbound_elf/valid.data \
  --destdir data-bin/finetune_instbound_elf/data \
  --workers 40

fairseq-preprocess \
  --only-source \
  --trainpref data-src/finetune_instbound_elf/train.label \
  --validpref data-src/finetune_instbound_elf/valid.label \
  --destdir data-bin/finetune_instbound_elf/label \
  --workers 40
