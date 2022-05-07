#!/usr/bin/env bash

fairseq-preprocess \
  --only-source \
  --srcdict data-bin/pretrain_gcc_clang/dict.txt \
  --trainpref data-src/finetune_instbound_ora/train.data \
  --validpref data-src/finetune_instbound_ora/valid.data \
  --destdir data-bin/finetune_instbound_ora/data \
  --workers 40

fairseq-preprocess \
  --only-source \
  --trainpref data-src/finetune_instbound_ora/train.label \
  --validpref data-src/finetune_instbound_ora/valid.label \
  --destdir data-bin/finetune_instbound_ora/label \
  --workers 40
