#!/usr/bin/env bash

fairseq-preprocess \
  --only-source \
  --trainpref data-src/pretrain_gcc_clang/train.in \
  --validpref data-src/pretrain_gcc_clang/valid.in \
  --destdir data-bin/pretrain_gcc_clang \
  --workers 40
