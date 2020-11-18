#!/usr/bin/env bash

fairseq-preprocess \
    --only-source \
    --trainpref data-src/pretrain_all/train.in \
    --validpref data-src/pretrain_all/valid.in \
    --destdir data-bin/pretrain_all \
    --workers 40