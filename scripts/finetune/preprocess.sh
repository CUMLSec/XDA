#!/usr/bin/env bash

fairseq-preprocess \
    --only-source \
    --srcdict data-bin/pretrain_all/dict.txt \
    --trainpref data-src/funcbound/train.data \
    --validpref data-src/funcbound/valid.data \
    --destdir data-bin/funcbound/data \
    --workers 40

fairseq-preprocess \
    --only-source \
    --trainpref data-src/funcbound/train.label \
    --validpref data-src/funcbound/valid.label \
    --destdir data-bin/funcbound/label \
    --workers 40