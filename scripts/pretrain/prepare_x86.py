import glob
import json
import random

train = open('data-src/pretrain_gcc_clang/train.in', 'w')
valid = open('data-src/pretrain_gcc_clang/valid.in', 'w')

for funcbyte in glob.glob('data-raw/funcbytes/x86-*'):
    with open(funcbyte, 'r') as f:
        d = json.loads(f.read())

    for funcname in d:
        for filename in d[funcname]:
            if random.random() > 0.001:
                train.write(' '.join(d[funcname][filename]) + '\n')
            else:
                valid.write(' '.join(d[funcname][filename]) + '\n')

train.close()
valid.close()
