import glob
import os

import numpy as np
import torch

opt = 'O1'  # change accordingly
train_prog = 'CPU2017_Windows_VS2019_x64_O1_blender_r'

if not os.path.exists(f'data-bin/robust_funcbound_{opt}_birnn'):
    os.mkdir(f'data-bin/robust_funcbound_{opt}_birnn')

train_data = []
train_label = []
valid_data = []
valid_label = []

label_dict = {'S': 1, 'E': 2}

for filename in glob.glob(f'data-raw/func_bound/CPU2017_Windows_VS2019_x64_{opt}*'):
    f = open(f'{filename}', 'r')
    # if train_prog in filename:
    if train_prog in filename:  # only one file for training
        data, label = train_data, train_label
    else:
        data, label = valid_data, valid_label

    for i, line in enumerate(f):
        line_split = line.strip().split('\t')
        data.append([np.float32(int(line_split[0], 16))])

        # if only one field
        if len(line_split) > 1:
            label.append(np.long(label_dict[line_split[1]]))
        else:
            label.append(np.long(0))
    f.close()

train_data_split = []
for i in range(0, len(train_data), 800):
    if i + 800 <= len(train_data):
        train_data_split.append(train_data[i:i + 800])

train_label_split = []
for i in range(0, len(train_label), 800):
    if i + 800 <= len(train_label):
        train_label_split.append(train_label[i:i + 800])

valid_data_split = []
for i in range(0, len(valid_data), 800):
    if i + 800 <= len(valid_data):
        valid_data_split.append(valid_data[i:i + 800])

valid_label_split = []
for i in range(0, len(valid_label), 800):
    if i + 800 <= len(valid_label):
        valid_label_split.append(valid_label[i:i + 800])

train_data_split = np.array(train_data_split)
train_label_split = np.array(train_label_split)
valid_data_split = np.array(valid_data_split)
valid_label_split = np.array(valid_label_split)

torch.save((train_data_split, train_label_split), f'data-bin/robust_funcbound_{opt}_birnn/train.pt')
torch.save((valid_data_split, valid_label_split), f'data-bin/robust_funcbound_{opt}_birnn/test.pt')
