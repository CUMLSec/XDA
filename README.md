## Introduction

XDA is a tool to disassemble instructions and recovers function boundaries of stripped binaries. It is based on transfer learning using Transformer encoder with masked language modeling objective [1, 2, 3]. It outperforms state-of-the-art tools (e.g., IDA Pro, Ghidra, and bidirectional RNN [4]). Please find the details in our paper: [XDA: Accurate, Robust Disassembly with Transfer Learning](https://arxiv.org/abs/2010.00770)


```
@inproceedings{pei2021xda,
    title={XDA: Accurate, Robust Disassembly with Transfer Learning},
    author={Pei, Kexin and Guan, Jonas and King, David Williams and Yang, Junfeng and Jana, Suman},
    year={2021},
    booktitle={Proceedings of the 2021 Network and Distributed System Security Symposium (NDSS)}
}
```


## Installation
We recommend using `conda` to setup the environment and install the required packages.

First, create the conda environment,

`conda create -n xda python=3.7 numpy scipy scikit-learn colorama`

and activate the conda environment:

`conda activate xda`

Then, install the latest Pytorch (assume you have GPU):

`conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`

Finally, enter the xda root directory: e.g., `path/to/xda`, and install XDA:

`pip install --editable .`

## Preparation

### Pretrained models:

Create the `checkpoints` and `checkpoints/pretrain_all` subdirectory in `path/to/xda`

`mkdir -p checkpoints/pretrain_all`

Download our [pretrained weight parameters](https://drive.google.com/file/d/18LMUt6xJGTrSJ4HoaXBUYt2le3YNGcOu/view?usp=sharing) and put in `checkpoints/pretrain_all`

### Finetuned models:

We also provide the finetuned model for you to directly play on function boundary recovery. The finetuned model is trained on binaries compiled by MSVC x64. Create the `checkpoints/finetune_msvs_funcbound_64` subdirectory in `path/to/xda`

`mkdir -p checkpoints/finetune_msvs_funcbound_64`

Download our [finetuned weight parameters](https://drive.google.com/file/d/1103Hq2ZShlF-4qRPudtjDru5fBqAckds/view?usp=sharing) and put in `checkpoints/finetune_msvs_funcbound_64`. 

#### Play with the finetuned model
We have put some sample data from BAP corpus compiled by MSVC x64 in `data-raw/msvs_funcbound_64_bap_test`. There are two columns in the data files. The first column is all raw bytes of the binary, and the second column is the label indicating it is function start (F), function end (R), or neither.

To predict the function boundary in these files, run:

`python scripts/play/play_func_bound.py`

This scripts will load the finetuned weights you put in `checkpoints/finetune_msvs_funcbound_64` and predict the function boundaries. It will also compare to the ground-truth and the results from IDA.


### Sample data with function boundaries

We provide the sample training/testing files of pretraining and finetuning in `data-src/`

- `data-src/pretrain_all` contains the sample raw bytes from stripped binaries for pretraining
- `data-src/funcbound` contains the sample raw bytes with function boundaries


We have already provided the [pretrained models](https://drive.google.com/file/d/18LMUt6xJGTrSJ4HoaXBUYt2le3YNGcOu/view?usp=sharing) on a huge number of binaries. But if you want to pretrain on your own collected data, you can prepare the sample files similar to the format in `data-src/pretrain_all` (concatenate all bytes from all binaries, and delimit by a newline `\n` to make sure each line does not exceed the max length that model accepts). 
Similarly, if you want to prepare the finetuning data yourself, make sure you follow the format shown in `data-src/funcbound`.

We have to binarize the data to make it ready to be trained. To binarize the training data for pretraining, run:

`./scripts/pretrain/preprocess-pretrain-all.sh`

The binarized training data ready for pretraining will be stored at `data-bin/pretrain_all`

To binarize the training data for finetuning, run:

`./scripts/finetune/preprocess.sh`

The binarized training data ready for finetuning (for function boundary) will be stored at `data-bin/funcbound`

## Training

If you are using your own parsed binaries for pretraining, and you have already binarized them in `data-bin/pretrain_all`, run:

`./scripts/pretrain/pretrain-all.sh`

To finetune the model, run:

`./scripts/finetune/finetune.sh`

The scripts loads the pretrained weight parameters from `checkpoints/pretrain_all/` and finetunes the model.

## RNN baseline
- bi-RNN implementation is released under ./bi-RNN/
- To run, download our sample processed SPEC 2017 O1 dataset [training](https://drive.google.com/file/d/1me1b5sbZM8nncVWevwf7v2jEEYNF_jm_/view?usp=sharing), [testing](https://drive.google.com/file/d/1FD_9pXMiDJ61mmmeaQse4xobM8RAzPZN/view?usp=sharing) and put in `birnn/`

## References
[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

[2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[3] Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[4] Shin, Eui Chul Richard, Dawn Song, and Reza Moazzezi. "Recognizing functions in binaries with neural networks." 24th USENIX Security Symposium. 2015.
