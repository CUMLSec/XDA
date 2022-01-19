from fairseq.models.roberta import RobertaModel
import os
from collections import defaultdict
from colorama import Fore, Back, Style
import torch
import sys

def int2hex(s):
    return s
    # return hex(int(s))[2:]


def ida_f1():
    for filename in os.listdir('data-raw-bak/msvs_funcbound_64_bap_test/ida_labeled_code'):
        TP = FP = FN = TN = 0
        f_ida = open(f'data-raw-bak/msvs_funcbound_64_bap_test/ida_labeled_code/{filename}', 'r')
        f_truth = open(f'data-raw-bak/msvs_funcbound_64_bap_test/truth_labeled_code/{filename}', 'r')
        for line_ida, line_truth in zip(f_ida, f_truth):
            line_ida_split = line_ida.strip().split()
            line_truth_split = line_truth.strip().split()
            if line_ida_split[1] == line_truth_split[1] and (line_truth_split[1] == 'F' or line_truth_split[1] == 'R'):
                TP += 1
            elif line_ida_split[1] == line_truth_split[1] == '-':
                TN += 1
            elif line_ida_split[1] in ['F', 'R'] and line_truth_split[1] == '-':
                FP += 1
            elif line_ida_split[1] == '-' and line_truth_split[1] in ['F', 'R']:
                FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        print(f'{filename}: TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, precision: {precision}, recall: {recall}, F1: {F1}')

        f_ida.close()
        f_truth.close()


def predict_color(filename, model, start_idx=0, end_idx=0):
    f_truth = open(f'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code/{filename}', 'r')
    f_ida = open(f'data-raw/msvs_funcbound_64_bap_test/ida_labeled_code/{filename}', 'r')

    tokens = []
    print('\nGround Truth:')
    for i, line_truth in enumerate(f_truth):
        if i < start_idx:
            continue

        line_truth_split = line_truth.strip().split()
        if line_truth_split[1] == '-':
            print(f'{int2hex(line_truth_split[0]).lower()}', end=" ")
        elif line_truth_split[1] == 'F':
            print(f'{Fore.RED}{int2hex(line_truth_split[0]).lower()}{Fore.RESET}', end=" ")
        elif line_truth_split[1] == 'R':
            print(f'{Fore.GREEN}{int2hex(line_truth_split[0]).lower()}{Fore.RESET}', end=" ")

        if i > end_idx:
            print(Style.RESET_ALL + '\n')
            break
    f_truth.close()

    f_truth = open(f'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code/{filename}', 'r')
    for i, line_truth in enumerate(f_truth):
        if i < start_idx:
            continue

        line_truth_split = line_truth.strip().split()
        print(f'{line_truth_split[1]}', end=" ")

        if i > end_idx:
            print(Style.RESET_ALL + '\n')
            break
    f_truth.close()

    print('IDA-PRO:')
    for i, line_ida in enumerate(f_ida):
        if i < start_idx:
            continue

        line_ida_split = line_ida.strip().split()
        if line_ida_split[1] == '-':
            print(f'{int2hex(line_ida_split[0]).lower()}', end=" ")
        elif line_ida_split[1] == 'F':
            print(f'{Fore.RED}{int2hex(line_ida_split[0]).lower()}{Fore.RESET}', end=" ")
        elif line_ida_split[1] == 'R':
            print(f'{Fore.GREEN}{int2hex(line_ida_split[0]).lower()}{Fore.RESET}', end=" ")

        # Prepare the tokens for prediction by XDA
        tokens.append(int2hex(line_ida_split[0]).lower())

        if i > end_idx:
            print(Style.RESET_ALL + '\n')
            break
    f_ida.close()

    encoded_tokens = model.encode(' '.join(tokens))
    # print(encoded_tokens)
    logprobs = model.predict('funcbound', encoded_tokens)
    labels = logprobs.argmax(dim=2).view(-1).data

    print('XDA:')
    func_start = []
    func_end = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 0:
            print(f'{token}', end=" ")
        elif label == 2:
            print(f'{Fore.RED}{token}{Fore.RESET}', end=" ")
            func_start.append((i, token))
        elif label == 1:
            print(f'{Fore.GREEN}{token}{Fore.RESET}', end=" ")
            func_end.append((i, token))

    print(Style.RESET_ALL + '\n')
    return tokens, func_start, func_end


print(f'{Fore.RED}red:function start{Fore.RESET}')
print(f'{Fore.GREEN}green:function end{Fore.RESET}')

# Load our model
roberta = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                       'data-bin/funcbound_msvs_64', bpe=None, user_dir='finetune_tasks')
roberta.eval()

# ida_f1()
#tokens, func_start, func_end = predict_color('msvs_64_O2_vim', roberta, start_idx=0, end_idx=510)
tokens, func_start, func_end = predict_color('msvs_64_O2_vim', roberta, start_idx=int(sys.argv[1]), end_idx=int(sys.argv[2]))
