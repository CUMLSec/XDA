from fairseq.models.roberta import RobertaModel
import os
import time
import torch


def int2hex(s):
    return s
    # return hex(int(s))[2:]


def run_batch(model, tokens):
    token_stack = []
    total_time = 0
    for token_chunk in tokens:
        encoded_tokens = model.encode(' '.join(token_chunk + ['00'] * (510 - len(token_chunk))))  # pad
        token_stack.append(encoded_tokens)

        if len(token_stack) == 64:
            stack = torch.stack(token_stack, dim=-1)
            t = time.time()
            model.predict('funcbound', stack)
            total_time += (time.time() - t)
            del stack
            del token_stack
            token_stack = []

    # last batch
    if len(token_stack) != 0:
        t = time.time()
        model.predict('funcbound', torch.stack(token_stack, dim=-1))
        total_time += (time.time() - t)

    return len(tokens) * 510 / total_time


def run(model, tokens):
    total_time = 0
    for token_chunk in tokens:
        encoded_tokens = model.encode(' '.join(token_chunk))
        t = time.time()
        model.predict('funcbound', encoded_tokens)
        total_time += (time.time() - t)

    return len(tokens) * 510 / total_time


def predict(filename, model):
    f_truth = open(f'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code/{filename}', 'r')
    tokens = []
    token_chunk = []
    for i, line in enumerate(f_truth):
        if (i + 1) % 510 == 0:
            tokens.append(token_chunk)
            token_chunk = []

        # Prepare the tokens for prediction by XDA
        line_split = line.strip().split()
        token_chunk.append(int2hex(line_split[0]).lower())
    f_truth.close()

    speed = run(model, tokens)
    print("seq:", speed)

    speed = run_batch(model, tokens)
    print("batch:", speed)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GPU Eval
# Load our model
roberta_gpu = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                           'data-bin/funcbound_msvs_64', bpe=None, user_dir='finetune_tasks')
print("GPU:")
roberta_gpu.cuda(2)
roberta_gpu.eval()

predict('msvs_64_O2_vim', roberta_gpu)
del roberta_gpu

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CPU Eval
# Load our model
roberta_cpu = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                           'data-bin/funcbound_msvs_64', bpe=None, user_dir='finetune_tasks')

print("CPU:")
roberta_cpu.eval()

predict('msvs_64_O2_vim', roberta_cpu)
