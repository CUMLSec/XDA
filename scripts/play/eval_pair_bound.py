import numpy as np
from fairseq.models.roberta import RobertaModel


# Translate predictions to code offsets
def translate_predictions(starts, ends):
    # Pair starts/ends to create function bounds
    bounds = []

    if len(starts) == 0 or len(ends) == 0:
        return bounds

    i_starts = 0
    i_ends = 0

    while i_starts + 1 < len(starts) and i_ends + 1 < len(ends):

        # To prevent errors from propagating...

        # The next end offset is before the next start offset, try to correct
        if ends[i_ends + 1] < starts[i_starts + 1]:
            i_ends += 1
            continue

        # The current end offset is after the next start offset, try to correct
        elif ends[i_ends] >= starts[i_starts + 1]:
            i_starts += 1
            continue

        # This bound pair seems valid, add to bounds
        else:
            bounds.append((starts[i_starts], ends[i_ends]))
            i_starts += 1
            i_ends += 1

    return bounds


def eval_bound_predictions(pred_bounds, real_bounds):
    # Build confusion matrix
    cm = np.zeros((2, 2), dtype=int)

    i, j = 0, 0

    while i < len(pred_bounds) and j < len(real_bounds):

        # True positive
        if pred_bounds[i][0] == real_bounds[j][0] and pred_bounds[i][1] == real_bounds[j][1]:
            cm[0, 0] += 1
            i += 1
            j += 1

        # False positive
        elif pred_bounds[i][0] < real_bounds[j][0]:
            cm[0, 1] += 1
            i += 1

        # False negative
        else:
            cm[1, 0] += 1
            j += 1

    while i < len(pred_bounds):
        cm[0, 1] += 1
        i += 1

    while j < len(real_bounds):
        cm[1, 0] += 1
        j += 1

    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    f1 = 2 * (precision * recall) / (precision + recall)

    print("F1: {:.6f}, Precision: {:.6f}, Recall: {:.6f}".format(f1, precision, recall))
    print(cm)

    return cm


def eval_bound_predictions1(pred_bounds, real_bounds):
    pred_bounds_set = set(pred_bounds)
    real_bounds_set = set(real_bounds)

    TP = pred_bounds_set & real_bounds_set
    p = len(TP) / len(pred_bounds_set)
    r = len(TP) / len(real_bounds_set)
    print(f"F1: {2 * (p * r) / (p + r)}, Precision: {p}, Recall: {r}")


def predict(filename, model):
    f_truth = open(f'{filename}', 'r')

    starts_label = []
    ends_label = []
    starts_pred = []
    ends_pred = []

    tokens = []
    for i, line_truth in enumerate(f_truth):
        # if i > 10000:
        #     break

        line_truth_split = line_truth.strip().split()
        tokens.append(line_truth_split[0].lower())
        if len(line_truth_split) > 1:
            # if line_truth_split[1] == 'F':
            if line_truth_split[1] == 'S':
                starts_label.append(i)
            # elif line_truth_split[1] == 'R':
            elif line_truth_split[1] == 'E':
                ends_label.append(i)
            else:
                pass
    f_truth.close()

    for i_block in range(0, len(tokens), 510):
        if i_block + 510 > len(tokens):
            to_encode_tokens = tokens[i_block:len(tokens)]
        else:
            to_encode_tokens = tokens[i_block:i_block + 510]

        encoded_tokens = model.encode(' '.join(to_encode_tokens))
        logprobs = model.predict('funcbound', encoded_tokens[:510])
        labels = logprobs.argmax(dim=2).view(-1).data

        for i_token, label in enumerate(labels):
            if label == 2:
                starts_pred.append(i_block + i_token)
            elif label == 1:
                ends_pred.append(i_block + i_token)

    return starts_label, ends_label, starts_pred, ends_pred


# Load our model
roberta = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                       'data-bin/funcbound_msvs_64', bpe=None,
                                       user_dir='finetune_tasks')
roberta.eval()

starts_label, ends_label, starts_pred, ends_pred = predict(
    'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code/msvs_64_O2_vim', roberta)
bounds_label = translate_predictions(starts_label, ends_label)
bounds_pred = translate_predictions(starts_pred, ends_pred)

eval_bound_predictions1(bounds_pred, bounds_label)
