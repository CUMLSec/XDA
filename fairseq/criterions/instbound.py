# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('instbound')
class InstboundCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
               'instbound' in model.classification_heads, \
            "model must provide instbound as classification list for --criterion=instbound"

        not_padded = sample['net_input']['src_tokens'].ne(self.padding_idx)

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='instbound',
        )
        targets = model.get_targets(sample, [logits])
        targets = targets[not_padded]

        max_sentences, tok_per_batch = sample['net_input']['src_tokens'].size()
        sample_size = targets.numel() / tok_per_batch

        predictions = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        predictions = predictions[not_padded]
        loss = F.nll_loss(predictions, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': targets.size(0),
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = predictions.view(-1, predictions.size(-1)).max(dim=-1)[1]  # get the index of largest

        # Boundary should only be start, then it will be correct
        logging_output.update(tp_B=((preds == targets) * (targets == 0)).sum().item())
        # Total number of instruction start
        logging_output.update(tp_fn_B=(targets == 0).sum().item())
        # Total number of predicted start
        logging_output.update(tp_fp_B=(preds == 0).sum().item())

        # Boundary should only be start, then it will be correct
        logging_output.update(tp_S=((preds == targets) * (targets == 1)).sum().item())
        # Total number of instruction start
        logging_output.update(tp_fn_S=(targets == 1).sum().item())
        # Total number of predicted start
        logging_output.update(tp_fp_S=(preds == 1).sum().item())

        # Boundary should only be start, then it will be correct
        logging_output.update(tp_N=((preds == targets) * (targets == 2)).sum().item())
        # Total number of instruction start
        logging_output.update(tp_fn_N=(targets == 2).sum().item())
        # Total number of predicted start
        logging_output.update(tp_fp_N=(preds == 2).sum().item())

        # Total
        logging_output.update(total_ncorrect=(preds == targets).sum().item())

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'total_ncorrect' in logging_outputs[0]:
            tp_B = sum(log.get('tp_B', 0) for log in logging_outputs)
            tp_fp_B = sum(log.get('tp_fp_B', 0) for log in logging_outputs)
            tp_fn_B = sum(log.get('tp_fn_B', 0) for log in logging_outputs)

            tp_S = sum(log.get('tp_S', 0) for log in logging_outputs)
            tp_fp_S = sum(log.get('tp_fp_S', 0) for log in logging_outputs)
            tp_fn_S = sum(log.get('tp_fn_S', 0) for log in logging_outputs)

            tp_N = sum(log.get('tp_N', 0) for log in logging_outputs)
            tp_fp_N = sum(log.get('tp_fp_N', 0) for log in logging_outputs)
            tp_fn_N = sum(log.get('tp_fn_N', 0) for log in logging_outputs)

            total_ncorrect = sum(log.get('total_ncorrect', 0) for log in logging_outputs)

            precision_B = 100 * tp_B / (tp_fp_B + 1e-10)
            recall_B = 100 * tp_B / (tp_fn_B + 1e-10)

            precision_S = 100 * tp_S / (tp_fp_S + 1e-10)
            recall_S = 100 * tp_S / (tp_fn_S + 1e-10)

            precision_N = 100 * tp_N / (tp_fp_N + 1e-10)
            recall_N = 100 * tp_N / (tp_fn_N + 1e-10)

            metrics.log_scalar('accuracy', 100.0 * total_ncorrect / ntokens, ntokens, round=1)

            metrics.log_scalar('precision_B', precision_B, ntokens, round=1)
            metrics.log_scalar('recall_B', recall_B, ntokens, round=1)
            metrics.log_scalar('F1_B', 2 * (precision_B * recall_B) / (precision_B + recall_B + 1e-10), ntokens,
                               round=1)

            metrics.log_scalar('precision_S', precision_S, ntokens, round=1)
            metrics.log_scalar('recall_S', recall_S, ntokens, round=1)
            metrics.log_scalar('F1_S', 2 * (precision_S * recall_S) / (precision_S + recall_S + 1e-10), ntokens,
                               round=1)

            metrics.log_scalar('precision_N', precision_N, ntokens, round=1)
            metrics.log_scalar('recall_N', recall_N, ntokens, round=1)
            metrics.log_scalar('F1_N', 2 * (precision_N * recall_N) / (precision_N + recall_N + 1e-10), ntokens,
                               round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
