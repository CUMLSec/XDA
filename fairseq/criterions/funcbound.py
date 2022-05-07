# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('funcbound')
class FuncboundCriterion(FairseqCriterion):

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
               'funcbound' in model.classification_heads, \
            "model must provide funcbound as classification list for --criterion=funcbound"

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='funcbound',
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        max_sentences, tok_per_batch = sample['net_input']['src_tokens'].size()
        sample_size = targets.numel() / tok_per_batch

        predictions = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        predictions = predictions.view(-1, predictions.size(-1))

        loss = F.nll_loss(predictions, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits.view(-1, logits.size(-1)).max(dim=-1)[1]  # get the index of largest

        # Boundary should only be start and end
        logging_output.update(ncorrect=((preds == targets) * (targets != 0)).sum().item())
        # Total number of function start and end
        logging_output.update(nbound=(targets != 0).sum().item())
        # Total number of predicted start and end
        logging_output.update(nbound_pred=(preds != 0).sum().item())

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

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0] \
                and 'nbound' in logging_outputs[0] and 'nbound_pred' in logging_outputs[0] \
                and 'total_ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            nbound = sum(log.get('nbound', 0) for log in logging_outputs)
            nbound_pred = sum(log.get('nbound_pred', 0) for log in logging_outputs)
            total_ncorrect = sum(log.get('total_ncorrect', 0) for log in logging_outputs)

            precision = 100 * ncorrect / (nbound_pred + 1e-10)
            recall = 100 * ncorrect / (nbound + 1e-10)

            metrics.log_scalar('accuracy', 100.0 * total_ncorrect / ntokens, ntokens, round=1)
            metrics.log_scalar('precision', precision, ntokens, round=1)
            metrics.log_scalar('recall', recall, ntokens, round=1)
            metrics.log_scalar('F1', 2 * (precision * recall) / (precision + recall + 1e-10), ntokens, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
