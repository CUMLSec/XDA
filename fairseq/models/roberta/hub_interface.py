# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=False) -> torch.LongTensor:
        """
        Encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        """
        # full_sentence = '<s> ' + sentence + ' </s>'
        # for s in addl_sentences:
        #     full_sentence += (' </s>' if not no_separator else '')
        #     full_sentence += ' ' + s + ' </s>'
        full_sentence = sentence
        tokens = self.task.source_dictionary.encode_line(full_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.task.source_dictionary.string(s) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False,
                         save_attn: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            save_attn=save_attn
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features

    def register_classification_head(
            self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens)
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = '<mask>'
        assert masked_token in masked_input and masked_input.count(masked_token) == 1, \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)

        tokens = self.task.source_dictionary.encode_line(
            '<s> ' + masked_input,
            append_eos=True,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token in enumerate(topk_predicted_token.split(' ')):
            topk_filled_outputs.append((
                masked_input.replace(masked_token, predicted_token),
                values[index].item(),
                predicted_token,
            ))
        return topk_filled_outputs

    def fill_multi_mask(self, masked_input: str, topk: int = 5):
        masked_token = '<mask>'
        assert masked_token in masked_input, \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)

        tokens = self.task.source_dictionary.encode_line(masked_input, append_eos=False)

        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        logits = features[0, :, :].squeeze()
        prob = logits.softmax(dim=1)
        values_seq, index_seq = prob.topk(k=topk, dim=1)

        topk_filled_outputs = []
        for values, index in zip(values_seq, index_seq):
            topk_predicted_token = self.task.source_dictionary.string(index)

            topk_filled_output = []
            for index, predicted_token in enumerate(topk_predicted_token.split(' ')):
                topk_filled_output.append((
                    masked_input.replace(masked_token, predicted_token),
                    values[index].item(),
                    predicted_token,
                ))
            topk_filled_outputs.append(topk_filled_output)
        return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(self.task, 'disambiguate_pronoun'), \
            'roberta.disambiguate_pronoun() requires a model trained with the WSC task.'
        with utils.eval(self.model):
            return self.task.disambiguate_pronoun(self.model, sentence, use_cuda=self.device.type == 'cuda')
