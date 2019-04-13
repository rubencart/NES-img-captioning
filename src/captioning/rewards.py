"""
Code from https://github.com/ruotianluo/self-critical.pytorch
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import time
# import misc.utils as utils
from collections import OrderedDict
import torch

import sys

from torch import nn

sys.path.append('cider')
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('cococaption')
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None


# CiderD_scorer = CiderD(df='corpus')

# default: coco-train-idxs
def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, self_critical):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    if self_critical:
        # get greedy decoding baseline
        model.eval()
        # with torch.no_grad():
        # mode sample but sample_max = 1 by default so greedy
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
        greedy_res = greedy_res.data.cpu().numpy()

        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_res[i])]

        # necessary?
        # model.train()

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
        # res__ = {i: res[i] for i in range(2 * batch_size)}
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    else:
        res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    score, scores = CiderD_scorer.compute_score(gts, res_)

    # if opt.cider_reward_weight > 0:
    #     _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    #     # print('Cider scores:', _)
    # else:
    #     cider_scores = 0
    # if opt.bleu_reward_weight > 0:
    #     _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
    #     bleu_scores = np.array(bleu_scores[3])
    #     # print('Bleu scores:', _[3])
    # else:
    #     bleu_scores = 0
    # scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    if self_critical:
        scores = scores[:batch_size] - scores[batch_size:]
        score = np.mean(scores)

    # scores[:, np.newaxis] makes a column vector of 1d array scores
    # scores = [1, 2, 3] --> scores[:, np.newaxis] = [[1], [2], [3]]
    # np.repeat: https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
    # repeats elements of scores gen_result.shape[1] times along first axis
    # [[1], [2], [3]] --> [[1, 1], [2, 2], [3, 3]]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return score, rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
