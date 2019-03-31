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

sys.path.append('cider')
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
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


# TODO try with and without self critical!!!!
def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    # model.eval()
    # with torch.no_grad():
    #     # todo mode sample?
    #     greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')

    # model.train()

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    # for i in range(batch_size):
    #     res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    # res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    # res__ = {i: res[i] for i in range(2 * batch_size)}
    # gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    _, cider_scores = CiderD_scorer.compute_score(gts, res_)

    # if opt.cider_reward_weight > 0:
    #     # todo check this
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

    scores = cider_scores

    # todo is this the self critical part?
    # scores = scores[:batch_size] - scores[batch_size:]

    # scores[:, np.newaxis] makes a column vector of 1d array scores
    # scores = [1, 2, 3] --> scores[:, np.newaxis] = [[1], [2], [3]]
    # np.repeat: https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
    # repeats elements of scores gen_result.shape[1] times along first axis
    # [[1], [2], [3]] --> [[1, 1], [2, 2], [3, 3]]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
