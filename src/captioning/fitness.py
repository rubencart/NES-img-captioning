"""
    Based on code from https://github.com/ruotianluo/self-critical.pytorch
"""

import math
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

sys.path.append('cider')
from pyciderevalcap.ciderD.ciderD import CiderD


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def compute_ciders(model, fc_feats, data, gen_result, self_critical):
    """
    Compute the reward of the generated sequences in gen_result. If necessary,
        also generate sequences by greedy decoding (to baseline when self-critical fitness is used).
    """
    cider_scorer = CiderD(df='coco-train-idxs')

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
        greedy_res, _ = model(fc_feats, greedy=True)
        greedy_res = greedy_res.data.cpu().numpy()

        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_res[i])]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    else:
        res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    score, scores = cider_scorer.compute_score(gts, res_)

    if self_critical:
        scores = scores[:batch_size] - scores[batch_size:]
        score = np.mean(scores)

    # scores[:, np.newaxis] makes a column vector of 1d array scores
    # scores = [1, 2, 3] --> scores[:, np.newaxis] = [[1], [2], [3]]
    # np.repeat: https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
    # repeats elements of scores gen_result.shape[1] times along first axis
    # [[1], [2], [3]] --> [[1, 1], [2, 2], [3, 3]]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return score.item(), rewards.copy()


class LogFitnessCriterion(nn.Module):
    """
    From https://github.com/ruotianluo/self-critical.pytorch
        Output = reward * log(prob)
        --> so when prob = 0 ==> output = -inf
                and prob = 1 ==> output = 0
    """

    def __init__(self):
        super(LogFitnessCriterion, self).__init__()

    def forward(self, input, seq, reward):
        """
        :param input:  logprobs
        :param seq:    generated sequence
        :param reward: reward per sequence
        :return:
        """
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)

        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return torch.empty_like(output).copy_(output)


class AltLogFitnessCriterion(nn.Module):
    """
    Different version of LogFitnessCriterion that computes a translated and rescaled log function.
        Output = reward * ( log_10(prob + 1/9) + log_10(9) )
        --> so that when prob = 0 ==> output = 0
                and when prob = 1 ==> output = reward
    """

    def __init__(self):
        super(AltLogFitnessCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)

        pfact = torch.log10(torch.exp(input) + 1/(10 - 1)) + np.log10(10 - 1)
        output = pfact * reward * mask

        output = torch.sum(output) / torch.sum(mask)
        return torch.empty_like(output).copy_(output)


class ExpFitnessCriterion(nn.Module):
    """
        Output = reward * exp(prob - 1) / (e - 1)
        --> so that prob = 0 ==> output = 0
                    prob = 1 ==> output = reward
    """

    def __init__(self):
        super(ExpFitnessCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)

        output = (torch.exp(torch.exp(input)) - 1) / (math.e - 1) * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return torch.empty_like(output).copy_(output)


class LinFitnessCriterion(nn.Module):
    """
        Output = reward * prob
        --> so that prob = 0 ==> output = 0
                    prob = 1 ==> output = reward
    """

    def __init__(self):
        super(LinFitnessCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)

        # input are logprobs so exp(input) is just the probabilities, between 0 and 1
        output = torch.exp(input) * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return torch.empty_like(output).copy_(output)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
