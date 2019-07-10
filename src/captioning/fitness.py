"""
    Based on code from https://github.com/ruotianluo/self-critical.pytorch
"""

import math

import numpy as np
import torch
from torch import nn


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
