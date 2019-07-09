"""
    Code based on https://github.com/ruotianluo/self-critical.pytorch
"""
import logging
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.nets import PolicyNet


class CaptionModel(PolicyNet, ABC):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
        super(CaptionModel, self).__init__(rng_state, from_param_file, grad, options, vbn)

    def forward(self, *args, **kwargs):
        return self._sample(*args, **kwargs)

    def forward_for_sensitivity(self, data, orig_bs=0, i=-1, split=100, length=5):
        """
        Compute output (i.e., logprobs) of model after length steps of greedy decoding, intended to be used
            for the sensitivity calculation. To reduce the output dimension, groups of split
            elements are aggregated.
        :param data:     input batch
        :param orig_bs:  batch size, if this is smaller than data size, only first orig_bs of data are used
        :param i:        if i is not -1, only compute output for 1 element of data
        :param split:    size of groups
        :param length:   number of decoding steps (< max seg length = 16)
        :return: aggregate of logprobs, with dim orig_bs x round_up(vocab size / split)
        """
        device = next(self.parameters()).device
        self.to(device)
        fc_feats = torch.from_numpy(data['fc_feats']).to(device)
        fc_feats = torch.zeros_like(fc_feats).copy_(fc_feats)

        # we assume 5 seqs per image
        # fc feats has length seq_per_img x batch_size, and every # seq_per_img imgs are equal
        fc_feats = fc_feats.index_select(dim=0, index=torch.arange(0, fc_feats.size(0), 5))

        if fc_feats.size(0) > orig_bs > 0:
            fc_feats = fc_feats[:orig_bs]
        if i >= 0:
            fc_feats = fc_feats[i].unsqueeze(0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        xt = self.img_embed(fc_feats)
        _, state = self.core(xt, state)

        it = fc_feats.data.new(batch_size).long().zero_()
        for t in range(length):
            xt = self.embed(it)
            output, state = self.core(xt, state)

            # |batch_size| x |vocab size|
            logprobs = F.log_softmax(self.logit(output), dim=1)

            _, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()

        cat_size = split - (logprobs.size(1) % split)
        extended_lp = torch.cat((logprobs, torch.zeros((batch_size, cat_size))), 1)
        split_lp = extended_lp.split(split, dim=1)

        # summed_lp = torch.stack(split_lp).sum(2).permute(1, 0)
        summed_lp = ((torch.stack(split_lp)**2).sum(2)**(1/2)).permute(1, 0)
        return summed_lp

    def _sample(self, fc_feats, greedy=True):
        raise NotImplementedError


class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)

        # virtual batch normalization
        self.vbn = opt.vbn
        self.layer_n = opt.layer_n
        if self.vbn:
            logging.info('VIRTUAL BATCH NORM ACTIVE')
            self.i2h_bn = nn.BatchNorm1d(5 * self.rnn_size, track_running_stats=False, affine=opt.vbn_affine)
            self.h2h_bn = nn.BatchNorm1d(5 * self.rnn_size, track_running_stats=False, affine=opt.vbn_affine)
            self.c_bn = nn.BatchNorm1d(self.rnn_size, track_running_stats=False, affine=opt.vbn_affine)
        elif self.layer_n:
            logging.info('LAYER NORM ACTIVE')
            self.i2h_ln = nn.LayerNorm(5 * self.rnn_size, elementwise_affine=opt.layer_n_affine)
            self.h2h_ln = nn.LayerNorm(5 * self.rnn_size, elementwise_affine=opt.layer_n_affine)
            self.c_ln = nn.LayerNorm(self.rnn_size, elementwise_affine=opt.layer_n_affine)

    def forward(self, xt, state):
        if self.vbn:
            xt_i2h = self.i2h_bn(self.i2h(xt))
            state_h2h = self.h2h_bn(self.h2h(state[0][-1]))
        elif self.layer_n:
            xt_i2h = self.i2h_ln(self.i2h(xt))
            state_h2h = self.h2h_ln(self.h2h(state[0][-1]))
        else:
            xt_i2h = self.i2h(xt)
            state_h2h = self.h2h(state[0][-1])

        all_input_sums = xt_i2h + state_h2h
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)

        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size)
        )

        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        if self.vbn:
            activated_next_c = torch.tanh(self.c_bn(next_c))
        elif self.layer_n:
            activated_next_c = torch.tanh(self.c_ln(next_c))
        else:
            activated_next_c = torch.tanh(next_c)
        next_h = out_gate * activated_next_c

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class FCModel(CaptionModel):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
        super(FCModel, self).__init__(rng_state, from_param_file, grad, options, vbn)
        opt = options

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.fc_feat_size = opt.fc_feat_size
        self.num_layers = 1
        self.seq_length = 16

        # CAUTION: order is important when you use vector safe mutations!
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.core = LSTMCore(opt)

        # virtual batch normalization
        self.vbn_e = opt.vbn_e
        if self.vbn_e:
            self.img_embed_bn = nn.BatchNorm1d(self.input_encoding_size, track_running_stats=False,
                                               affine=opt.vbn_affine)
            self.embed_bn = nn.BatchNorm1d(self.input_encoding_size, track_running_stats=False,
                                           affine=opt.vbn_affine)
            # self.core_bn = nn.BatchNorm1d(self.rnn_size)

            self.img_embed = torch.nn.Sequential(self.img_embed, self.img_embed_bn)
            self.embed = torch.nn.Sequential(self.embed, self.embed_bn)
            # self.img_embed = torch.nn.Sequential(OrderedDict([
            #     ('img_embed', self.img_embed),
            #     ('img_embed_bn', self.img_embed_bn)
            # ]))
            # self.img_embed = torch.nn.Sequential(OrderedDict([
            #     ('embed', self.embed),
            #     ('embed_bn', self.embed_bn)
            # ]))
            # self.core = torch.nn.Sequential(self.core, self.core_bn)

        self.initialize_params()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def _sample(self, fc_feats, greedy=True):

        # we assume model is on single device
        device = next(self.parameters()).device

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seq_logprobs = fc_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output), dim=1)

            # sample the next_word
            if t == self.seq_length + 1:  # skip if we achieve maximum length
                break
            if greedy:
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)

                # this:
                # it = torch.multinomial(prob_prev, 1).to(device)

                # equals this:
                np_prob_prev = prob_prev.numpy()
                result = []
                for row in np_prob_prev:
                    n_row = row / np.linalg.norm(row, ord=1)
                    sample = np.random.choice(len(n_row), 1, p=n_row)
                    result.append(sample.item())

                it = torch.LongTensor(result).unsqueeze(1).to(device)
                # --- until here, but faster!
                # see https://pytorch.org/docs/stable/torch.html#torch.multinomial
                # https://github.com/pytorch/pytorch/issues/11931
                # https://github.com/pytorch/pytorch/issues/13018

                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it  # seq[t] the input of t+2 time step
                seq_logprobs[:, t - 1] = sample_logprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seq_logprobs
