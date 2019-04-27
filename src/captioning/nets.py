"""
Code from https://github.com/ruotianluo/self-critical.pytorch
"""

from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from .CaptionModel import CaptionModel
from algorithm.nets import PolicyNet


class CaptionModel(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, vbn=False):
        super(CaptionModel, self).__init__(rng_state, from_param_file, grad, vbn)

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _contained_forward(self, data, i=-1, split=100):
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        device = next(self.parameters()).device
        tmp = [_ if _ is None else torch.from_numpy(_).to(device) for _ in tmp]
        self.to(device)
        fc_feats, _, _, _, _ = tmp

        fc_feats = torch.zeros_like(fc_feats).copy_(fc_feats)

        # warning we assume 5 seqs per image
        fc_feats = fc_feats.index_select(dim=0, index=torch.arange(0, fc_feats.size(0), 5))

        if i >= 0:
            fc_feats = fc_feats[i].unsqueeze(0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        xt = self.img_embed(fc_feats)

        output, state = self.core(xt, state)

        it = fc_feats.data.new(batch_size).long().zero_()
        xt = self.embed(it)

        output, state = self.core(xt, state)
        # |batch_size| x |vocab size|
        logprobs = F.log_softmax(self.logit(output), dim=1)

        extended_lp = torch.cat((logprobs, torch.zeros((batch_size, 10**4 - logprobs.size(1)))), 1)

        # print('extended_lp ', extended_lp.size())

        split_lp = extended_lp.split(split, dim=1)

        # print('split_lp ', split_lp.size())

        summed_lp = torch.stack(split_lp).sum(2).permute(1, 0)
        # todo del the rest
        return summed_lp

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q, ix[q, c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_unaug_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        bdash = beam_size // group_size  # beam per group

        device = next(self.parameters()).device

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = [_.chunk(group_size) if _ is not None else [None] * group_size for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if decoding_constraint and t - divm > 0:
                        # logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).cuda(), float('-inf'))
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).to(device), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm], \
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t - divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t - divm + 1)
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time

                    it = beam_seq_table[divm][t - divm]
                    # logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(
                    #             args[divm] + [state_table[divm]]))
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.to(device), *(
                            args[divm] + [state_table[divm]]))

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]

        # todo reduce
        done_beams = reduce(lambda a, b: a + b, done_beams_table)
        return done_beams


class LSTMCore(nn.Module):
    def __init__(self, opt, vbn=False):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        # self.drop_prob_lm = opt.drop_prob_lm

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        # self.dropout = nn.Dropout(self.drop_prob_lm)

        # virtual batch normalization
        self.vbn = vbn
        if self.vbn:
            self.i2h_bn = nn.BatchNorm1d(5 * self.rnn_size)
            self.h2h_bn = nn.BatchNorm1d(5 * self.rnn_size)
            self.c_bn = nn.BatchNorm1d(self.rnn_size)

    def forward(self, xt, state):
        xt_i2h = self.i2h_bn(self.i2h(xt)) if self.vbn else self.i2h(xt)
        state_h2h = self.h2h_bn(self.h2h(state[0][-1])) if self.vbn else self.h2h(state[0][-1])

        all_input_sums = xt_i2h + state_h2h
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)

        # sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        # todo ????????
        in_transform = torch.max(
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size)
        )

        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        activated_next_c = torch.tanh(self.c_bn(next_c)) if self.vbn else torch.tanh(next_c)
        # next_h = out_gate * F.tanh(next_c)
        next_h = out_gate * activated_next_c

        # output = self.dropout(next_h)
        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class FCModel(CaptionModel):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
        super(FCModel, self).__init__(rng_state, from_param_file, grad, vbn)
        opt = options

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0  # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        # virtual batch normalization
        self.vbn = opt.vbn
        if self.vbn:
            self.img_embed_bn = nn.BatchNorm1d(self.input_encoding_size, track_running_stats=False)
            self.embed_bn = nn.BatchNorm1d(self.input_encoding_size, track_running_stats=False)
            # self.core_bn = nn.BatchNorm1d(self.rnn_size)

            self.img_embed = torch.nn.Sequential(self.img_embed, self.img_embed_bn)
            self.embed = torch.nn.Sequential(self.embed, self.embed_bn)
            # self.core = torch.nn.Sequential(self.core, self.core_bn)

        self._initialize_params()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.embed.weight.data.uniform_(-initrange, initrange)
    #     self.logit.bias.data.fill_(0)
    #     self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:  # otherwise no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind))
                        # fetch prev distribution: shape Nx(M+1)
                        # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)

                        # hopefully this:
                        # t_sample = torch.multinomial(prob_prev, 1).view(-1)

                        # equals this:
                        np_prob_prev = prob_prev.numpy()
                        result = []
                        for row in np_prob_prev:
                            n_row = row / np.linalg.norm(row, ord=1)
                            sample = np.random.choice(len(n_row), 1, p=n_row)
                            result.append(sample.item())

                        t_sample = torch.LongTensor(result)
                        # ---- until here :)
                        # see https://pytorch.org/docs/stable/torch.html#torch.multinomial
                        # https://github.com/pytorch/pytorch/issues/11931
                        # https://github.com/pytorch/pytorch/issues/13018

                        sample = t_sample.index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample)
                else:
                    it = seq[:, i - 1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i - 1].sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt, state)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' is contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # lets assume this for now, otherwise this corner case causes a few headaches down the road.
        # can be dealt with in future if needed
        assert beam_size <= self.vocab_size + 1

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats=None, att_masks=None, opt={}):
        # sample_max = 1 --> greedy? 0 --> random?
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        # device = opt.get('cuda', False)
        # Todo we assume model is on single device
        device = next(self.parameters()).device

        if beam_size > 1 and att_feats is not None:
            return self._sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
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
            if sample_max:
                # sample_max --> greedy
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                # it = torch.multinomial(prob_prev, 1).cuda()

                # hopefully this:
                # it = torch.multinomial(prob_prev, 1).to(device)

                # equals this:
                np_prob_prev = prob_prev.numpy()
                result = []
                for row in np_prob_prev:
                    n_row = row / np.linalg.norm(row, ord=1)
                    sample = np.random.choice(len(n_row), 1, p=n_row)
                    result.append(sample.item())

                it = torch.LongTensor(result).unsqueeze(1).to(device)
                # --- until here
                # see https://pytorch.org/docs/stable/torch.html#torch.multinomial
                # https://github.com/pytorch/pytorch/issues/11931
                # https://github.com/pytorch/pytorch/issues/13018

                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it  # seq[t] the input of t+2 time step
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs
