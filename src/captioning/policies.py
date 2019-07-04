import logging
import random
import time
from enum import Enum

import torch
from abc import ABC

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy, NetsPolicy, SeedsPolicy
from captioning.rewards import init_scorer, get_self_critical_reward, RewardCriterion, \
    GreedyExpRewardCriterion, GreedyLogRewardCriterion, GreedyLinRewardCriterion

logger = logging.getLogger(__name__)


class Fitness(Enum):
    # todo beam?
    SAMPLE = 'sample'
    GREEDY = 'greedy'
    SELF_CRITICAL = 'self_critical'
    SC_LOSS = 'sc_loss'
    GR_LOGPROB = 'greedy_logprob'
    GR_EXPPROB = 'greedy_expprob'
    GR_LINPROB = 'greedy_linprob'
    DEFAULT = GREEDY

    @classmethod
    def needs_criterion(cls, fitness):
        return fitness in (cls.SC_LOSS, cls.GR_LOGPROB, cls.GR_EXPPROB, cls.GR_LINPROB)

    @classmethod
    def is_self_critical(cls, fitness):
        return fitness in (cls.SC_LOSS, cls.SELF_CRITICAL)

    @classmethod
    def is_greedy(cls, fitness):
        return fitness in (cls.GR_LINPROB, cls.GR_EXPPROB, cls.GR_LOGPROB, cls.GREEDY)

    @classmethod
    def get_criterium(cls, fitness):
        assert cls.needs_criterion(fitness)
        if fitness == Fitness.SC_LOSS:
            return RewardCriterion()
        elif fitness == Fitness.GR_LOGPROB:
            return GreedyLogRewardCriterion()
        elif fitness == Fitness.GR_EXPPROB:
            return GreedyExpRewardCriterion()
        else:
            return GreedyLinRewardCriterion()


class GenPolicy(Policy, ABC):

    def calculate_all_sensitivities(self, task_data, loader, directory, batch_size):
        index = random.randrange(len(task_data.parents))
        parent_id, parent = task_data.parents[index]
        self.set_model(parent)

        for i, data in enumerate(loader):
            # torch.set_grad_enabled(False)
            # tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            # tmp = [_ if _ is None else torch.from_numpy(_) for _ in tmp]
            # fc_feats, att_feats, labels, masks, att_masks = tmp

            self.policy_net.calc_sensitivity(i, 0, data, batch_size, directory)

    def rollout(self, placeholder, data, config):

        torch.set_grad_enabled(False)

        init_scorer(cached_tokens='coco-train-idxs')

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')

        self.policy_net.to(device)

        tmp = [_ if _ is None else torch.from_numpy(_).to(device) for _ in tmp]
        # tmp = [_ if _ is None else torch.from_numpy(_) for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        sample_max = 1 if Fitness.is_greedy(self.fitness) else 0

        # virtual batch norm
        if self.vbn:
            self.policy_net.train()
            # todo will this work with ref_batch being dict with 'fc_feats',...
            self.policy_net(torch.empty_like(self.ref_batch).copy_(self.ref_batch),
                            opt={'sample_max': sample_max}, mode='sample')

        self.policy_net.eval()

        gen_result, sample_logprobs = self.policy_net(fc_feats, att_feats, att_masks,
                                                      opt={'sample_max': sample_max}, mode='sample')

        # logging.warning('logprobs: %s', sample_logprobs.mean())
        # time.sleep(1000)
        self_critical = Fitness.is_self_critical(self.fitness)
        reward, rewards = get_self_critical_reward(self.policy_net, fc_feats, att_feats,
                                                   att_masks, data, gen_result, self_critical)

        if Fitness.needs_criterion(self.fitness):
            # todo change name loss (because we actually use - loss, to maximize)
            crit = Fitness.get_criterium(self.fitness)
            loss = crit(sample_logprobs.data, gen_result.data, torch.from_numpy(rewards).float().to(device))
            result = float(loss.item())
            del loss, crit
        else:
            result = float(reward * 100)

        del reward, rewards, gen_result, sample_logprobs, fc_feats, att_feats, labels, masks, att_masks, tmp,
        return result

    def accuracy_on(self, dataloader, config, directory) -> float:
        assert directory is not None

        torch.set_grad_enabled(False)

        # num_batches = config.num_val_batches if config and config.num_val_batches else 0
        num = config.num_val_items

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')
        # logger.info('***** DEVICE : {} *****'.format(device))

        self.policy_net.to(device)

        lang_stats = eval_utils.eval_split(self.policy_net, dataloader.loader, directory, num=num)

        # logging.info('CIDEr: {}'.format(float(lang_stats['CIDEr'])))
        return float(lang_stats['CIDEr'])


class NetsGenPolicy(GenPolicy, NetsPolicy):
    pass


class SeedsGenPolicy(GenPolicy, SeedsPolicy):
    pass
