import logging
from enum import Enum

import torch
from abc import ABC

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy, NetsPolicy, SeedsPolicy
from captioning.rewards import init_scorer, get_self_critical_reward, RewardCriterion

logger = logging.getLogger(__name__)


class Fitness(Enum):
    SAMPLE = 'sample'
    SELF_CRITICAL = 'self_critical'
    SC_LOSS = 'sc_loss'


class GenPolicy(Policy, ABC):

    def rollout(self, placeholder, data, config):
        # cuda = options.get('cuda', False)
        # fitness = options.get('fitness', Fitness.SC_LOSS)
        fitness = self.options.fitness
        logger.warning('fitness: %s, %s', fitness, str(fitness != Fitness.SAMPLE))
        self_critical = fitness != Fitness.SAMPLE

        torch.set_grad_enabled(False)

        init_scorer(cached_tokens='coco-train-idxs')

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')

        self.policy_net.to(device)

        tmp = [_ if _ is None else torch.from_numpy(_).to(device) for _ in tmp]
        # tmp = [_ if _ is None else torch.from_numpy(_) for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        gen_result, sample_logprobs = self.policy_net(fc_feats, att_feats, att_masks,
                                                      opt={'sample_max': 0}, mode='sample')

        reward, rewards = get_self_critical_reward(self.policy_net, fc_feats, att_feats,
                                                   att_masks, data, gen_result, self_critical)

        if fitness == Fitness.SC_LOSS:
            rl_crit = RewardCriterion()

            # device = next(data).device
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().to(device))
            # loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float())
            result = float(loss.detach().item())
            del loss, rl_crit
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

        # logging.info('******* eval run complete: {} *******'.format(float(lang_stats['CIDEr'])))
        return float(lang_stats['CIDEr'])


class NetsGenPolicy(GenPolicy, NetsPolicy):
    pass


class SeedsGenPolicy(GenPolicy, SeedsPolicy):
    pass
