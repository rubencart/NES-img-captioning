import logging

import torch
from abc import ABC

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy, NetsPolicy, SeedsPolicy
from captioning.rewards import init_scorer, get_self_critical_reward, RewardCriterion

logger = logging.getLogger(__name__)


class GenPolicy(Policy, ABC):

    def rollout(self, data, config):
        torch.set_grad_enabled(False)

        init_scorer(cached_tokens='coco-train-idxs')

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')
        # logger.info('***** DEVICE : {} *****'.format(device))

        self.policy_net.to(device)

        tmp = [_ if _ is None else torch.from_numpy(_).to(device) for _ in tmp]
        # tmp = [_ if _ is None else torch.from_numpy(_) for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        # logger.info('evaluating {} images'.format(labels.size()))

        gen_result, sample_logprobs = self.policy_net(fc_feats, att_feats, att_masks,
                                                      opt={'sample_max': 0}, mode='sample')

        reward, rewards, scores = get_self_critical_reward(self.policy_net, fc_feats, att_feats,
                                                           att_masks, data, gen_result)

        # rl_crit = RewardCriterion()

        # device = next(data).device
        # loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().to(device))
        # loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float())

        # return loss.item()
        return reward * 100  # scores.sum() / (fc_feats.size(0) / 256)

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
