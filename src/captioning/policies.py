import logging

import torch
from abc import ABC

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy, NetsPolicy, SeedsPolicy
from captioning.rewards import init_scorer, get_self_critical_reward


class GenPolicy(Policy, ABC):

    def rollout(self, data):
        init_scorer(cached_tokens='coco-train-idxs')

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        # tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        tmp = [_ if _ is None else torch.from_numpy(_) for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        gen_result, sample_logprobs = self.policy_net(fc_feats, att_feats, att_masks,
                                                      opt={'sample_max': 0}, mode='sample')

        reward = get_self_critical_reward(self.policy_net, fc_feats, att_feats,
                                          att_masks, data, gen_result)

        return reward

    def accuracy_on(self, dataloader, config, directory):
        assert directory is not None

        # num_batches = config.num_val_batches if config and config.num_val_batches else 0
        num = config.num_val_items

        lang_stats = eval_utils.eval_split(self.policy_net, dataloader.loader, directory, num=num)

        logging.info('******* eval run complete: {} *******'.format(float(lang_stats['CIDEr'])))
        return float(lang_stats['CIDEr'])


class NetsGenPolicy(GenPolicy, NetsPolicy):
    pass


class SeedsGenPolicy(GenPolicy, SeedsPolicy):
    pass
