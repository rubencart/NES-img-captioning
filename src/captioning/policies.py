
import logging
import random
from enum import Enum

import torch

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy
from algorithm.tools.utils import Config
from captioning.dataloader import DataLoader
from captioning.fitness import compute_ciders, LogFitnessCriterion, \
    ExpFitnessCriterion, AltLogFitnessCriterion, LinFitnessCriterion

logger = logging.getLogger(__name__)


class Fitness(Enum):
    """
    Enum of different supported fitness functions
    """

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
            return LogFitnessCriterion()
        elif fitness == Fitness.GR_LOGPROB:
            return AltLogFitnessCriterion()
        elif fitness == Fitness.GR_EXPPROB:
            return ExpFitnessCriterion()
        else:
            return LinFitnessCriterion()


class CaptPolicy(Policy):
    """
    Img captioning policy
    """

    def calculate_all_sensitivities(self, task_data, loader, directory, batch_size):
        """
        Method that calculates a sensitivity vector for the entire dataloader, per batch, and saves
            it to the offspring dir.
        """
        index = random.randrange(len(task_data.parents))
        parent_id, parent = task_data.parents[index]
        self.set_model(parent)

        for i, data in enumerate(loader):
            self.policy_net.calc_sensitivity(i, 0, data, batch_size, directory)

    def rollout(self, placeholder: torch.Tensor, data: dict, config: Config) -> float:
        """
        Compute the fitness of this individual (this is a subclass of Policy so represents an individual,
            the model etc is kept in superclass Policy). Which fitness is calculated depends on the setting.
        :param placeholder: an empty tensor that is reused in an attempt to solve a memory issue
                            so we don't have to instantiate a new tensor for every batch
        :param data:    batch
        :param config:  experiment wide config setting
        :return: a float value
        """

        torch.set_grad_enabled(False)
        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')
        self.policy_net.to(device)

        fc_feats = torch.from_numpy(data['fc_feats']).to(device)

        # virtual batch norm
        if self.vbn:
            self.policy_net.train()
            # forward pass
            self.policy_net(torch.empty_like(self.ref_batch).copy_(self.ref_batch),
                            greedy=Fitness.is_greedy(self.fitness))

        self.policy_net.eval()

        # forward pass
        gen_result, sample_logprobs = self.policy_net(fc_feats,
                                                      greedy=Fitness.is_greedy(self.fitness))

        self_critical = Fitness.is_self_critical(self.fitness)
        cider, ciders = compute_ciders(self.policy_net, fc_feats, data, gen_result, self_critical)

        if Fitness.needs_criterion(self.fitness):
            crit = Fitness.get_criterium(self.fitness)
            reward = crit(sample_logprobs.data, gen_result.data, torch.from_numpy(ciders).float().to(device))
            result = float(reward.item())
            del reward, crit
        else:
            result = float(cider * 100)

        del cider, ciders, gen_result, sample_logprobs, fc_feats,
        return result

    def accuracy_on(self, dataloader: DataLoader, config: Config, directory: str) -> float:
        """
        Calculate the accuracy on dataloader (should be validation set)
        """
        assert directory is not None
        torch.set_grad_enabled(False)

        num = config.num_val_items
        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')

        self.policy_net.to(device)
        lang_stats = eval_utils.eval_split(self.policy_net, dataloader.loader, directory, num=num)

        return float(lang_stats['CIDEr'])

