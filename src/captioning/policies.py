
import random
import sys
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch

import captioning.eval_utils as eval_utils

from algorithm.policies import Policy
from algorithm.tools.utils import Config, array_to_str
from captioning.dataloader import DataLoader
from captioning.fitness import LogFitnessCriterion, \
    ExpFitnessCriterion, AltLogFitnessCriterion, LinFitnessCriterion, AvgLogFitnessCriterion

sys.path.append('cider')
from pyciderevalcap.ciderD.ciderD import CiderD


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
    GR_AVGPROB = 'greedy_avgprob'
    DEFAULT = GREEDY

    @classmethod
    def needs_criterion(cls, fitness):
        return fitness in (cls.SC_LOSS, cls.GR_LOGPROB, cls.GR_EXPPROB, cls.GR_LINPROB, cls.GR_AVGPROB)

    @classmethod
    def is_self_critical(cls, fitness):
        return fitness in (cls.SC_LOSS, cls.SELF_CRITICAL)

    @classmethod
    def is_greedy(cls, fitness):
        return fitness in (cls.GR_LINPROB, cls.GR_EXPPROB, cls.GR_LOGPROB, cls.GREEDY, cls.GR_AVGPROB)

    @classmethod
    def get_criterium(cls, fitness):
        assert cls.needs_criterion(fitness)
        if fitness == Fitness.SC_LOSS:
            return LogFitnessCriterion()
        elif fitness == Fitness.GR_LOGPROB:
            return AltLogFitnessCriterion()
        elif fitness == Fitness.GR_EXPPROB:
            return ExpFitnessCriterion()
        elif fitness == Fitness.GR_AVGPROB:
            return AvgLogFitnessCriterion()
        else:
            return LinFitnessCriterion()


class CaptPolicy(Policy):
    """
    Img captioning policy
    """

    def __init__(self, dataset, options):
        super().__init__(dataset, options)
        # takes a while so we make sure we only have to do this once for every worker
        self.cider_scorer = CiderD(df='coco-train-idxs')

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
        cider, ciders = self.compute_ciders(self.policy_net, fc_feats, data, gen_result, self_critical)

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
        lang_stats, _ = eval_utils.eval_split(self.policy_net, dataloader.loader, directory, num=num)

        return float(lang_stats['CIDEr'])

    def compute_ciders(self, model, fc_feats, data, gen_result, self_critical):
        """
        Compute the reward of the generated sequences in gen_result. If necessary,
            also generate sequences by greedy decoding (to baseline when self-critical fitness is used).
        """

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

        score, scores = self.cider_scorer.compute_score(gts, res_)

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

