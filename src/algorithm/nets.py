"""
contains code from https://github.com/uber-research/safemutations
and https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
"""
import logging
from enum import Enum
from abc import ABC

import torch
from torch import nn

from algorithm.safe_mutations import Sensitivity
from algorithm.tools.utils import random_state


class Mutation(Enum):
    SAFE_GRAD_SUM = 'SM-G-SUM'
    SAFE_GRAD_ABS = 'SM-G-ABS'
    SAFE_VECTOR = 'SM-VECTOR'
    SAFE_PROPORTIONAL = 'SM-PROPORTIONAL'
    DEFAULT = ''


class PolicyNet(nn.Module, ABC):
    """
        Abstract base class for all networks that are evolved. Implements initialization and
            evolve (mutate) methods.
    """

    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        self.from_param_file = from_param_file

        if rng_state:
            torch.manual_seed(rng_state)

        self.nb_learnable_params = 0
        self.nb_params = 0
        self.grad = grad
        self.vbn = vbn

        self.eval()

        self.mutations = Mutation((options and options.safe_mutations) or '')
        if self.mutations != Mutation.DEFAULT:
            self.sensitivity_wrapper = Sensitivity(self, options.safe_mutation_underflow, self.mutations)
            if self.mutations == Mutation.SAFE_VECTOR:
                self.set_sensitivity_vector(options.safe_mutation_vector)

    def initialize_params(self):
        """
            Call at end of __init__ method of all subclasses! Layers and params need to be defined
            before this method makes sense.
        """

        if self.from_param_file:
            logging.info('From param file!')
            self.load_state_dict(torch.load(self.from_param_file, map_location='cpu'))

        for name, param in self.named_parameters():
            # exclude batch norm layers from xavier initialization
            if not self.from_param_file and 'bn' not in name and 'ln' not in name and '1' not in name:
                if 'weight' in name:
                    # nn.init.kaiming_normal_(tensor)
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    param.data.zero_()

            logging.info('Params: %s, %s, %s', name, param.size(), param.data.abs().mean())

        self.nb_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params = self.count_parameters()

        logging.warning('Number of learnable params: {}'.format(self.nb_learnable_params))

        # freeze all params (gradientwise)
        if not self.grad:
            for param in self.parameters():
                param.requires_grad = False

    def evolve(self, sigma, rng_state=None):
        """
        Evolve (mutate) the current params one step. Can either be (depending on mutation type
            set in self.mutations): standard mutation, proportional mutation, safe mutation by
            output grad sensitivity, or by a precalculated vector of sensitivities.
        :param  sigma: mutation power
        :param  rng_state: optionally provide a random seed
        :return delta: the mutation vector
        """
        rng = rng_state if rng_state is not None else random_state()
        torch.manual_seed(rng)

        safe, proportional = False, False
        if self.mutations in [Mutation.SAFE_GRAD_SUM, Mutation.SAFE_GRAD_ABS, Mutation.SAFE_VECTOR]:
            safe = True
        elif self.mutations == Mutation.SAFE_PROPORTIONAL:
            proportional = True

        param_vector = nn.utils.parameters_to_vector(self.parameters())
        noise = torch.empty_like(param_vector, requires_grad=False).normal_(mean=0.0, std=sigma)

        if safe:
            noise /= self.sensitivity_wrapper.get_sensitivity()
            # logging.info('new noise: %s', noise)
        if proportional:
            params = torch.empty_like(param_vector).copy_(param_vector)
            mean = params.abs().mean()
            params[params == 0.0] = mean
            noise *= params.abs()

        new_param_vector = param_vector + noise
        self.set_from_vector(new_param_vector)

        for param in self.parameters():
            param.requires_grad = False

        return torch.empty_like(noise).copy_(noise).numpy()

    def calc_sensitivity(self, task_id, parent_id, experiences, batch_size, directory):
        if self.mutations in [Mutation.SAFE_GRAD_SUM, Mutation.SAFE_GRAD_ABS]:
            self.sensitivity_wrapper.calc_sensitivity(task_id, parent_id, experiences, batch_size, directory)

    def set_sensitivity_vector(self, vector):
        self.sensitivity_wrapper.set_sensitivity(vector)

    def get_sensitivity_vector(self):
        return self.sensitivity_wrapper.get_sensitivity()

    def extract_grad(self):
        """
            Used for sensitivity calculation. Pytorch stores gradients in graph nodes, that means, at
            the resp. parameters. This method extracts all of the gradients and puts them in a 1D vector.
            Caution: order is important here! The order of self.parameters() is used.
        """
        tot_size = self.nb_params
        pvec = torch.zeros(tot_size, dtype=torch.float)
        count = 0
        for param in self.parameters():
            sz = param.grad.data.flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.flatten()
            count += sz
        return pvec.clone().detach()  # , params

    def count_parameters(self):
        count = nn.utils.parameters_to_vector(self.parameters()).size(0)
        return count

    def get_nb_learnable_params(self):
        return self.nb_learnable_params

    def serialize(self, path=''):
        torch.save(self.state_dict(), path)
        return path

    def from_serialized(self, serialized):
        # todo map_location not cpu if on gpu!
        state_dict = torch.load(serialized, map_location='cpu')
        self.from_param_file = serialized
        self.load_state_dict(state_dict)

    def set_from_vector(self, vector):
        assert len(vector) == self.nb_learnable_params
        nn.utils.vector_to_parameters(vector, self.parameters())

    def parameter_vector(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def forward_for_sensitivity(self, x, orig_bs=0, i=-1):
        raise NotImplementedError
