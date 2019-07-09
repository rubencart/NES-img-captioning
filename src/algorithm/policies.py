import copy
from collections import namedtuple
from enum import Enum
import logging
import os
from abc import ABC

import numpy as np
import torchvision
import torch

from algorithm.nets import PolicyNet
from algorithm.tools.utils import mkdir_p, random_state


class SuppDataset(Enum):
    MNIST = 'mnist'
    MSCOCO = 'mscoco'


class Net(Enum):
    MNIST = 'mnist'
    FC_CAPTION = 'fc_caption'


DATASETS = {
    SuppDataset.MNIST: torchvision.datasets.MNIST,
    SuppDataset.MSCOCO: None
}

_opt_fields = ['net', 'safe_mutations', 'model_options', 'safe_mutation_underflow', 'fitness',
               'vbn', 'safe_mutation_batch_size', 'safe_mutation_vector']
PolicyOptions = namedtuple('PolicyOptions', field_names=_opt_fields,
                           defaults=[None, '', {}, 0.01, None, False, 32, None])

_model_opt_fields = ['vocab_size', 'input_encoding_size', 'rnn_type', 'rnn_size', 'num_layers',
                     'drop_prob_lm', 'seq_length', 'fc_feat_size', 'vbn', 'vbn_e', 'vbn_affine', 'layer_n',
                     'layer_n_affine', 'safe_mutation_underflow', 'safe_mutations', 'safe_mutation_vector',
                     'safe_mutation_batch_size']
ModelOptions = namedtuple('ModelOptions', field_names=_model_opt_fields,
                          defaults=(0,) * len(_model_opt_fields))


class Policy(ABC):
    """
    Abstract class that implements a policy that can be evolved or fitness/acc can be calculated
        Basically a wrapper around the model
    """
    def __init__(self, dataset: SuppDataset, options: PolicyOptions):
        if dataset == SuppDataset.MSCOCO:
            from captioning.policies import Fitness
            self.fitness = Fitness(options.fitness if options.fitness else Fitness.DEFAULT)
        else:
            self.fitness = None

        self.policy_net: PolicyNet = None

        assert isinstance(dataset, SuppDataset)
        self.dataset = dataset
        self.net = Net(options.net)
        self.options = options
        self.vbn = bool(options.vbn)
        self.ref_batch = None
        options.model_options['vbn'] = self.vbn

        self.model_options = ModelOptions(**options.model_options)

        from classification.nets import MnistNet
        from captioning.nets import FCModel

        self.NETS = {
            Net.MNIST: MnistNet,
            Net.FC_CAPTION: FCModel,
        }

        self.init_model(self.generate_model())

    def calc_sensitivity(self, task_id, parent_id, experiences, batch_size, directory):
        """
        :param task_id:
        :param parent_id:
        :param experiences: input batch to calculate the sensitivity on
        :param batch_size:
        :param directory:
        :return:
        """
        assert self.policy_net is not None, 'set model first!'
        self.policy_net.calc_sensitivity(task_id, parent_id, experiences, batch_size, directory)

    def set_ref_batch(self, ref_batch):
        self.ref_batch = ref_batch

    def save(self, path, filename):
        assert self.policy_net is not None, 'set model first!'
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def get_net_class(self):
        return self.NETS[self.net]

    def parameter_vector(self):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.parameter_vector()

    def set_from_parameter_vector(self, vector):
        assert self.policy_net is not None, 'set model first!'
        assert isinstance(vector, np.ndarray) or isinstance(vector, torch.Tensor)
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)
        self.policy_net.set_from_vector(vector)

    def nb_learnable_params(self):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.get_nb_learnable_params()

    def serialized(self, path=''):
        return self.policy_net.serialize(path=path)

    def init_model(self, model=None):
        assert self.policy_net is None
        assert isinstance(model, PolicyNet), '{}'.format(type(model))
        self.policy_net = model

    def set_model(self, model):
        assert self.policy_net is not None
        assert isinstance(model, PolicyNet) or isinstance(model, dict) or \
               isinstance(model, str), '{}'.format(type(model))
        if isinstance(model, PolicyNet):
            self._set_from_statedict_model(model.state_dict())
        elif isinstance(model, dict):
            self._set_from_statedict_model(model)
        elif isinstance(model, str):
            self._set_from_path_model(model)
        else:
            logging.error('Trying to set policy model from invalid input, setting random model instead')
            self._set_from_statedict_model(self.generate_model().state_dict())

    def _set_from_path_model(self, serialized):
        assert self.policy_net is not None, 'Set model first!'
        self.policy_net.from_serialized(serialized)

    def _set_from_statedict_model(self, serialized):
        assert isinstance(serialized, dict), '{}'.format(type(serialized))
        assert self.policy_net is not None, 'Set model first!'

        copied = copy.deepcopy(serialized)
        self.policy_net.load_state_dict(copied)

    def generate_model(self, from_param_file=None, start_rng=None):
        if from_param_file:
            return self.get_net_class()(from_param_file=from_param_file, options=self.model_options, vbn=self.vbn)
        elif start_rng:
            return self.get_net_class()(rng_state=start_rng, options=self.model_options, vbn=self.vbn)
        else:
            return self.get_net_class()(rng_state=random_state(), options=self.model_options, vbn=self.vbn)

    def evolve_model(self, sigma):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.evolve(sigma)

    def get_model(self):
        return self.policy_net

    def calculate_all_sensitivities(self, task_data, loader, directory, batch_size):
        raise NotImplementedError

    def rollout(self, placeholder, data, config):
        raise NotImplementedError

    def accuracy_on(self, dataloader, config, directory) -> float:
        raise NotImplementedError


class PolicyFactory:
    @staticmethod
    def create(dataset: SuppDataset, exp):
        options = PolicyOptions(**exp['policy_options'])

        if dataset == SuppDataset.MNIST:
            from classification.policies import ClfPolicy
            return ClfPolicy(dataset, options)

        elif dataset == SuppDataset.MSCOCO:
            from captioning.policies import CaptPolicy
            return CaptPolicy(dataset, options)
