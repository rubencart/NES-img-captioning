import copy
from collections import namedtuple
from enum import Enum
import logging
import os
from abc import ABC

import numpy as np
import torchvision
import torch
from torch import nn

from algorithm.nets import CompressedModel, PolicyNet
from algorithm.tools.utils import mkdir_p, random_state

logger = logging.getLogger(__name__)


class SuppDataset(Enum):
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'
    MSCOCO = 'mscoco'


class Net(Enum):
    CIFAR10 = 'cifar10'
    CIFAR10_SMALL = 'cifar10_small'
    MNIST = 'mnist'
    FC_CAPTION = 'fc_caption'


DATASETS = {
    SuppDataset.CIFAR10: torchvision.datasets.CIFAR10,
    SuppDataset.MNIST: torchvision.datasets.MNIST,
    SuppDataset.MSCOCO: None  # todo
}


class Mutation(Enum):
    SAFE_GRAD_SUM = 'SM-G-SUM'
    SAFE_GRAD_ABS = 'SM-G-ABS'
    SAFE_VECTOR = 'SM-VECTOR'
    SAFE_PROPORTIONAL = 'SM-PROPORTIONAL'
    DEFAULT = ''


_opt_fields = ['net', 'safe_mutations', 'model_options', 'safe_mutation_underflow', 'fitness',
               'vbn', 'safe_mutation_batch_size', 'safe_mutation_vector']
PolicyOptions = namedtuple('PolicyOptions', field_names=_opt_fields,
                           defaults=[None, '', None, 0.01, None, False, 32, None])

_model_opt_fields = ['vocab_size', 'input_encoding_size', 'rnn_type', 'rnn_size', 'num_layers',
                     # todo dropout can go
                     'drop_prob_lm', 'seq_length', 'fc_feat_size', 'vbn', 'vbn_e', 'vbn_affine', 'layer_n',
                     'layer_n_affine']
ModelOptions = namedtuple('ModelOptions', field_names=_model_opt_fields,
                          defaults=(None,) * len(_model_opt_fields))


class Policy(ABC):
    def __init__(self, dataset: SuppDataset, options: PolicyOptions):
        if dataset == SuppDataset.MSCOCO:
            self.model_options = ModelOptions(**options.model_options)

            from captioning.policies import Fitness
            self.fitness = Fitness(options.fitness if options.fitness else Fitness.DEFAULT)
        else:
            self.model_options = None
            self.fitness = None

        self.policy_net: PolicyNet = None
        self.serial_net = None

        assert isinstance(dataset, SuppDataset)
        self.dataset = dataset
        self.net = Net(options.net)
        self.options = options
        self.vbn = bool(options.vbn)
        self.ref_batch = None
        self.mutations = Mutation(options.safe_mutations)
        # if self.mutations == Mutation.SAFE_PARAM_TYPE:
        #     self.policy_net.set_sensitivity_vector(options.safe_mutation_vector)

        from classification.nets import Cifar10Net, Cifar10Net_Small, MnistNet
        from captioning.nets import FCModel

        self.NETS = {
            Net.CIFAR10: Cifar10Net,
            Net.CIFAR10_SMALL: Cifar10Net_Small,
            Net.MNIST: MnistNet,
            Net.FC_CAPTION: FCModel,
        }

        self.init_model(self.generate_model())

    def calc_sensitivity(self, task_id, parent_id, experiences, batch_size, directory):
        assert self.policy_net is not None, 'set model first!'
        if self.mutations == Mutation.SAFE_VECTOR and \
                self.policy_net.get_sensitivity_vector() is None:
            self.policy_net.set_sensitivity_vector(self.options.safe_mutation_vector)

        elif self.mutations == Mutation.SAFE_GRAD_SUM or self.mutations == Mutation.SAFE_GRAD_ABS:
            underflow = self.options.safe_mutation_underflow
            self.policy_net.calc_sensitivity(task_id, parent_id, experiences, batch_size, directory,
                                             underflow, self.mutations)

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

    # def get_serial_model(self):
    #     return copy.deepcopy(self.serial_net)

    def nb_learnable_params(self):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.get_nb_learnable_params()

    # def from_serialized(self, serialized):
    #     model = self.generate_model()
    #     model.from_serialized(serialized)
    #     return model

    def serialized(self, path=''):
        return self.policy_net.serialize(path=path)

    def calculate_all_sensitivities(self, task_data, loader, directory, batch_size):
        raise NotImplementedError

    def rollout(self, placeholder, data, config) -> float:
        raise NotImplementedError

    def accuracy_on(self, data, config, directory) -> float:
        raise NotImplementedError

    def evolve_model(self, sigma):
        raise NotImplementedError

    def set_model(self, model):
        raise NotImplementedError

    def generate_model(self, from_param_file=None, start_rng=None):
        raise NotImplementedError

    def init_model(self, model=None):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    # def save_to_disk(self, log_dir):
    #     dir = 'tmp_offspring'
    #     filename = 'offspring_params_i{i}.pth'
    #     path_to_offspring = os.path.join(log_dir, dir, filename)
    #
    #     torch.save(self._po.serialize(), path_to_elite)
    #     return path_to_elite


class NetsPolicy(Policy, ABC):
    # TODO inconsistent: init model from PolicyNet but afterwards only with paths --> unify
    # TODO cleanup serial_net & policy_net

    def init_model(self, model=None):
        assert self.policy_net is None
        assert isinstance(model, PolicyNet), '{}'.format(type(model))
        self.policy_net = model
        # self.serial_net = model.state_dict()

    def set_model(self, model):
        assert self.policy_net is not None
        assert isinstance(model, PolicyNet) or isinstance(model, dict) \
               or isinstance(model, str) or model is None, '{}'.format(type(model))
        if isinstance(model, PolicyNet):
            self._set_from_statedict_model(model.state_dict())
        elif isinstance(model, dict):
            self._set_from_statedict_model(model)
        elif isinstance(model, str):
            self._set_from_path_model(model)
        else:
            logger.error('[netspolicy] trying to set policy from None')
            self._set_from_statedict_model(self.generate_model().state_dict())

    def _set_from_path_model(self, serialized):
        # assert isinstance(serialized, dict), '{}'.format(type(serialized))
        assert self.policy_net is not None, 'Set model first!'

        # copied = copy.deepcopy(serialized)
        self.policy_net.from_serialized(serialized)
        # self.serial_net = serialized

    def _set_from_statedict_model(self, serialized):
        assert isinstance(serialized, dict), '{}'.format(type(serialized))
        assert self.policy_net is not None, 'Set model first!'

        copied = copy.deepcopy(serialized)
        self.policy_net.load_state_dict(copied)
        # self.serial_net = copied

    def generate_model(self, from_param_file=None, start_rng=None):
        if from_param_file:
            return self.get_net_class()(from_param_file=from_param_file, options=self.model_options, vbn=self.vbn)
        elif start_rng:
            return self.get_net_class()(rng_state=start_rng, options=self.model_options, vbn=self.vbn)
        else:
            return self.get_net_class()(rng_state=random_state(), options=self.model_options, vbn=self.vbn)

    def evolve_model(self, sigma):
        assert self.policy_net is not None, 'set model first!'
        if self.mutations in [Mutation.SAFE_GRAD_SUM, Mutation.SAFE_GRAD_ABS, Mutation.SAFE_VECTOR]:
            return self.policy_net.evolve(sigma, safe=True)
        elif self.mutations == Mutation.SAFE_PROPORTIONAL:
            return self.policy_net.evolve(sigma, proportional=True)
        else:
            return self.policy_net.evolve(sigma)

    def get_model(self):
        return self.policy_net


class SeedsPolicy(Policy, ABC):
    # TODO REMOVE

    def init_model(self, model=None):
        self.set_model(model if model else CompressedModel())

    def set_model(self, model):
        assert isinstance(model, CompressedModel)
        self._set_compr_model(model)

    def generate_model(self, from_param_file=None, start_rng=None):
        # todo other rng?
        return CompressedModel(from_param_file=from_param_file,
                               start_rng=start_rng)

    def evolve_model(self, sigma):
        self.serial_net.evolve(sigma)
        self._set_compr_model(self.serial_net)

    def get_model(self):
        return self.serial_net

    def _set_compr_model(self, compressed_model):
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=self.get_net_class())
        assert isinstance(uncompressed_model, PolicyNet)
        self.policy_net = uncompressed_model
        self.serial_net = copy.deepcopy(compressed_model)


class PolicyFactory:
    @staticmethod
    def create(dataset: SuppDataset, mode, exp):
        options = PolicyOptions(**exp['policy_options'])

        if dataset == SuppDataset.MNIST or dataset == SuppDataset.CIFAR10:
            from classification.policies import SeedsClfPolicy, NetsClfPolicy
            if mode == 'seeds':
                return SeedsClfPolicy(dataset, options)
            else:
                return NetsClfPolicy(dataset, options)

        elif dataset == SuppDataset.MSCOCO:
            from captioning.policies import NetsGenPolicy, SeedsGenPolicy
            if mode == 'seeds':
                return SeedsGenPolicy(dataset, options)
            else:
                return NetsGenPolicy(dataset, options)
