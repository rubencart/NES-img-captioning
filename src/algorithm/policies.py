import copy
from enum import Enum
import logging
import os
from abc import ABC
from typing import Union

import torchvision
import torch
from torch import nn

<<<<<<< HEAD:src/algorithm/policies.py
from algorithm.nets import CompressedModel
from captioning.nets import FCModel
from classification.nets import PolicyNet, Cifar10Net, MnistNet, Cifar10Net_Small
from algorithm.tools.utils import mkdir_p, random_state
from classification.policies import SeedsClfPolicy, NetsClfPolicy
=======
from algorithm.nets import CompressedModel, PolicyNet
# from captioning.nets import FCModel
# from classification.nets import Cifar10Net  # , Cifar10Net, MnistNet, Cifar10Net_Small
from algorithm.tools.utils import mkdir_p, random_state
# from classification.policies import SeedsClfPolicy, NetsClfPolicy
>>>>>>> d30c45c29de3208621bfa4689d7ea024ac190c45:src/algorithm/policies.py

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


class Policy(ABC):
    def __init__(self, dataset: SuppDataset, net: Net):
        self.policy_net = None
        self.serial_net = None

        assert isinstance(dataset, SuppDataset)
        self.dataset = dataset
        self.net = net

        from classification.nets import Cifar10Net, Cifar10Net_Small, MnistNet
        from captioning.nets import FCModel

        self.NETS = {
            Net.CIFAR10: Cifar10Net,
            Net.CIFAR10_SMALL: Cifar10Net_Small,
            Net.MNIST: MnistNet,
            Net.FC_CAPTION: FCModel,
        }

    def save(self, path, filename):
        # todo! also save serial?
        assert self.policy_net is not None, 'set model first!'
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def get_net_class(self):
        return self.NETS[self.net]

    def parameter_vector(self):
        assert self.policy_net is not None, 'set model first!'
        return nn.utils.parameters_to_vector(self.policy_net.parameters())

    def get_serial_model(self):
        return copy.deepcopy(self.serial_net)

    def nb_learnable_params(self):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.get_nb_learnable_params()

    def from_serialized(self, serialized: Union[CompressedModel, dict]):
        model = self.generate_model()
        model.from_serialized(serialized)
        return model

    def rollout(self, data):
        raise NotImplementedError

    def accuracy_on(self, data):
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


class NetsPolicy(Policy, ABC):

    def init_model(self, model=None):
        assert isinstance(model, PolicyNet), '{}'.format(type(model))
        self.policy_net = model
        self.serial_net = model.state_dict()

    def set_model(self, model):
        assert isinstance(model, PolicyNet) or isinstance(model, dict), '{}'.format(type(model))
        if isinstance(model, PolicyNet):
            self._set_serialized_net_model(model.state_dict())
        else:
            self._set_serialized_net_model(model)

    def _set_serialized_net_model(self, serialized):
        assert isinstance(serialized, dict), '{}'.format(type(serialized))
        assert self.policy_net is not None, 'Set model first!'

        copied = copy.deepcopy(serialized)
        self.policy_net.load_state_dict(copied)
        self.serial_net = copied

    def generate_model(self, from_param_file=None, start_rng=None):
        if from_param_file:
            return self.get_net_class()(from_param_file=from_param_file)  # .state_dict()
        elif start_rng:
            return self.get_net_class()(rng_state=start_rng)  # .state_dict()
        else:
            return self.get_net_class()(rng_state=random_state())  # .state_dict()

    def evolve_model(self, sigma):
        assert self.policy_net is not None, 'set model first!'
        self.policy_net.evolve(sigma)
        self.serial_net = copy.deepcopy(self.policy_net.state_dict())

    def get_model(self):
        return self.policy_net


class SeedsPolicy(Policy, ABC):

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
    def create(dataset: SuppDataset, mode, net: Net):
        from classification.policies import SeedsClfPolicy, NetsClfPolicy

        if dataset == SuppDataset.MNIST or dataset == SuppDataset.CIFAR10:
            if mode == 'seeds':
                return SeedsClfPolicy(dataset, net)
            else:
                return NetsClfPolicy(dataset, net)
