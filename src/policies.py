import copy
from enum import Enum
import logging
import os
from abc import ABC

import torchvision
import torch
from torch import nn

from nets import PolicyNet, CompressedModel, random_state, Cifar10Net, MnistNet, Cifar10Net_Small
from utils import mkdir_p

logger = logging.getLogger(__name__)


class SuppDataset(Enum):
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'


class Net(Enum):
    CIFAR10 = 'cifar10'
    CIFAR10_SMALL = 'cifar10_small'
    MNIST = 'mnist'


DATASETS = {
    SuppDataset.CIFAR10: torchvision.datasets.CIFAR10,
    SuppDataset.MNIST: torchvision.datasets.MNIST,
}


class Policy(ABC):
    def __init__(self, dataset: SuppDataset, net: Net):
        self.policy_net = None
        self.serial_net = None

        self.nets = {
            Net.CIFAR10: Cifar10Net,
            Net.CIFAR10_SMALL: Cifar10Net_Small,
            Net.MNIST: MnistNet,
        }

        assert isinstance(dataset, SuppDataset)
        self.dataset = dataset
        self.net = net

    def save(self, path, filename):
        # todo! also save serial?
        assert self.policy_net is not None, 'set model first!'
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def get_net_class(self):
        return self.nets[self.net]

    def parameter_vector(self):
        assert self.policy_net is not None, 'set model first!'
        return nn.utils.parameters_to_vector(self.policy_net.parameters())

    def get_serial_model(self):
        return copy.deepcopy(self.serial_net)

    def nb_learnable_params(self):
        assert self.policy_net is not None, 'set model first!'
        return self.policy_net.get_nb_learnable_params()

    # def load(self, filename):
    #     raise NotImplementedError

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

    def init_model(self, model: PolicyNet = None):
        assert isinstance(model, PolicyNet), '{}'.format(type(model))
        self.policy_net = model
        self.serial_net = model.state_dict()

    def set_model(self, model):
        assert isinstance(model, PolicyNet) or isinstance(model, dict)
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


class ClfPolicy(Policy, ABC):
    def rollout(self, data):
        # CAUTION: memory: https://pytorch.org/docs/stable/notes/faq.html
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        inputs, labels = data
        outputs = self.policy_net(inputs)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        result = -float(loss.item())

        del inputs, labels, outputs, loss, criterion
        return result

    def accuracy_on(self, data):
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        inputs, labels = data
        outputs = self.policy_net(inputs)

        prediction = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        accuracy = float(correct) / labels.size()[0]

        del inputs, labels, outputs, prediction, correct
        return accuracy


class NetsClfPolicy(ClfPolicy, NetsPolicy):
    pass


class SeedsClfPolicy(ClfPolicy, SeedsPolicy):
    pass


class PolicyFactory:
    @staticmethod
    def create(dataset: SuppDataset, mode, net: Net):
        if dataset == SuppDataset.MNIST or dataset == SuppDataset.CIFAR10:
            if mode == 'seeds':
                return SeedsClfPolicy(dataset, net)
            else:
                return NetsClfPolicy(dataset, net)
