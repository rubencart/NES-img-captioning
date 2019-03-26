import copy
from enum import Enum
import logging
import os
from abc import ABC

import torchvision


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import mkdir_p

logger = logging.getLogger(__name__)


class SuppDataset(Enum):
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'


DATASETS = {
    SuppDataset.CIFAR10: torchvision.datasets.CIFAR10,
    SuppDataset.MNIST: torchvision.datasets.MNIST,
}


class PolicyNet(nn.Module, ABC):
    def __init__(self, rng_state=None, from_param_file=None):
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        self.from_param_file = from_param_file

        if rng_state:
            torch.manual_seed(rng_state)

        self.evolve_states = []
        self.add_tensors = {}

        self.nb_learnable_params = 0

    def _initialize_params(self):
        if self.from_param_file:
            self.load_state_dict(torch.load(self.from_param_file))

        for name, tensor in self.named_parameters():
            # todo this is completely unnecessary
            if tensor.size() not in self.add_tensors:
                # print('CAUTION: new tensor size: ', tensor.size())
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())

            if not self.from_param_file:
                if 'weight' in name:
                    # todo kaiming normal or:
                    # We use Xavier initialization (Glorot & Bengio, 2010) as our policy initialization
                    # function φ where all bias weights are set to zero, and connection weights are drawn
                    # from a standard normal distribution with variance 1/Nin, where Nin is the number of
                    # incoming connections to a neuron
                    # nn.init.kaiming_normal_(tensor)
                    nn.init.xavier_normal_(tensor)
                else:
                    tensor.data.zero_()

        self.nb_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # freeze all params
        for param in self.parameters():
            param.requires_grad = False

    def evolve(self, sigma, rng_state=None):
        # Evolve params 1 step
        # rng_state = int
        rng = rng_state if rng_state is not None else random_state()

        torch.manual_seed(rng)
        # todo this is needed for when in seeds mode we want to be able to evolve uncompressed networks
        # as well (not strictly necessary)
        # self.evolve_states.append((sigma, rng))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            # fill to_add elements sampled from normal distr
            to_add.normal_(mean=0.0, std=sigma)
            tensor.data.add_(to_add)

    def get_nb_learnable_params(self):
        return self.nb_learnable_params

    # see comment in evolve()
    # def compress(self):
    #     return CompressedModel(self.rng_state, self.evolve_states, self.from_param_file)

    # def serialize(self):
    #     return copy.deepcopy(self.state_dict())

    # def deserialize(self, serialized):
    #     self.load_state_dict(serialized)

    # def forward(self, x):
    #     pass


class Cifar10Net(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None):
        super(Cifar10Net, self).__init__(rng_state, from_param_file)

        # 100K params, max ~65 acc?
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, 5)
        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        self.fc3 = nn.Linear(120, 10)

        # 291466 params
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #
        # self.fc2 = nn.Linear(128, 32)
        # self.fc3 = nn.Linear(32, 10)
        # self.fc4 = nn.Linear(8, 2)

        self._initialize_params()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
        # def _conv_conv_pool(conv1, conv2, pool, x):
        #     x = F.relu(conv1(x))
        #     x = F.relu(conv2(x))
        #     return pool(x)
        #
        # x = _conv_conv_pool(self.conv1, self.conv2, self.pool, x)
        # x = _conv_conv_pool(self.conv3, self.conv4, self.pool, x)
        # x = _conv_conv_pool(self.conv5, self.conv6, self.avg_pool, x)
        #
        # x = x.view(-1, 128)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # return x


class BlockSlidesNet32(nn.Module):

    def __init__(self):
        super(BlockSlidesNet32, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # self.avg_pool = nn.AvgPool2d(8, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 2)

        # self.bn_fc2 = nn.BatchNorm1d(32)
        # self.bn_fc3 = nn.BatchNorm1d(8)

    def forward(self, x):
        def _conv_conv_pool(conv1, conv2, pool, x):
            x = F.relu(conv1(x))
            x = F.relu(conv2(x))
            return pool(x)

        x = _conv_conv_pool(self.conv1, self.conv2, self.pool, x)
        x = _conv_conv_pool(self.conv3, self.conv4, self.pool, x)
        x = _conv_conv_pool(self.conv5, self.conv6, self.avg_pool, x)

        x = x.view(-1, 128)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class MnistNet(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None):
        super(MnistNet, self).__init__(rng_state, from_param_file)

        # todo compare fitness incr rate with and without weight norm + time per generation
        # self.conv1 = nn.utils.weight_norm(nn.Conv2d(1, 10, 5, 1))
        # self.conv2 = nn.utils.weight_norm(nn.Conv2d(10, 20, 5, 1))
        # self.fc1 = nn.utils.weight_norm(nn.Linear(4*4*20, 100))
        # self.fc2 = nn.utils.weight_norm(nn.Linear(100, 10))

        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 40, 5, 1)
        # self.fc1 = nn.Linear(4*4*40, 100)
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(4*4*20, 80)
        self.fc2 = nn.Linear(80, 10)

        self._initialize_params()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CompressedModel:
    def __init__(self, start_rng: int = None, other_rng: list = None, from_param_file: str = None):
        if start_rng is None and from_param_file is None:
            self.start_rng, self.from_param_file = random_state(), None
        elif start_rng is None and from_param_file is not None:
            self.start_rng, self.from_param_file = None, from_param_file
        elif start_rng is not None and from_param_file is None:
            self.start_rng, self.from_param_file = start_rng, None
        else:
            raise ValueError('start_rng and from_param_file cannot be both set')

        self.other_rng = other_rng if other_rng is not None else []

    def evolve(self, sigma, rng_state=None):
        # evolve params 1 step
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))

    def uncompress(self, to_class_name=MnistNet):
        # evolve through all steps
        m = to_class_name(self.start_rng, self.from_param_file)
        for sigma, rng in self.other_rng:
            m.evolve(sigma, rng)
        return m

    # def serialize(self):
    #     return self

    # def deserialize(self, serialized):
    #     self = serialized

    def __str__(self):
        start = self.start_rng if self.start_rng else self.from_param_file

        result = '[( {start} )'.format(start=start)
        for _, rng in self.other_rng:
            result += '( {rng} )'.format(rng=rng)
        return result + ']'


def random_state():
    rs = np.random.RandomState()
    return rs.randint(0, 2 ** 31 - 1)


class Policy(ABC):
    def __init__(self, dataset: SuppDataset):
        self.policy_net = None
        self.serial_net = None

        self.nets = {
            SuppDataset.CIFAR10: Cifar10Net,
            SuppDataset.MNIST: MnistNet,
        }

        assert isinstance(dataset, SuppDataset)
        self.dataset = dataset

    def save(self, path, filename):
        # todo! also save serial?
        assert self.policy_net is not None, 'set model first!'
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def get_net_class(self):
        return self.nets[self.dataset]

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
    def create(dataset, mode):
        if dataset == SuppDataset.MNIST or dataset == SuppDataset.CIFAR10:
            if mode == 'seeds':
                return SeedsClfPolicy(dataset)
            else:
                return NetsClfPolicy(dataset)
