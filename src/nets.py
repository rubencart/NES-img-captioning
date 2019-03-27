import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ABCModel:

    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @abstractmethod
    def from_serialized(self, serialized):
        raise NotImplementedError


class PolicyNet(nn.Module, ABCModel, ABC):
    def __init__(self, rng_state=None, from_param_file=None, grad=False):
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        self.from_param_file = from_param_file

        if rng_state:
            torch.manual_seed(rng_state)

        self.evolve_states = []
        self.add_tensors = {}

        self.nb_learnable_params = 0
        self.grad = grad

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
        if not self.grad:
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

    def serialize(self):
        return copy.deepcopy(self.state_dict())

    def from_serialized(self, serialized):
        assert isinstance(serialized, dict)
        self.load_state_dict(serialized)

    # def forward(self, x):
    #     pass


class Cifar10Net_Small(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None, grad=False):
        super(Cifar10Net_Small, self).__init__(rng_state, from_param_file, grad=grad)

        # 100K params, max ~65 acc?
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, 5)
        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        self.fc3 = nn.Linear(120, 10)

        self._initialize_params()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class Cifar10Net(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None, grad=False):
        super(Cifar10Net, self).__init__(rng_state, from_param_file, grad=grad)

        # 291466 params
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(128, 10)

        self._initialize_params()

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
        return x


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
    def __init__(self, rng_state=None, from_param_file=None, grad=False):
        super(MnistNet, self).__init__(rng_state=rng_state, from_param_file=from_param_file, grad=grad)

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


class CompressedModel(ABCModel):
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

    def uncompress(self, to_class_name):
        # evolve through all steps
        m = to_class_name(self.start_rng, self.from_param_file)
        for sigma, rng in self.other_rng:
            m.evolve(sigma, rng)
        return m

    def serialize(self):
        return self.__dict__

    def from_serialized(self, serialized):
        assert isinstance(serialized, CompressedModel)
        self.start_rng = serialized.start_rng
        self.other_rng = serialized.other_rng
        self.from_param_file = serialized.from_param_file

    def __str__(self):
        start = self.start_rng if self.start_rng else self.from_param_file

        result = '[( {start} )'.format(start=start)
        for _, rng in self.other_rng:
            result += '( {rng} )'.format(rng=rng)
        return result + ']'


def random_state():
    rs = np.random.RandomState()
    return rs.randint(0, 2 ** 31 - 1)

