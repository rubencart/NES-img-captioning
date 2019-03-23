import logging
import os

# todo necessary?
# import h5py
import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.layers as layers

# from . import tf_util as U
import torch
from torch import nn
import torch.nn.functional as F

from utils import mkdir_p

logger = logging.getLogger(__name__)


class PolicyNet(nn.Module):
    def __init__(self, rng_state=None, from_param_file=None):
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        self.from_param_file = from_param_file

        if rng_state:
            torch.manual_seed(rng_state)
        self.evolve_states = []
        self.add_tensors = {}

    def _initialize_params(self):
        if self.from_param_file:
            self.load_state_dict(torch.load(self.from_param_file))

        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
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

    def evolve(self, sigma, rng_state):
        # Evolve params 1 step
        # rng_state = int
        torch.manual_seed(rng_state)
        self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            # fill to_add elements sampled from normal distr
            to_add.normal_(mean=0.0, std=sigma)
            tensor.data.add_(to_add)

    def compress(self):
        return CompressedModel(self.rng_state, self.evolve_states, self.from_param_file)

    def forward(self, x):
        pass


class Cifar10Net(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None):
        super(Cifar10Net, self).__init__(rng_state, from_param_file)

        # 60K params
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, 5)
        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
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
        # x = F.relu(self.fc2(x))
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
        self.fc1 = nn.Linear(4*4*20, 100)
        self.fc2 = nn.Linear(100, 10)

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
    def __init__(self, start_rng: float = None, other_rng: list = None, from_param_file: str = None):
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

    def __str__(self):
        start = self.start_rng if self.start_rng else self.from_param_file

        result = '[( {start} )'.format(start=start)
        for _, rng in self.other_rng:
            result += '( {rng} )'.format(rng=rng)
        return result + ']'


def random_state():
    rs = np.random.RandomState()
    return rs.randint(0, 2 ** 31 - 1)


class Policy:
    def __init__(self):
        # todo adjust to pytorch
        # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        # self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self.policy_net = None

    # def rollout(self, data):
    #     assert self.policy_net is not None, 'set model first!'
    #     torch.set_grad_enabled(False)
    #     self.policy_net.eval()
    #
    #     inputs, labels = data
    #     outputs = self.policy_net(inputs)
    #
    #     # todo for now use cross entropy loss as fitness
    #     criterion = nn.CrossEntropyLoss()
    #     loss = criterion(outputs, labels)
    #     # print(loss) --> tensor(2.877)
    #     result = -float(loss.item())
    #
    #     del inputs, labels, outputs, loss, criterion
    #     return result

    def rollout_(self, data, compressed_model):
        # CAUTION: memory: https://pytorch.org/docs/stable/notes/faq.html

        assert compressed_model is not None, 'Pass model!'
        policy_net: PolicyNet = self._uncompress_model(compressed_model)

        torch.set_grad_enabled(False)
        policy_net.eval()

        inputs, labels = data
        outputs = policy_net(inputs)

        # todo for now use cross entropy loss as fitness
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        result = -float(loss.item())

        del inputs, labels, outputs, loss, criterion, policy_net
        return result

    # def accuracy_on(self, data):
    #     assert self.policy_net is not None, 'set model first!'
    #     torch.set_grad_enabled(False)
    #     self.policy_net.eval()
    #
    #     inputs, labels = data
    #     outputs = self.policy_net(inputs)
    #
    #     prediction = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #     correct = prediction.eq(labels.view_as(prediction)).sum().item()
    #     accuracy = float(correct) / labels.size()[0]
    #
    #     del inputs, labels, outputs, prediction, correct
    #     return accuracy

    def accuracy_on_(self, data, compressed_model):
        assert compressed_model is not None, 'Pass model!'
        policy_net: PolicyNet = self._uncompress_model(compressed_model)

        torch.set_grad_enabled(False)
        policy_net.eval()

        inputs, labels = data
        outputs = policy_net(inputs)

        prediction = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        accuracy = float(correct) / labels.size()[0]

        del inputs, labels, outputs, prediction, correct, policy_net
        return accuracy

    def save(self, path, filename):
        # todo check self-critical --> also save iteration,... not only params?
        assert self.policy_net is not None, 'set model first!'
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load(self, filename):
        pass

    @staticmethod
    def nb_learnable_params():
        pass

    def set_model(self, compressed_model):
        pass

    def _uncompress_model(self, compressed_model):
        pass

    def reinitialize_params(self):
        # todo rescale weights manually (can we use weight norm?)
        pass

    def parameter_vector(self):
        assert self.policy_net is not None, 'set model first!'
        return nn.utils.parameters_to_vector(self.policy_net.parameters())


class Cifar10Policy(Policy):
    def __init__(self):
        super(Cifar10Policy, self).__init__()
        # self.model = Cifar10Classifier(random_state())
        # self.model = None

    def set_model(self, compressed_model):
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=Cifar10Net)
        assert isinstance(uncompressed_model, PolicyNet)
        self.policy_net = uncompressed_model

    def _uncompress_model(self, compressed_model) -> Cifar10Net:
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=Cifar10Net)
        assert isinstance(uncompressed_model, PolicyNet)
        return uncompressed_model

    @staticmethod
    def nb_learnable_params():
        torch.set_grad_enabled(True)
        model = Cifar10Net(random_state())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.set_grad_enabled(False)
        return count


class MnistPolicy(Policy):
    def __init__(self):
        super(MnistPolicy, self).__init__()
        # self.model = Cifar10Classifier(random_state())
        # self.model = None

    def set_model(self, compressed_model):
        assert isinstance(compressed_model, CompressedModel), '{}'.format(type(compressed_model))
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=MnistNet)
        assert isinstance(uncompressed_model, PolicyNet)
        self.policy_net = uncompressed_model

    def _uncompress_model(self, compressed_model) -> MnistNet:
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=MnistNet)
        assert isinstance(uncompressed_model, PolicyNet)
        return uncompressed_model

    @staticmethod
    def nb_learnable_params():
        torch.set_grad_enabled(True)
        model = MnistNet(random_state())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.set_grad_enabled(False)
        return count
