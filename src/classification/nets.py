
from torch import nn
import torch.nn.functional as F

from algorithm.nets import PolicyNet


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

