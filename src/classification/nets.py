import torch
from torch import nn
import torch.nn.functional as F

from algorithm.nets import PolicyNet


class MnistNet(PolicyNet):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
        super(MnistNet, self).__init__(rng_state=rng_state, from_param_file=from_param_file, grad=grad,
                                       options=options, vbn=vbn)

        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(4*4*20, 10)

        if self.vbn:
            # running stats = false because we only ever keep normalization values calculated
            # on the reference minibatch, which should be small. So keeping running stats
            # should amount to the same but with more overhead
            self.bn1 = nn.BatchNorm2d(10, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(20, track_running_stats=False)

        self.initialize_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.vbn else x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x) if self.vbn else x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = self.fc1(x)
        return x

    def forward_for_sensitivity(self, x, orig_bs=0, i=-1):
        if i >= 0:
            data, _ = x
            data = data[i]
            data.unsqueeze_(0)
        else:
            data, _ = x
        _data = torch.Tensor(data).clone().detach().requires_grad_(False)
        del data, _
        return self.forward(_data)
