import os

import torch
import torchvision
from torchvision.transforms import transforms

from utils import mkdir_p


class Experiment(object):
    """
    Wrapper class for a bunch of experiment wide settings
    """

    def __init__(self, exp, config):
        self._exp = exp
        self._population_size = exp['population_size']
        self._truncation = exp['truncation']
        self._num_elites = exp['num_elites']  # todo use num_elites instead of 1
        self._mode = exp['mode']

        # self._policy_type = policy_class

        self.trainloader, self.valloader, self.testloader = self.init_loaders(exp, config=config)
        self._orig_trainloader_lth = len(self.trainloader)

        self._log_dir = 'logs/es_{}_master_{}'.format(self._mode, os.getpid())
        self._snapshot_dir = 'snapshots/es_{}_master_{}'.format(self._mode, os.getpid())
        mkdir_p(self._log_dir)
        mkdir_p(self._snapshot_dir)

    def to_dict(self):
        return {
            # todo other stuff? + needs from_dict method as well
            'trainloader_lth': self._orig_trainloader_lth,
        }

    def reset_trainloader(self, batch_size):
        self.trainloader = self.init_loaders(self._exp, batch_size=batch_size)

    # todo to dedicated classes
    @staticmethod
    def init_loaders(exp, config=None, batch_size=None, workers=None):
        dataset = exp['dataset']

        if dataset == 'mnist':
            return Experiment._init_mnist_loaders(config, batch_size, workers)
        elif dataset == 'cifar10':
            return Experiment._init_cifar10_loaders(config, batch_size, workers)
        else:
            raise ValueError('dataset must be mnist|cifar10, now: {}'.format(dataset))

    @staticmethod
    def _init_mnist_loaders(config, batch_size, workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        comp_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                  download=True, transform=transform)
        n1, n2 = len(comp_testset) // 2, len(comp_testset) - (len(comp_testset) // 2)
        valset, testset = torch.utils.data.random_split(comp_testset, (n1, n2))

        if config:
            bs = config.batch_size
            num_workers = config.num_dataloader_workers if config.num_dataloader_workers else 1
        else:
            assert isinstance(batch_size, int)
            bs = batch_size
            num_workers = workers if workers else 1

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=num_workers)
        # todo batch size?
        valloader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                                shuffle=True, num_workers=num_workers)

        testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                 shuffle=True, num_workers=num_workers)

        return trainloader, valloader, testloader

    @staticmethod
    def _init_cifar10_loaders(config, batch_size, workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        comp_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
        n1, n2 = len(comp_testset) // 2, len(comp_testset) - (len(comp_testset) // 2)
        valset, testset = torch.utils.data.random_split(comp_testset, (n1, n2))

        if config:
            bs = config.batch_size
            num_workers = config.num_dataloader_workers if config.num_dataloader_workers else 1
        else:
            assert isinstance(batch_size, int)
            bs = batch_size
            num_workers = workers if workers else 1

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                                shuffle=True, num_workers=num_workers)
        # todo batch size?
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                 shuffle=True, num_workers=num_workers)

        return trainloader, valloader, testloader

    def population_size(self):
        return self._population_size

    def truncation(self):
        return self._truncation

    def orig_trainloader_lth(self):
        return self._orig_trainloader_lth

    def mode(self):
        return self._mode

    def log_dir(self):
        return self._log_dir

    def snapshot_dir(self):
        return self._snapshot_dir
