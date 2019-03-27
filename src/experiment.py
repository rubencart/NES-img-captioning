import json
import os
from abc import ABC

import torch
import torchvision
from torchvision.transforms import transforms

from policies import SuppDataset
from utils import mkdir_p


class Experiment(ABC):
    """
    Wrapper class for a bunch of experiment wide settings
    """

    def __init__(self, exp, config):
        self._exp = exp
        self._population_size = exp['population_size']
        self._truncation = exp['truncation']
        self._num_elites = exp['num_elites']  # todo use num_elites instead of 1
        self._mode = exp['mode']

        self.trainloader, self.valloader, self.testloader = self.init_loaders(config=config)
        self._orig_trainloader_lth = len(self.trainloader)

        self._log_dir = 'logs/es_{}'.format(os.getpid())
        mkdir_p(self._log_dir)
        with open(os.path.join(self._log_dir, 'experiment.json'), 'w') as f:
            json.dump(exp, f)

    def to_dict(self):
        return {
            # todo other stuff? + needs from_dict method as well
            # like log dir?
            'trainloader_lth': self._orig_trainloader_lth,
        }

    def increase_loader_batch_size(self, batch_size):
        self.trainloader, self.valloader, self.testloader = self.init_loaders(batch_size=batch_size)

    def _init_loaders(self, dataset, transform, config, batch_size, workers):
        trainset = dataset(root='./data', train=True,
                           download=True, transform=transform)
        valset, testset = self._split_testset(dataset, transform)

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

    def _split_testset(self, dataset, transform):
        comp_testset = dataset(root='./data', train=False,
                               download=True, transform=transform)
        n1, n2 = len(comp_testset) // 2, len(comp_testset) - (len(comp_testset) // 2)
        valset, testset = torch.utils.data.random_split(comp_testset, (n1, n2))
        return valset, testset

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
        return self._log_dir

    def init_loaders(self, config=None, batch_size=None, workers=None):
        raise NotImplementedError


class MnistExperiment(Experiment):
    def __init__(self, exp, config):
        super().__init__(exp, config)

    def init_loaders(self, config=None, batch_size=None, workers=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        return self._init_loaders(torchvision.datasets.MNIST, transform, config, batch_size, workers)


class Cifar10Experiment(Experiment):
    def __init__(self, exp, config):
        super().__init__(exp, config)

    def init_loaders(self, config=None, batch_size=None, workers=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return self._init_loaders(torchvision.datasets.CIFAR10, transform, config, batch_size, workers)


class ExperimentFactory:

    @staticmethod
    def create(dataset: SuppDataset, exp, config):
        if dataset == SuppDataset.MNIST:
            return MnistExperiment(exp, config)
        elif dataset == SuppDataset.CIFAR10:
            return Cifar10Experiment(exp, config)
