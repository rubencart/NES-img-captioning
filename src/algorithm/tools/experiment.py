import json
import logging
import os
from abc import ABC

import torch

from algorithm.policies import SuppDataset
from algorithm.tools.utils import mkdir_p, Config

logger = logging.getLogger(__name__)


class Experiment(ABC):
    """
    Wrapper class for a bunch of experiment wide settings
    """

    def __init__(self, exp: dict, config: Config, master=True):
        self._exp = exp
        self._dataset = exp['dataset']
        self._algorithm = exp['algorithm']
        self._net = exp['policy_options']['net']
        self._nb_offspring = exp['nb_offspring']

        self.trainloader, self.valloader, self.testloader = None, None, None
        self._orig_trainloader_lth = 0

        self._orig_bs = config.batch_size
        self.init_loaders(batch_size=self._orig_bs)

        self.ref_batch_size = config.ref_batch_size if config.ref_batch_size else config.batch_size
        self.ref_batch = self.take_ref_batch(self.ref_batch_size)

        self._master = master
        if master:
            self._log_dir = exp['log_dir']
            self._snapshot_dir = os.path.join(self._log_dir, 'snapshot')
            mkdir_p(self._snapshot_dir)

            with open(os.path.join(self._snapshot_dir, 'experiment.json'), 'w') as f:
                json.dump(exp, f)

    def to_dict(self):
        return {
            'trainloader_lth': self._orig_trainloader_lth,
            'algorithm': self._algorithm,
            'orig_bs': self._orig_bs,
        }

    def init_from_infos(self, infos: dict):
        self._orig_bs = infos['orig_bs'] if 'orig_bs' in infos else self._orig_bs
        self._orig_trainloader_lth = infos['trainloader_lth'] if 'trainloader_lth' in infos \
            else self._orig_trainloader_lth
        self._algorithm = infos['algorithm'] if 'algorithm' in infos else self._algorithm

        batch_size = infos['batch_size'] if 'batch_size' in infos else self._orig_bs
        if batch_size != self._orig_bs:
            self.init_loaders(batch_size=batch_size)

    def get_ref_batch(self):
        return self.ref_batch

    def increase_loader_batch_size(self, batch_size):
        self.init_loaders(batch_size=batch_size)

    def _init_torchvision_loaders(self, dataset, transform, config, batch_size, workers):
        trainset = dataset(root='./data', train=True,
                           download=True, transform=transform)
        valset, testset = self._split_testset(dataset, transform)

        if config:
            bs = config.batch_size
            val_bs = config.val_batch_size if config.val_batch_size else len(valset)
            num_workers = config.num_dataloader_workers if config.num_dataloader_workers else 1
        else:
            assert isinstance(batch_size, int)
            bs = batch_size
            val_bs = len(valset)
            num_workers = workers if workers else 0

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=0)
        valloader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                                shuffle=True, num_workers=0)

        testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                 shuffle=True, num_workers=0)
        return trainloader, valloader, testloader

    def _split_testset(self, dataset, transform):
        comp_testset = dataset(root='./data', train=False,
                               download=True, transform=transform)
        n1, n2 = len(comp_testset) // 2, len(comp_testset) - (len(comp_testset) // 2)
        valset, testset = torch.utils.data.random_split(comp_testset, (n1, n2))
        return valset, testset

    def take_ref_batch(self, batch_size):
        return next(iter(self.trainloader))[0][:batch_size]

    def get_trainloader(self):
        return self.trainloader

    def nb_offspring(self):
        return self._nb_offspring

    def orig_trainloader_lth(self):
        return self._orig_trainloader_lth

    def orig_batch_size(self):
        return self._orig_bs

    def log_dir(self):
        assert self._master
        return self._log_dir

    def snapshot_dir(self):
        assert self._master
        return self._snapshot_dir

    def init_loaders(self, config=None, batch_size=None, workers=None, exp=None):
        raise NotImplementedError


class ExperimentFactory:
    @staticmethod
    def create(dataset: SuppDataset, exp, config: Config, master=True):
        from classification.experiment import MnistExperiment
        from captioning.experiment import MSCocoExperiment
        from algorithm.nic_es.experiment import ESExperiment
        from algorithm.nic_nes.experiment import NESExperiment

        if exp['algorithm'] == 'nic_es':
            if dataset == SuppDataset.MNIST:
                class MnistESExperiment(MnistExperiment, ESExperiment):
                    pass
                return MnistESExperiment(exp, config, master=master)
            elif dataset == SuppDataset.MSCOCO:
                class MSCocoESExperiment(MSCocoExperiment, ESExperiment):
                    pass
                return MSCocoESExperiment(exp, config, master=master)

        elif exp['algorithm'] == 'nic_nes':
            if dataset == SuppDataset.MNIST:
                class MnistNESExperiment(MnistExperiment, NESExperiment):
                    pass
                return MnistNESExperiment(exp, config, master=master)
            elif dataset == SuppDataset.MSCOCO:
                class MSCocoNESExperiment(MSCocoExperiment, NESExperiment):
                    pass
                return MSCocoNESExperiment(exp, config, master=master)
