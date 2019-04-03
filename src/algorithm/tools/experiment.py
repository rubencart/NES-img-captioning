import json
import os
from abc import ABC
from collections import namedtuple

import torch
import torchvision
from torchvision.transforms import transforms

from algorithm.policies import SuppDataset
from algorithm.tools.utils import mkdir_p


class Experiment(ABC):
    """
    Wrapper class for a bunch of experiment wide settings
    """

    def __init__(self, exp, config):
        self._exp = exp
        self._population_size = exp['population_size']
        self._truncation = exp['truncation']
        self._num_elites = exp['num_elites']  # todo use num_elites instead of 1
        self._dataset = exp['dataset']
        self._net = exp['net']

        assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])
        self._mode = exp['mode']

        self.trainloader, self.valloader, self.testloader = self.init_loaders(config=config, exp=exp)
        self._orig_trainloader_lth = len(self.trainloader)

        self._log_dir = 'logs/es_{}_{}_{}_{}'.format(self._dataset,
                                                     self._net, self._mode, os.getpid())
        mkdir_p(self._log_dir)
        exp.update({'log_dir': self._log_dir})

        self._snapshot_dir = os.path.join(self._log_dir, 'snapshot')
        mkdir_p(self._snapshot_dir)

        _models_dir = os.path.join(self._log_dir, 'models')
        self._parents_dir = os.path.join(_models_dir, 'parents')
        self._offspring_dir = os.path.join(_models_dir, 'parents')
        self._elite_dir = os.path.join(_models_dir, 'elite')

        mkdir_p(self._parents_dir)
        mkdir_p(self._offspring_dir)
        mkdir_p(self._elite_dir)
        # exp.update({
        #     'parents_dir': self._parents_dir,
        #     'offspring_dir': self._offspring_dir,
        # })

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
            num_workers = workers if workers else 1

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=num_workers)
        # todo batch size?
        valloader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                                shuffle=True, num_workers=num_workers)

        # todo batch size?
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
        return self._snapshot_dir

    def parents_dir(self):
        return self._parents_dir

    def offspring_dir(self):
        return self._offspring_dir

    def elite_dir(self):
        return self._elite_dir

    def init_loaders(self, config=None, batch_size=None, workers=None, exp=None):
        raise NotImplementedError


class MnistExperiment(Experiment):
    def __init__(self, exp, config):
        super().__init__(exp, config)

    def init_loaders(self, config=None, batch_size=None, workers=None, exp=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        return self._init_torchvision_loaders(torchvision.datasets.MNIST, transform, config, batch_size, workers)


class Cifar10Experiment(Experiment):
    def __init__(self, exp, config):
        super().__init__(exp, config)

    def init_loaders(self, config=None, batch_size=None, workers=None, exp=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return self._init_torchvision_loaders(torchvision.datasets.CIFAR10, transform, config, batch_size, workers)


_opt_fields = ['input_json', 'input_fc_dir', 'input_att_dir', 'input_label_h5', 'use_att', 'use_box',
               'norm_att_feat', 'norm_box_feat', 'input_box_dir', 'train_only', 'seq_per_img']
CaptionOptions = namedtuple('Opt', field_names=_opt_fields, defaults=(None,) * len(_opt_fields))


class MSCocoExperiment(Experiment):
    def __init__(self, exp, config):
        self.opt = CaptionOptions(**exp['caption_options'])

        # Deal with feature things before anything
        # self.options.use_att = utils.if_use_att(self.options.caption_model)
        # if self.options.use_box:
        #     self.options.att_feat_size = self.options.att_feat_size + 5

        # self.options.vocab_size = loader.vocab_size
        # self.options.seq_length = loader.seq_length

        super().__init__(exp, config)

        self.vocab_size = self.trainloader.loader.vocab_size
        self.seq_length = self.trainloader.loader.seq_length

        exp['caption_model_options'].update({
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
        })

    def init_loaders(self, config=None, batch_size=None, workers=None, exp=None):
        # TODO MSCOCO as torchvision.dataset?????

        from captioning.dataloader import DataLoader
        tloader = DataLoader(opt=self.opt, config=config, batch_size=batch_size)

        # config.val_batch_size will be None (default value) if not set so ok
        vloader = DataLoader(opt=self.opt, config=config, batch_size=config.val_batch_size)

        trainloader = MSCocoDataLdrWrapper(loader=tloader, split='train')
        valloader = MSCocoDataLdrWrapper(loader=vloader, split='val')
        testloader = MSCocoDataLdrWrapper(loader=vloader, split='test')

        return trainloader, valloader, testloader


class MSCocoDataLdrWrapper:
    def __init__(self, loader, split):
        from captioning.dataloader import DataLoader

        self.loader: DataLoader = loader
        self.split = split

    def reset(self):
        self.loader.reset_iterator(split=self.split)

    def __iter__(self):
        return self

    def __next__(self):
        # todo raise stopiter
        return self.loader.get_batch(self.split)

    def __len__(self):
        return self.loader.length_of_split(self.split) // self.loader.batch_size


class ExperimentFactory:
    @staticmethod
    def create(dataset: SuppDataset, exp, config):
        if dataset == SuppDataset.MNIST:
            return MnistExperiment(exp, config)
        elif dataset == SuppDataset.CIFAR10:
            return Cifar10Experiment(exp, config)
        elif dataset == SuppDataset.MSCOCO:
            return MSCocoExperiment(exp, config)
