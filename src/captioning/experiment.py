from collections import namedtuple

import torch

from algorithm.tools.experiment import Experiment
from algorithm.tools.utils import Config


_opt_fields = ['input_json', 'input_fc_dir', 'input_att_dir', 'input_label_h5', 'use_att', 'use_box',
               'norm_att_feat', 'norm_box_feat', 'input_box_dir', 'train_only', 'seq_per_img', 'fitness']
CaptionOptions = namedtuple('CaptionOptions', field_names=_opt_fields, defaults=(None,) * len(_opt_fields))


class MSCocoExperiment(Experiment):
    """
        Subclass for MSCOCO experiment
    """

    def __init__(self, exp, config: Config, master=True):
        self.opt: CaptionOptions = CaptionOptions(**exp['caption_options'])

        super().__init__(exp, config, master=master)

        self.vocab_size = self.trainloader.loader.vocab_size
        self.seq_length = self.trainloader.loader.seq_length

        exp['policy_options']['model_options'].update({
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
        })

    def init_loaders(self, config=None, batch_size=None, workers=None, _=None):
        assert not (config is None and batch_size is None)

        from captioning.dataloader import DataLoader
        tloader = DataLoader(opt=self.opt, config=config, batch_size=batch_size)

        val_bs = config.val_batch_size if config and config.val_batch_size else batch_size
        vloader = DataLoader(opt=self.opt, config=config, batch_size=val_bs)

        trainloader = MSCocoDataLdrWrapper(loader=tloader, split='train')
        valloader = MSCocoDataLdrWrapper(loader=vloader, split='val')
        testloader = MSCocoDataLdrWrapper(loader=vloader, split='test')

        self.trainloader, self.valloader, self.testloader = trainloader, valloader, testloader
        self._orig_trainloader_lth = len(self.trainloader)

    def take_ref_batch(self, batch_size):
        return self.trainloader.take_ref_batch(bs=batch_size)


class MSCocoDataLdrWrapper:
    """
        Wrapper for to map API from dataloader from https://github.com/ruotianluo/self-critical.pytorch
        to expected API
    """

    def __init__(self, loader, split):
        from captioning.dataloader import DataLoader

        self.loader: DataLoader = loader
        self.split = split
        self.batch_size = loader.batch_size
        self.seq_per_img = loader.seq_per_img

        self.get_vocab = loader.get_vocab

    def reset(self):
        self.loader.reset_iterator(split=self.split)

    def __iter__(self):
        return self

    def __next__(self):
        # todo raise stopiter
        return self.loader.get_batch(self.split)

    def __len__(self):
        return self.loader.length_of_split(self.split) // self.loader.batch_size

    def take_ref_batch(self, bs):
        return torch.from_numpy(self.loader.get_batch(self.split, batch_size=bs)['fc_feats'])
