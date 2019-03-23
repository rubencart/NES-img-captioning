import json
from collections import namedtuple

import torch
import torchvision
from torchvision.transforms import transforms

from policies import CompressedModel

config_fields = [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch', 'stdev_decr_divisor',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq', 'num_dataloader_workers', 'log_dir',
    'return_proc_mode', 'episode_cutoff_mode', 'batch_size', 'max_nb_epochs', 'patience'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))

ALL_DATASETS = {
    'cifar10': torchvision.datasets.CIFAR10,
    'mnist': torchvision.datasets.MNIST,
}

ALL_TRANSFORMS = {
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
}


def setup(exp):
    import policies
    config = Config(**exp['config'])
    Policy = getattr(policies, exp['policy']['type'])  # (**exp['policy']['args'])

    def _from_zero():
        _epoch = 0
        _iteration = 0
        _parents = [(model_id, None) for model_id in range(exp['truncation'])]
        # first elite = random
        _elite = CompressedModel()
        _score_stats = [[], [], []]
        _time_stats = []
        _acc_stats = [[], []]
        _norm_stats = []
        return _epoch, _iteration, _elite, _parents, _score_stats, _time_stats, _acc_stats, _norm_stats

    if 'continue_from_population' in exp and exp['continue_from_population'] is not None:
        with open(exp['continue_from_population']) as f:
            infos = json.load(f)
        epoch = infos['epoch'] - 1
        iteration = infos['iter'] - 1

        parents = [(i, CompressedModel(start_rng=p_dict['start_rng'],
                                       other_rng=p_dict['other_rng'],
                                       from_param_file=p_dict['from_param_file']))
                   for (i, p_dict) in enumerate(infos['parents'])]

        elite = parents[0][1]
        score_stats = infos['score_stats']
        time_stats = infos['time_stats']
        acc_stats = infos['acc_stats']
        norm_stats = infos['norm_stats']

    elif 'continue_from_single' in exp and exp['continue_from_single'] is not None:
        parents = [(i, CompressedModel(from_param_file=exp['continue_from_single']))
                   for i in range(exp['truncation'])]

        elite = parents[0][1]
        (epoch, iteration, _, _, score_stats, time_stats, acc_stats, norm_stats) = _from_zero()

    else:
        (epoch, iteration, elite, parents, score_stats, time_stats, acc_stats, norm_stats) = _from_zero()

    trainloader = init_trainldr(exp, config=config)
    return (config, Policy, epoch, iteration, elite, parents, score_stats,
            time_stats, acc_stats, norm_stats, trainloader)


def init_trainldr(exp, config=None, batch_size=None, workers=None):
    dataset = exp['dataset']
    uninit_dataset = ALL_DATASETS[dataset]
    transform = ALL_TRANSFORMS[dataset]

    dataset = uninit_dataset(root='./data', train=True, download=True, transform=transform)

    if config:
        bs = config.batch_size
        num_workers = config.num_dataloader_workers if config.num_dataloader_workers else 1
    else:
        assert isinstance(batch_size, int)
        bs = batch_size
        num_workers = workers if workers else 1

    loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                         shuffle=True, num_workers=num_workers)
    return loader
