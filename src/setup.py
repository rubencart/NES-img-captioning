import json
from collections import namedtuple

import torch
import torchvision
from torchvision.transforms import transforms

from policies import CompressedModel, Cifar10Policy, MnistPolicy

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

POLICIES = {
    'cifar10': Cifar10Policy,
    'mnist': MnistPolicy,
}


def setup(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    Policy = POLICIES[exp['policy']['type']]
    policy = Policy()

    if 'continue_from_population' in exp and exp['continue_from_population'] is not None:
        with open(exp['continue_from_population']) as f:
            infos = json.load(f)

        (epoch, iteration, score_stats, time_stats, acc_stats, norm_stats,
         std_stats) = _load_stats_from_infos(infos)

        parents, elite = _load_parents_from_pop(infos, exp['mode'])
    else:
        (epoch, iteration, score_stats, time_stats, acc_stats, norm_stats, std_stats) = _init_stats()

        if 'continue_from_single' in exp and exp['continue_from_single'] is not None:
            parents, elite = _load_parents_from_single(exp['continue_from_single'], exp['mode'],
                                                       exp['truncation'], policy)
        else:
            parents, elite = _init_parents(exp, policy)

    trainloader, valloader, testloader = init_loaders(exp, config=config)

    return (config, policy, epoch, iteration, elite, parents, score_stats,
            time_stats, acc_stats, norm_stats, std_stats, trainloader, valloader, testloader)


def _load_parents_from_pop(infos, mode, policy):
    if mode == 'seeds':
        parents = [(i, CompressedModel(start_rng=p_dict['start_rng'],
                                       other_rng=p_dict['other_rng'],
                                       from_param_file=p_dict['from_param_file']))
                   for (i, p_dict) in enumerate(infos['parents'])]
    else:
        # todo serial / deserial
        # see https://github.com/pytorch/tutorials/blob/0eec7facdb659269be33289f5add6e5acf4493c9
        # /beginner_source/saving_loading_models.py#L289
        raise NotImplementedError
        NetClass = policy.get_net_class()
        parents = [(i, NetClass(from_param_file=p_dict['from_param_file']))
                   for (i, p_dict) in enumerate(infos['parents'])]

    return parents, parents[0][1]


def _load_parents_from_single(param_file, mode, truncation, policy):
    if mode == 'seeds':
        parents = [(i, CompressedModel(from_param_file=param_file))
                   for i in range(truncation)]
    else:
        # todo get_net_class nicest solution?
        parents = [(i, policy.get_net_class()(from_param_file=param_file))
                   for i in range(truncation)]

    return parents, parents[0][1]


def _init_parents(mode, truncation, policy):
    if mode == 'seeds':
        # important that this stays None:
        # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
        #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
        # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
        #       TRUNC random parents == less random!
        parents = [(model_id, None) for model_id in range(truncation)]
        elite = CompressedModel()
    else:
        parents = [(model_id, None) for model_id in range(truncation)]
        elite = policy.generate_net()

    return parents, elite


def _load_stats_from_infos(infos):
    epoch = infos['epoch'] - 1
    iteration = infos['iter'] - 1
    score_stats = infos['score_stats']
    time_stats = infos['time_stats']
    acc_stats = infos['acc_stats']
    norm_stats = infos['norm_stats']

    # todo
    std_stats = infos['noise_std_stats'] if 'noise_std_stats' in infos else _init_stats()[-1]
    best_elite = infos['best_elite'] if 'noise_std_stats' in infos else None
    trainloader_length = infos['trainloader_lth'] if 'noise_std_stats' in infos else None
    best_parents = infos['best_parents'] if 'noise_std_stats' in infos else None
    bs_stats = infos['batch_size_stats'] if 'batch_size_stats' in infos else []
    batch_size = bs_stats[-1] if bs_stats else None

    return epoch, iteration, score_stats, time_stats, acc_stats, norm_stats, std_stats


def _init_stats():
    _epoch = 0
    _iteration = 0
    _score_stats = [[], [], []]
    _time_stats = []
    _acc_stats = [[], []]
    _norm_stats = []
    _noise_std_stats = []
    return (_epoch, _iteration, _score_stats, _time_stats, _acc_stats,
            _norm_stats, _noise_std_stats)


def init_loaders(exp, config=None, batch_size=None, workers=None):
    dataset = exp['dataset']

    if dataset == 'mnist':
        return _init_mnist_loaders(config, batch_size, workers)
    elif dataset == 'cifar10':
        return _init_cifar10_loaders(config, batch_size, workers)
    else:
        raise ValueError('dataset must be mnist|cifar10, now: {}'.format(dataset))


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
    valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset),
                                            shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=num_workers)

    return trainloader, valloader, testloader


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

    valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset),
                                            shuffle=True, num_workers=num_workers)
    # todo batch size?
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=num_workers)

    return trainloader, valloader, testloader
