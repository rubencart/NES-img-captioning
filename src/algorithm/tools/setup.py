import json

import torch

from algorithm.tools.experiment import ExperimentFactory
from algorithm.tools.iteration import Iteration, Checkpoint
from algorithm.policies import SuppDataset, PolicyFactory, Net
from algorithm.tools.statistics import Statistics
from algorithm.tools.utils import Config


def setup_worker(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']), exp=exp)

    # elite = policy.generate_model()
    policy.init_model(policy.generate_model())
    return config, policy


def setup_master(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    experiment = ExperimentFactory.create(SuppDataset(exp['dataset']), exp, config)
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']), exp=exp)
    statistics = Statistics()
    iteration = Iteration(config, exp)

    if 'from_population' in exp and exp['from_population'] is not None:
        # todo
        with open(exp['from_population']['infos']) as f:
            infos = json.load(f)

        models_checkpt = Checkpoint(**torch.load(exp['from_population']['models']))

        statistics.init_from_infos(infos)
        iteration.init_from_infos(infos, models_checkpt, policy)

    elif 'from_single' in exp and exp['from_single'] is not None:
        # todo

        # single_state_dict = torch.load(exp['from_single'])
        iteration.init_from_single(exp['from_single'], exp['truncation'], policy)

    else:
        iteration.init_parents(exp['truncation'], policy)

    policy.init_model(policy.generate_model())
    policy.set_model(iteration.elite())
    return config, policy, statistics, iteration, experiment
