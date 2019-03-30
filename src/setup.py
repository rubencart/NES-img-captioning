import json

import torch

from experiment import ExperimentFactory
from iteration import Iteration, Checkpoint
from policies import SuppDataset, PolicyFactory, Net
from statistics import Statistics
from utils import Config


def setup_worker(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']))

    # _, elite = _init_parents(exp['truncation'], policy)
    elite = policy.generate_model()
    policy.init_model(elite)
    return config, policy


def setup_master(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']))

    experiment = ExperimentFactory.create(SuppDataset(exp['dataset']), exp, config)
    statistics = Statistics()
    iteration = Iteration(config)

    if 'from_population' in exp and exp['from_population'] is not None:
        with open(exp['from_population']['infos']) as f:
            infos = json.load(f)

        models_checkpt = Checkpoint(**torch.load(exp['from_population']['models']))

        statistics.init_from_infos(infos)
        iteration.init_from_infos(infos, models_checkpt, policy)

    elif 'from_single' in exp and exp['from_single'] is not None:

        # single_state_dict = torch.load(exp['from_single'])
        iteration.init_from_single(exp['from_single'], exp['truncation'], policy)

    else:
        iteration.init_parents(exp['truncation'], policy)

    policy.init_model(iteration.elite())
    return config, policy, statistics, iteration, experiment
