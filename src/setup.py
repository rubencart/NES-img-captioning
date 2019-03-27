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

    elif 'continue_from_single' in exp and exp['continue_from_single'] is not None:
        # todo
        parents, elite = _load_parents_from_single(exp['continue_from_single'],
                                                   exp['truncation'], policy)
    else:
        iteration.init_parents(exp['truncation'], policy)

    policy.init_model(iteration.elite())
    return config, policy, statistics, iteration, experiment


def _load_parents_from_single(param_file, truncation, policy):
    parents = [(i, policy.generate_model(from_param_file=param_file))
               for i in range(truncation)]
    return parents, parents[0][1]


# todo to iteration?
# def _init_parents(truncation, policy):
#     # important that this stays None:
#     # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
#     #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
#     # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
#     #       TRUNC random parents == less random!
#     parents = [(model_id, None) for model_id in range(truncation)]
#     elite = policy.generate_model()
#     return parents, elite
