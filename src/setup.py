import json

from experiment import ExperimentFactory
from iteration import Iteration
from policies import CompressedModel, SuppDataset, PolicyFactory, Net
from statistics import Statistics
from utils import Config


def setup_worker(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']))

    _, elite = _init_parents(exp['truncation'], policy)
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

    if 'continue_from_population' in exp and exp['continue_from_population'] is not None:
        with open(exp['continue_from_population']) as f:
            infos = json.load(f)

        statistics.init_from_infos(infos)
        iteration.init_from_infos(infos, config)

        parents, elite = _load_parents_from_pop(infos, exp['mode'], policy)
    else:
        if 'continue_from_single' in exp and exp['continue_from_single'] is not None:
            parents, elite = _load_parents_from_single(exp['continue_from_single'],
                                                       exp['truncation'], policy)
        else:
            parents, elite = _init_parents(exp['truncation'], policy)

    # todo parents also in iteration?
    iteration.set_parents(parents)
    iteration.set_elite(elite)

    policy.init_model(elite)
    return config, policy, elite, parents, statistics, iteration, experiment


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


def _load_parents_from_single(param_file, truncation, policy):
    parents = [(i, policy.generate_model(from_param_file=param_file))
               for i in range(truncation)]
    return parents, parents[0][1]


def _init_parents(truncation, policy):
    # important that this stays None:
    # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
    #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
    # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
    #       TRUNC random parents == less random!
    parents = [(model_id, None) for model_id in range(truncation)]
    elite = policy.generate_model()
    return parents, elite
