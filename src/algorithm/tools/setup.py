import json


from algorithm.tools.experiment import ExperimentFactory
from algorithm.tools.iteration import Iteration
from algorithm.policies import SuppDataset, PolicyFactory, Net
from algorithm.tools.statistics import Statistics
from algorithm.tools.utils import Config


def setup_worker(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    experiment = ExperimentFactory.create(SuppDataset(exp['dataset']), exp, config, master=False)
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']), exp=exp)

    # elite = policy.generate_model()
    policy.init_model(policy.generate_model())
    return config, policy, experiment


def setup_master(exp):
    assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

    config = Config(**exp['config'])
    experiment = ExperimentFactory.create(SuppDataset(exp['dataset']), exp, config)
    policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                  net=Net(exp['net']), exp=exp)
    statistics = Statistics()
    iteration = Iteration(config, exp)

    if 'from_population' in exp and exp['from_population'] is not None:

        with open(exp['from_population']) as f:
            infos = json.load(f)

        statistics.init_from_infos(infos)
        iteration.init_from_infos(infos)
        experiment.init_from_infos(infos)

    elif 'from_single' in exp and exp['from_single'] is not None:
        iteration.init_from_single(exp['from_single'], exp['truncation'], exp['num_elite_cands'], policy)

    else:
        iteration.init_parents(exp['truncation'], exp['num_elite_cands'], policy)

    policy.init_model(policy.generate_model())
    # policy.set_model(iteration.elite())
    return config, policy, statistics, iteration, experiment
