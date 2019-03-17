import copy
import json
import logging
import os
import time
from collections import namedtuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from es_distributed.dist import MasterClient, WorkerClient
from es_distributed.main import mkdir_p
from es_distributed.policies import CompressedModel
from es_distributed.utils import plot_stats, save_snapshot

logger = logging.getLogger(__name__)

# todo clean
ga_task_fields = ['elite', 'population', 'ob_mean', 'ob_std',
                  'timestep_limit', 'batch_data', 'parents']
GATask = namedtuple('GATask', field_names=ga_task_fields, defaults=(None,) * len(ga_task_fields))

config_fields = [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq', 'num_dataloader_workers',
    'return_proc_mode', 'episode_cutoff_mode', 'batch_size', 'max_nb_epochs',
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])

result_fields = [
    'worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
]
Result = namedtuple('Result', field_names=result_fields, defaults=(None,) * len(result_fields))


def setup(exp):
    # todo
    from . import policies
    config = Config(**exp['config'])
    policy = getattr(policies, exp['policy']['type'])(**exp['policy']['args'])
    return config, policy


def run_master(master_redis_cfg, log_dir, exp, plot):
    logger.info('run_master: {}'.format(locals()))
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)

    # config, env, sess, policy = setup(exp, single_threaded=False)
    config, policy = setup(exp)

    # redis master
    master = MasterClient(master_redis_cfg)

    # noise = SharedNoiseTable()
    rs = np.random.RandomState()

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # MNIST has 60k training images
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    # batch_size = config.batch_size if config.batch_size else 256
    batch_size = config.batch_size

    # todo num_workers?
    num_workers = config.num_dataloader_workers

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=num_workers)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.set_grad_enabled(False)

    tstart = time.time()

    # this puts up a redis key, value pair with the experiment
    master.declare_experiment(exp)

    best_score = float('-inf')
    population_size = exp['population_size']
    truncation = exp['truncation']
    num_elites = exp['num_elites']      # todo use num_elites instead of 1

    # todo continue_from
    parents = [(model_id, None) for model_id in range(truncation)]

    # todo use best so far as elite instead of best of last gen?
    elite = CompressedModel()
    policy.set_model(elite)

    score_stats = [[], [], []]
    time_stats = []
    acc_stats = [[], []]
    norm_stats = []

    epoch = 0
    max_nb_epochs = 1000
    try:
        # todo max epochs?
        while True or epoch < max_nb_epochs:
            epoch += 1

            # todo max generations
            # todo check how many times training set has been gone through
            for iteration, batch_data in enumerate(trainloader, 0):
                total_iteration = ((epoch - 1) * len(trainloader) + iteration)

                step_tstart = time.time()

                # todo what is ob_stat?
                # if policy.needs_ob_stat:
                #     ob_stat = RunningStat(
                #         env.observation_space.shape,
                #         eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
                #     )

                curr_task_id = master.declare_task(GATask(
                    elite=elite,
                    parents=parents,
                    batch_data=batch_data,
                ))

                tlogger.log('********** Iteration {} **********'.format(total_iteration))
                logger.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                curr_task_results, eval_rets, worker_ids = [], [], []
                num_results_skipped = 0

                nb_models_to_evaluate = population_size
                elite_evaluated = False
                at_least_one_eval = False

                while nb_models_to_evaluate > 0 or not elite_evaluated or not at_least_one_eval:
                    # if nb_models_to_evaluate % 50 == 0 and nb_models_to_evaluate != population_size:
                    #     logger.info('Nb of models left to evaluate: {nb}. Elite evaluated: {el}'
                    #                 .format(nb=nb_models_to_evaluate, el=elite_evaluated))

                    if nb_models_to_evaluate <= 0 and not elite_evaluated:
                        logger.warning('Only the elite still has to be evaluated')

                    if nb_models_to_evaluate <= 0 and not at_least_one_eval:
                        logger.warning('Waiting for an eval run')

                    # Wait for a result
                    task_id, result = master.pop_result()
                    assert isinstance(task_id, int) and isinstance(result, Result)

                    worker_ids.append(result.worker_id)

                    if result.eval_return is not None:
                        # This was an eval job
                        # Store the result only for current tasks
                        if task_id == curr_task_id:
                            eval_rets.append(result.eval_return)
                            at_least_one_eval = True

                    elif result.evaluated_model_id is not None:
                        # assert result.returns_n2.dtype == np.float32
                        assert result.fitness.dtype == np.float32

                        # Store results only for current tasks
                        if task_id == curr_task_id:

                            nb_models_to_evaluate -= 1
                            curr_task_results.append(result)
                            if result.evaluated_model_id == 0:
                                elite_evaluated = True

                        else:
                            num_results_skipped += 1
                    else:
                        # first generation eval runs are empty
                        pass

                # Compute skip fraction
                frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
                if num_results_skipped > 0:
                    logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                        num_results_skipped, 100. * frac_results_skipped))

                scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                                 for result in curr_task_results]
                elite_scored_models = [t for t in scored_models if t[0] == 0]
                other_scored_models = [t for t in scored_models if t[0] != 0]

                best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))

                scored_models = [best_elite_score] + other_scored_models

                scored_models.sort(key=lambda x: x[2], reverse=True)
                scores = np.array([fitness for (_, _, fitness) in scored_models])

                # pick parents for next generation, give new index
                parents = [(i, model) for (i, (_, model, _)) in enumerate(scored_models[:truncation])]

                logger.info('Best 5: ', [(i, round(f, 2)) for (i, _, f) in scored_models[:5]])
                # input('PRESS ENTER')

                score_stats[0].append(scores.min())
                score_stats[1].append(scores.mean())
                score_stats[2].append(scores.max())

                step_tend = time.time()
                time_stats.append(step_tend - step_tstart)

                norm = float(np.sqrt(np.square(policy.parameter_vector()).sum()))
                norm_stats.append(norm)

                # todo also keep conf interv
                acc_stats[0].append(max([fit for fit, _ in eval_rets]))
                acc_stats[1].append(max([acc for _, acc in eval_rets]))

                # todo assertions
                # assert len(population) == population_size
                # assert np.max(returns_n2) == population_score[0]

                tlogger.record_tabular("RewMax", scores.max())
                tlogger.record_tabular("RewMean", scores.mean())
                tlogger.record_tabular("RewMin", scores.min())
                tlogger.record_tabular("RewStd", scores.std())

                # todo apart from norm, would also be interesting to see how far params are from
                # each other in param space (distance between param_vectors)
                tlogger.record_tabular("Norm", norm)

                if eval_rets:
                    tlogger.record_tabular("MaxAcc", acc_stats[1][-1])

                num_unique_workers = len(set(worker_ids))
                tlogger.record_tabular("UniqueWorkers", num_unique_workers)
                tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
                tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)

                tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
                tlogger.record_tabular("TimeElapsed", step_tend - tstart)
                tlogger.dump_tabular()

                if config.snapshot_freq != 0 and total_iteration % config.snapshot_freq == 0:

                    filename = save_snapshot(acc_stats, epoch, iteration, parents, policy, len(trainloader))

                    # todo adjust tlogger to log like logger (time, pid)
                    logger.info('Saved snapshot {}'.format(filename))

                    if plot:
                        plot_stats(log_dir, score_stats,
                                   time=(time_stats, 'Time per gen'),
                                   norm=(norm_stats, 'Norm of params'),
                                   # todo also plot fitness
                                   acc=(acc_stats[1], 'Elite accuracy'))

                # set policy to new elite
                elite = scored_models[0][1]
                policy.set_model(elite)

    except KeyboardInterrupt:
        if plot:
            plot_stats(log_dir, score_stats,
                       time=(time_stats, 'Time per gen'),
                       norm=(norm_stats, 'Norm of params'),
                       # todo also plot fitness
                       acc=(acc_stats[1], 'Elite accuracy'))


def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    torch.set_grad_enabled(False)

    # redis client
    worker = WorkerClient(master_redis_cfg, relay_redis_cfg)

    exp = worker.get_experiment()
    # config, env, sess, policy = setup(exp, single_threaded=True)
    config, policy = setup(exp)

    rs = np.random.RandomState()  # todo randomstate vs seed? blogpost
    worker_id = rs.randint(2 ** 31)
    # todo worker_id random int???? what if two get the same?

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, GATask)

        if rs.rand() < config.eval_prob:
            model = copy.deepcopy(task_data.elite)

            policy.set_model(model)
            fitness = policy.rollout(task_data.batch_data)

            accuracy = policy.accuracy_on(task_data.batch_data)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                eval_return=(fitness, accuracy)
            ))
        else:
            # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?

            index = rs.randint(len(task_data.parents))
            parent_id, compressed_parent = task_data.parents[index]

            if compressed_parent is None:
                # 0'th iteration: first models still have to be generated
                model = CompressedModel()
            else:
                model = copy.deepcopy(compressed_parent)
                # elite doesn't have to be evolved
                if index != 0:
                    model.evolve(config.noise_stdev)
                    assert isinstance(model, CompressedModel)

            policy.set_model(model)

            fitness = policy.rollout(task_data.batch_data)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                evaluated_model_id=parent_id,
                evaluated_model=model,
                fitness=np.array([fitness], dtype=np.float32)
            ))
