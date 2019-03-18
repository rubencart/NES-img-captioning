import copy
import json
import logging
import os
import psutil
import time
from collections import namedtuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from memory_profiler import profile

from es_distributed.dist import MasterClient, WorkerClient
from es_distributed.main import mkdir_p
from es_distributed.policies import CompressedModel
from es_distributed.utils import plot_stats, save_snapshot, readable_bytes

logger = logging.getLogger(__name__)

# todo clean
ga_task_fields = ['elite', 'population', 'ob_mean', 'ob_std',
                  'timestep_limit', 'batch_data', 'parents', 'noise_stdev']
GATask = namedtuple('GATask', field_names=ga_task_fields, defaults=(None,) * len(ga_task_fields))

config_fields = [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch', 'stdev_decr_divisor',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq', 'num_dataloader_workers',
    'return_proc_mode', 'episode_cutoff_mode', 'batch_size', 'max_nb_epochs', 'patience'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])

result_fields = [
    'worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count', 'mem_usage'
]
Result = namedtuple('Result', field_names=result_fields, defaults=(None,) * len(result_fields))


def setup(exp):
    # todo
    from . import policies
    config = Config(**exp['config'])
    Policy = getattr(policies, exp['policy']['type'])  # (**exp['policy']['args'])

    # todo continue from 1 model instead of set of parents should also be possible
    if 'continue_from' in exp and exp['continue_from'] is not None:
        with open(exp['continue_from']) as f:
            infos = json.load(f)
        epoch = infos['epoch'] - 1
        iteration = infos['iter']   # todo -1 ?
        parents = [(i, CompressedModel(p_dict['start_rng'], p_dict['other_rng']))
                   for (i, p_dict) in enumerate(infos['parents'])]
        elite = parents[0][1]
        score_stats = infos['score_stats']
        time_stats = infos['time_stats']
        acc_stats = infos['acc_stats']
        norm_stats = infos['norm_stats']
    else:
        epoch = 0
        iteration = 0
        elite = CompressedModel()
        parents = [(model_id, None) for model_id in range(exp['truncation'])]
        score_stats = [[], [], []]
        time_stats = []
        acc_stats = [[], []]
        norm_stats = []

    return (config, Policy, epoch, iteration, elite, parents,
            score_stats, time_stats, acc_stats, norm_stats)


def run_master(master_redis_cfg, log_dir, exp, plot):
    logger.info('run_master: {}'.format(locals()))
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)

    lower_memory_usage = True

    # config, env, sess, policy = setup(exp, single_threaded=False)
    (config, Policy, epoch, iteration, elite, parents,
     score_stats, time_stats, acc_stats, norm_stats) = setup(exp)

    # redis master
    master = MasterClient(master_redis_cfg)

    # noise = SharedNoiseTable()
    rs = np.random.RandomState()

    # todo parameterize trainloaders
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

    # num_workers? https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/2
    # 0 means this process will load data
    num_workers = config.num_dataloader_workers if config.num_dataloader_workers else 1  # os.cpu_count()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=num_workers)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.set_grad_enabled(False)

    tstart = time.time()

    # master, virt, worker
    mem_stats = [[], [], []]

    # this puts up a redis key, value pair with the experiment
    master.declare_experiment(exp)

    # best_score = float('-inf')
    population_size = exp['population_size']
    truncation = exp['truncation']
    num_elites = exp['num_elites']      # todo use num_elites instead of 1
    min_eval_runs = int(population_size * config.eval_prob) / 2

    # todo use best so far as elite instead of best of last gen?
    policy = Policy()
    policy.set_model(elite)

    # todo also to and from info.json
    best_parents_so_far = (float('-inf'), [])
    bad_generations = 0
    current_noise_stdev = config.noise_stdev

    max_nb_epochs = config.max_nb_epochs if config.max_nb_epochs else 0
    try:
        # todo max epochs?
        while True or epoch < max_nb_epochs:
            epoch += 1

            # todo max generations
            # todo check how many times training set has been gone through
            for _, batch_data in enumerate(trainloader, 0):
                iteration += 1
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
                    noise_stdev=current_noise_stdev,
                ))

                tlogger.log('********** Iteration {} **********'.format(total_iteration))
                logger.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                curr_task_results, eval_rets, worker_ids, scored_models = [], [], [], []
                num_results_skipped = 0

                nb_models_to_evaluate = population_size
                elite_evaluated = False
                eval_runs = 0

                mem_usages = []
                worker_mem_usage = {}
                while nb_models_to_evaluate > 0 or not elite_evaluated or not eval_runs >= min_eval_runs:

                    if nb_models_to_evaluate <= 0 and not elite_evaluated:
                        logger.warning('Only the elite still has to be evaluated')

                    if nb_models_to_evaluate <= 0 and not eval_runs >= min_eval_runs:
                        logger.warning('Waiting for eval runs')

                    # Wait for a result
                    task_id, result = master.pop_result()
                    assert isinstance(task_id, int) and isinstance(result, Result)

                    # https://psutil.readthedocs.io/en/latest/#memory
                    mem_usage = psutil.Process(os.getpid()).memory_info().rss
                    mem_usages.append(mem_usage)
                    if psutil.virtual_memory().percent > 80.0 and not lower_memory_usage:
                        tlogger.warn('MEMORY USAGE TOO HIGH, GOING INTO LOWER MEM MODE')
                        lower_memory_usage = True

                    worker_ids.append(result.worker_id)
                    value = max(result.mem_usage, worker_mem_usage[result.worker_id]) \
                        if result.worker_id in worker_mem_usage else result.mem_usage
                    worker_mem_usage.update({result.worker_id: value})

                    if result.eval_return is not None:
                        # This was an eval job
                        # Store the result only for current tasks
                        if task_id == curr_task_id and eval_runs < min_eval_runs:
                            eval_rets.append(result.eval_return)
                            eval_runs += 1

                    elif result.evaluated_model_id is not None:
                        # assert result.returns_n2.dtype == np.float32
                        assert result.fitness.dtype == np.float32

                        # Store results only for current tasks
                        if task_id == curr_task_id:

                            nb_models_to_evaluate -= 1

                            # uncomment if mem usage too high
                            if lower_memory_usage:
                                scored_models.append((result.evaluated_model_id, result.evaluated_model,
                                                      result.fitness.item()))
                                elite_scored_models = [t for t in scored_models if t[0] == 0]
                                other_scored_models = [t for t in scored_models if t[0] != 0]

                                if elite_scored_models:
                                    best_elite_score = [(
                                        0, elite_scored_models[0][1],
                                        max([score for _, _, score in elite_scored_models])
                                    )]
                                else:
                                    best_elite_score = []

                                scored_models = best_elite_score + other_scored_models
                                scored_models.sort(key=lambda x: x[2], reverse=True)
                                scored_models = scored_models[:truncation]
                                del elite_scored_models, other_scored_models, best_elite_score
                            else:
                                curr_task_results.append(result)

                            if result.evaluated_model_id == 0:
                                elite_evaluated = True

                        else:
                            num_results_skipped += 1
                    else:
                        # first generation eval runs are empty
                        pass

                # Compute skip fraction
                # frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
                # if num_results_skipped > 0:
                #     logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                #         num_results_skipped, 100. * frac_results_skipped))

                # comment if mem usage too high
                if not lower_memory_usage:
                    scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                                     for result in curr_task_results]
                    elite_scored_models = [t for t in scored_models if t[0] == 0]
                    other_scored_models = [t for t in scored_models if t[0] != 0]

                    best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))

                    scored_models = [best_elite_score] + other_scored_models
                else:
                    tlogger.warn('MEMORY USAGE TOO HIGH, IN LOWER MEM MODE')

                scored_models.sort(key=lambda x: x[2], reverse=True)
                scores = np.array([fitness for (_, _, fitness) in scored_models])

                # pick parents for next generation, give new index
                parents = [(i, model) for (i, (_, model, _)) in enumerate(scored_models[:truncation])]

                if not best_parents_so_far[1] or (scores.max() > best_parents_so_far[0] + 10):
                    best_parents_so_far = (scores.max(), copy.deepcopy(parents))
                elif config.patience:
                    bad_generations += 1
                    if bad_generations > config.patience:
                        # todo tlogger like logger
                        logger.warning('MAX PATIENCE REACHED, SETTING LOWER NOISE STD DEV')
                        # todo also set parents to best parents
                        current_noise_stdev /= config.stdev_decr_divisor
                        bad_generations = 0
                        parents = best_parents_so_far[1]

                logger.info('Best 5: {}'.format([(i, round(f, 2)) for (i, _, f) in scored_models[:5]]))
                del scored_models
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

                mem_usage = max(mem_usages) if mem_usages else 0
                mem_stats[0].append(mem_usage)
                mem_stats[1].append(psutil.virtual_memory().percent)

                num_unique_workers = len(set(worker_ids))
                mem_stats[2].append(sum(worker_mem_usage.values()) / num_unique_workers)

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
                tlogger.record_tabular("NoiseStd", current_noise_stdev)

                if eval_rets:
                    tlogger.record_tabular("MaxAcc", acc_stats[1][-1])

                tlogger.record_tabular("UniqueWorkers", num_unique_workers)
                tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
                # tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)

                tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
                tlogger.record_tabular("TimeElapsed", step_tend - tstart)
                tlogger.record_tabular("MemUsage", psutil.virtual_memory().percent)
                tlogger.dump_tabular()

                if config.snapshot_freq != 0 and total_iteration % config.snapshot_freq == 0:

                    filename = save_snapshot(acc_stats, time_stats, norm_stats, score_stats,
                                             epoch, iteration, parents, policy, len(trainloader))

                    # todo adjust tlogger to log like logger (time, pid)
                    logger.info('Saved snapshot {}'.format(filename))

                    if plot:
                        plot_stats(log_dir, score_stats,
                                   time=(time_stats, 'Time per gen'),
                                   norm=(norm_stats, 'Norm of params'),
                                   # todo also plot eval fitness
                                   acc=(acc_stats[1], 'Elite accuracy'),
                                   master_mem=(mem_stats[0], 'Master mem usage'),
                                   worker_mem=(mem_stats[2], 'Worker mem usage'),
                                   virtmem=(mem_stats[1], 'Virt mem usage'))

                # set policy to new elite
                elite = parents[0][1]
                policy.set_model(elite)

            iteration = 0

    except KeyboardInterrupt:
        if plot:
            plot_stats(log_dir, score_stats,
                       time=(time_stats, 'Time per gen'),
                       norm=(norm_stats, 'Norm of params'),
                       # todo also plot fitness
                       acc=(acc_stats[1], 'Elite accuracy'),
                       master_mem=(mem_stats[0], 'Master mem usage'),
                       worker_mem=(mem_stats[2], 'Worker mem usage'),
                       virtmem=(mem_stats[1], 'Virt mem usage'))
        filename = save_snapshot(acc_stats, time_stats, norm_stats, score_stats,
                                 epoch, iteration, parents, policy, len(trainloader))

        logger.info('Saved snapshot {}'.format(filename))


@profile(stream=open('profile/memory_profile_worker.log', 'w+'))
def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    torch.set_grad_enabled(False)

    # redis client
    worker = WorkerClient(master_redis_cfg, relay_redis_cfg)

    exp = worker.get_experiment()
    # config, env, sess, policy = setup(exp, single_threaded=True)
    config, Policy, *_ = setup(exp)

    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)
    # todo worker_id random int???? what if two get the same?

    i = 0
    while i < 1000:
        time.sleep(1)
        i += 1
        mem_usages = []
        policy = Policy()

        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, GATask)

        if rs.rand() < config.eval_prob:
            model = copy.deepcopy(task_data.elite)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            policy.set_model(model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            fitness = policy.rollout(task_data.batch_data)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            accuracy = policy.accuracy_on(task_data.batch_data)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                eval_return=(fitness, accuracy),
                mem_usage=max(mem_usages)
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

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            policy.set_model(model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            fitness = policy.rollout(task_data.batch_data)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                evaluated_model_id=parent_id,
                evaluated_model=model,
                fitness=np.array([fitness], dtype=np.float32),
                mem_usage=max(mem_usages)
            ))

        # del task_data
