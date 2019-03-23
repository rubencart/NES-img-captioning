import copy
import gc
import logging
import os
import psutil
import time
from collections import namedtuple

from setup import init_trainldr, setup

print('importing mkl, setting num threads')
import mkl
mkl.set_num_threads(1)

print('importing torch')
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# from memory_profiler import profile

from dist import MasterClient, WorkerClient
from policies import CompressedModel
from utils import plot_stats, save_snapshot, mkdir_p

logger = logging.getLogger(__name__)

# todo clean
ga_task_fields = ['elite', 'population', 'ob_mean', 'ob_std',
                  'timestep_limit', 'batch_data', 'parents', 'noise_stdev']
GATask = namedtuple('GATask', field_names=ga_task_fields, defaults=(None,) * len(ga_task_fields))

Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])

result_fields = [
    'worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count', 'mem_usage'
]
Result = namedtuple('Result', field_names=result_fields, defaults=(None,) * len(result_fields))


# todo cpu profile_exp on server! --> a lot of waiting in workers??
# - snelste lijkt 2 workers --> hele master / worker setup nog nodig?
# - memory problemen door CompressedModel? Misschien zonder hele serializatie proberen?
# - meer CPUs wel veel sneller? Misschien toch GPU overwegen voor snellere FW?

# from meeting
# - VCS --> slides graham
# x gc.collect(), doesn't help
# x mkl.set_num_threads(1), test on server
# - start workers from CL
# - leave seeds altogether

# next things:
# x make possible to start from 1 net .pt file, to pretrain with SGD
# - implement test on test set!
# - get rid of tlogger (just use logger but also dump to file like tlogger)
def run_master(master_redis_cfg, exp, log_dir, plot):

    (config, Policy, epoch, iteration, elite, parents,
     score_stats, time_stats, acc_stats, norm_stats, trainloader) = setup(exp)

    logger.info('run_master: {}'.format(locals()))

    # todo to utils
    log_dir = os.path.expanduser(log_dir) if log_dir else 'logs/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)

    import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)

    import matplotlib.pyplot as plt

    # redis master
    master = MasterClient(master_redis_cfg)
    torch.set_grad_enabled(False)

    rs = np.random.RandomState()
    tstart = time.time()

    # master, virt, worker
    mem_stats = [[], [], []]

    # this puts up a redis key, value pair with the experiment
    master.declare_experiment(exp)

    # best_score = float('-inf')
    population_size = exp['population_size']
    truncation = exp['truncation']
    num_elites = exp['num_elites']      # todo use num_elites instead of 1

    # todo use best so far as elite instead of best of last gen?
    policy = Policy()
    policy.set_model(elite)

    # todo also to and from info.json
    best_parents_so_far = (float('-inf'), [])
    bad_generations = 0
    current_noise_stdev = config.noise_stdev
    noise_std_stats = []

    batch_size = config.batch_size

    # lower_memory_usage = False

    max_nb_epochs = config.max_nb_epochs if config.max_nb_epochs else 0
    try:
        # todo max epochs?
        while True or epoch < max_nb_epochs:
            epoch += 1

            # todo max generations
            for _, batch_data in enumerate(trainloader, 0):
                gc.collect()

                iteration += 1
                total_iteration = ((epoch - 1) * len(trainloader) + iteration)
                step_tstart = time.time()

                curr_task_id = master.declare_task(GATask(
                    elite=elite,
                    parents=parents,
                    batch_data=batch_data,
                    noise_stdev=current_noise_stdev,
                ))

                tlogger.log('********** Iteration {} **********'.format(total_iteration))
                logger.info('Searching {nb} params for NW'.format(nb=Policy.nb_learnable_params()))

                curr_task_results, eval_rets, worker_ids, scored_models = [], [], [], []
                num_results_skipped = 0

                nb_models_to_evaluate = population_size
                elite_evaluated = False
                eval_ran = False

                mem_usages = []
                worker_mem_usage = {}
                while nb_models_to_evaluate > 0 or not elite_evaluated or not eval_ran:

                    if nb_models_to_evaluate <= 0 and not elite_evaluated:
                        logger.warning('Only the elite still has to be evaluated')

                    if nb_models_to_evaluate <= 0 and not eval_ran:
                        logger.warning('Waiting for eval runs')

                    # Wait for a result
                    task_id, result = master.pop_result()
                    assert isinstance(task_id, int) and isinstance(result, Result)

                    # https://psutil.readthedocs.io/en/latest/#memory
                    mem_usage = psutil.Process(os.getpid()).memory_info().rss
                    mem_usages.append(mem_usage)
                    # if psutil.virtual_memory().percent > 80.0 and not lower_memory_usage:
                    #     logger.warning('MEMORY USAGE TOO HIGH, GOING INTO LOWER MEM MODE')
                    #     lower_memory_usage = True

                    worker_ids.append(result.worker_id)
                    value = max(result.mem_usage, worker_mem_usage[result.worker_id]) \
                        if result.worker_id in worker_mem_usage else result.mem_usage
                    worker_mem_usage.update({result.worker_id: value})

                    if result.eval_return is not None:
                        # This was an eval job
                        # Store the result only for current tasks
                        if task_id == curr_task_id and not eval_ran:
                            eval_rets.append(result.eval_return)
                            eval_ran = True

                    elif result.evaluated_model_id is not None:
                        # assert result.returns_n2.dtype == np.float32
                        assert result.fitness.dtype == np.float32

                        # Store results only for current tasks
                        if task_id == curr_task_id:

                            nb_models_to_evaluate -= 1

                            # uncomment if mem usage too high
                            # if lower_memory_usage:
                            #     scored_models.append((result.evaluated_model_id, result.evaluated_model,
                            #                           result.fitness.item()))
                            #     elite_scored_models = [t for t in scored_models if t[0] == 0]
                            #     other_scored_models = [t for t in scored_models if t[0] != 0]
                            #
                            #     if elite_scored_models:
                            #         best_elite_score = [(
                            #             0, elite_scored_models[0][1],
                            #             max([score for _, _, score in elite_scored_models])
                            #         )]
                            #     else:
                            #         best_elite_score = []
                            #
                            #     scored_models = best_elite_score + other_scored_models
                            #     scored_models.sort(key=lambda x: x[2], reverse=True)
                            #     scored_models = scored_models[:truncation]
                            #     del elite_scored_models, other_scored_models, best_elite_score
                            # else:
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
                # if not lower_memory_usage:
                scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                                 for result in curr_task_results]
                elite_scored_models = [t for t in scored_models if t[0] == 0]
                other_scored_models = [t for t in scored_models if t[0] != 0]

                best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))

                scored_models = [best_elite_score] + other_scored_models
                # else:
                #     logger.warning('MEMORY USAGE TOO HIGH, IN LOWER MEM MODE')

                scored_models.sort(key=lambda x: x[2], reverse=True)
                scores = np.array([fitness for (_, _, fitness) in scored_models])

                # pick parents for next generation, give new index
                parents = [(i, model) for (i, (_, model, _)) in enumerate(scored_models[:truncation])]

                if not best_parents_so_far[1] or (scores.max() > best_parents_so_far[0]):
                    best_parents_so_far = (scores.max(), copy.deepcopy(parents))
                    bad_generations = 0
                elif config.patience:
                    logger.info('BAD GENERATION')
                    bad_generations += 1
                    if bad_generations > config.patience:
                        # todo tlogger like logger
                        logger.warning('Max patience reached; setting lower noise stdev & bigger batch_size')
                        # todo also set parents to best parents
                        current_noise_stdev /= config.stdev_decr_divisor
                        bad_generations = 0
                        parents = best_parents_so_far[1]
                        # todo default_best_parents_so_far() --> also with a lot of other stuff
                        best_parents_so_far = (float('-inf'), [])

                        batch_size *= 2
                        trainloader = init_trainldr(exp, batch_size=batch_size)

                logger.info('Best 5: {}'.format([(i, round(f, 2)) for (i, _, f) in scored_models[:5]]))
                # input('PRESS ENTER')

                score_stats[0].append(scores.min())
                score_stats[1].append(scores.mean())
                score_stats[2].append(scores.max())

                step_tend = time.time()
                time_stats.append(step_tend - step_tstart)

                norm = float(np.sqrt(np.square(policy.parameter_vector()).sum()))
                norm_stats.append(norm)

                # todo also keep conf interv
                acc_stats[0].append(max([fit for fit, _ in eval_rets]) if eval_rets else 0)
                acc_stats[1].append(max([acc for _, acc in eval_rets]) if eval_rets else 0)

                mem_usage = max(mem_usages) if mem_usages else 0
                mem_stats[0].append(mem_usage)
                mem_stats[1].append(psutil.virtual_memory().percent)

                num_unique_workers = len(set(worker_ids))
                mem_stats[2].append(sum(worker_mem_usage.values()) / num_unique_workers)

                noise_std_stats.append(current_noise_stdev)

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
                tlogger.record_tabular("BatchSize", batch_size)

                if eval_rets:
                    tlogger.record_tabular("MaxAcc", acc_stats[1][-1])

                if config.patience:
                    tlogger.record_tabular("BadGen", str(bad_generations) + '/' + str(config.patience))

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
                        plot_stats(log_dir, plt, score_stats,
                                   time=(time_stats, 'Time per gen'),
                                   norm=(norm_stats, 'Norm of params'),
                                   # todo also plot eval fitness
                                   acc=(acc_stats[1], 'Elite accuracy'),
                                   master_mem=(mem_stats[0], 'Master mem usage'),
                                   worker_mem=(mem_stats[2], 'Worker mem usage'),
                                   virtmem=(mem_stats[1], 'Virt mem usage'),
                                   noise_std=(noise_std_stats, 'Noise stdev'))

                # set policy to new elite
                elite = parents[0][1]
                policy.set_model(elite)

            iteration = 0

    except KeyboardInterrupt:
        if plot:
            plot_stats(log_dir, plt, score_stats,
                       time=(time_stats, 'Time per gen'),
                       norm=(norm_stats, 'Norm of params'),
                       # todo also plot fitness
                       acc=(acc_stats[1], 'Elite accuracy'),
                       master_mem=(mem_stats[0], 'Master mem usage'),
                       worker_mem=(mem_stats[2], 'Worker mem usage'),
                       virtmem=(mem_stats[1], 'Virt mem usage'),
                       noise_std=(noise_std_stats, 'Noise stdev'))
        filename = save_snapshot(acc_stats, time_stats, norm_stats, score_stats,
                                 epoch, iteration, parents, policy, len(trainloader))

        logger.info('Saved snapshot {}'.format(filename))


# @profile_exp(stream=open('profile_exp/memory_profile_worker.log', 'w+'))
def run_worker(master_redis_cfg, relay_redis_cfg):
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

    # i = 0
    # while i < 100:
    #     i += 1
    while True:

        gc.collect()
        policy = Policy()
        time.sleep(0.01)
        mem_usages = []

        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, GATask)

        if rs.rand() < config.eval_prob:
            model = copy.deepcopy(task_data.elite)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            # policy.set_model(model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            fitness = policy.rollout_(data=task_data.batch_data, compressed_model=model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            accuracy = policy.accuracy_on_(data=task_data.batch_data, compressed_model=model)

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

            # policy.set_model(model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            fitness = policy.rollout_(data=task_data.batch_data, compressed_model=model)

            mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                evaluated_model_id=parent_id,
                evaluated_model=model,
                fitness=np.array([fitness], dtype=np.float32),
                mem_usage=max(mem_usages)
            ))

        # del task_data
