"""
    Contains code and ideas from https://github.com/openai/evolution-strategies-starter
                            and https://github.com/uber-research/deep-neuroevolution
    Largest part is written by myself
"""
import copy
import gc
import logging
import os
from collections import namedtuple

import psutil

import numpy as np
import torch
# from memory_profiler import profile

from dist import MasterClient
from algorithm.nic_es.experiment import ESExperiment
from algorithm.nic_es.iteration import ESIteration
from algorithm.policies import Policy
from algorithm.tools.setup import setup_master, Config
from algorithm.tools.snapshot import save_snapshot
from algorithm.tools.statistics import Statistics


result_fields = ['worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
                 'eval_return', 'mem_usage', 'evaluated_cand', 'evaluated_cand_id',
                 'score']
ESResult = namedtuple('ESResult', field_names=result_fields, defaults=(None,) * len(result_fields))

es_task_fields = ['elite', 'population', 'batch_data', 'parents', 'noise_stdev',
                  'log_dir', 'elites', 'ref_batch']
ESTask = namedtuple('ESTask', field_names=es_task_fields, defaults=(None,) * len(es_task_fields))


class ESMaster(object):

    def __init__(self, exp, master_redis_cfg):
        setup_tuple = setup_master(exp)
        self.config: Config = setup_tuple[0]
        self.policy: Policy = setup_tuple[1]
        self.stats: Statistics = setup_tuple[2]
        self.it: ESIteration = setup_tuple[3]
        self.experiment: ESExperiment = setup_tuple[4]

        # redis master
        self.master = MasterClient(master_redis_cfg)
        # this puts up a redis key, value pair with the experiment
        self.master.declare_experiment(exp)

        self.rs = np.random.RandomState()

    # @profile(stream=open('../output/memory_profile_worker.txt', 'w+'))
    def run_master(self, plot):
        logging.info('run_master: {}'.format(locals()))

        config, experiment, rs, master, policy, stats, it = \
            self.config, self.experiment, self.rs, self.master, self.policy, self.stats, self.it

        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        try:
            while not config.max_nb_iterations or it.iteration() < config.max_nb_iterations:
                it.incr_epoch()

                for batch_data in experiment.get_trainloader():
                    gc.collect()
                    it.incr_iteration()
                    stats.set_step_tstart()

                    # publish task
                    data = copy.deepcopy(batch_data)
                    curr_task_id = master.declare_task(ESTask(
                        elites=it.elites_to_evaluate(),
                        parents=it.parents(),
                        batch_data=data,
                        noise_stdev=it.get_noise_stdev(),
                        ref_batch=experiment.get_ref_batch(),
                    ))

                    logging.info('********** Iteration {} **********'.format(it.iteration()))
                    logging.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                    stats.reset_it_mem_usages()

                    while it.models_left_to_evolve() or it.models_left_to_eval():

                        # this is just for logging
                        it.warn_waiting_for_evaluations()

                        # wait for a result
                        task_id, result = master.pop_result()
                        assert isinstance(task_id, int) and isinstance(result, ESResult)

                        # https://psutil.readthedocs.io/en/latest/#memory
                        master_mem_usage = psutil.Process(os.getpid()).memory_info().rss
                        stats.record_it_master_mem_usage(master_mem_usage)
                        stats.record_it_worker_mem_usage(result.worker_id, result.mem_usage)

                        it.record_worker_id(result.worker_id)

                        if result.evaluated_cand_id is not None:
                            # this was an eval job, store the result only for current tasks
                            if task_id == curr_task_id:
                                it.record_eval_result(result)

                        elif result.evaluated_model_id is not None:
                            # this was an evolve job, store results only for current tasks
                            if task_id == curr_task_id:
                                it.record_task_result(result)
                                if rs.rand() < config.eval_prob:
                                    logging.info('Incoming result: %.2f' % result.fitness.item())

                    best_ev_acc, best_ev_elite = it.process_evaluated_elites()
                    policy.set_model(best_ev_elite)

                    parents, scores = self.selection(it.task_results(), experiment.population_size(),
                                                     experiment.num_elites())

                    best_individuals = parents[:experiment.num_elite_cands()]
                    it.set_next_elites_to_evaluate(best_individuals)

                    it.record_parents(parents)
                    if it.patience_reached() or it.schedule_reached():
                        experiment.increase_loader_batch_size(it.batch_size())

                    stats.record_score_stats(scores)
                    stats.record_bs_stats(it.batch_size())
                    stats.record_step_time_stats()
                    stats.record_norm_stats(policy.parameter_vector())
                    stats.record_acc_stats(best_ev_acc)
                    stats.record_best_acc_stats(it.best_elites()[0][1])
                    stats.record_std_stats(it.noise_stdev())
                    stats.update_mem_stats()

                    stats.log_stats()
                    it.log_stats()

                    if it.patience_reached() or it.schedule_reached():
                        # to use new trainloader!
                        break

                    if config.snapshot_freq != 0 and it.iteration() % config.snapshot_freq == 0:
                        save_snapshot(stats, it, experiment)
                        if plot:
                            stats.plot_stats(experiment.snapshot_dir())

        except KeyboardInterrupt:
            save_snapshot(stats, it, experiment)
            if plot:
                stats.plot_stats(experiment.snapshot_dir())

    @staticmethod
    def selection(curr_task_results, pop_size, num_elites):
        scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                         for result in curr_task_results]

        scored_models.sort(key=lambda x: x[2], reverse=True)
        scores = np.array([fitness for (_, _, fitness) in scored_models])

        # pick parents for next generation
        parents = [model for (_, model, _) in scored_models[:pop_size - num_elites]]

        logging.info('Best 5: {}'.format([(i, round(f, 2)) for (i, _, f) in scored_models[:5]]))
        return parents, scores
