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

from algorithm.nic_nes.iteration import NESIteration
from dist import MasterClient
from algorithm.nic_nes.experiment import NESExperiment
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_master
from algorithm.tools.snapshot import save_snapshot
from algorithm.tools.statistics import Statistics


nes_task_fields = ['current', 'batch_data', 'noise_stdev', 'log_dir', 'ref_batch', 'batch_size']
NESTask = namedtuple('NESTask', field_names=nes_task_fields, defaults=(None,) * len(nes_task_fields))

result_fields = ['worker_id', 'eval_score', 'evolve_noise', 'fitness', 'mem_usage']
NESResult = namedtuple('NESResult', field_names=result_fields, defaults=(None,) * len(result_fields))


class NESMaster(object):

    def __init__(self, exp, master_redis_cfg):
        setup_tuple = setup_master(exp)
        self.config: Config = setup_tuple[0]
        self.policy: Policy = setup_tuple[1]
        self.stats: Statistics = setup_tuple[2]
        self.it: NESIteration = setup_tuple[3]
        self.experiment: NESExperiment = setup_tuple[4]

        self.policy.set_model(self.it.current_model())
        self.optimizer = self.experiment.init_optimizer(self.policy.parameter_vector().numpy())

        # redis master
        self.master = MasterClient(master_redis_cfg)
        # this puts up a redis key, value pair with the experiment
        self.master.declare_experiment(exp)

        self.rs = np.random.RandomState()

    # Uncomment to profile memory usage
    # @profile(stream=open('../output/memory_profile_worker.txt', 'w+'))
    def run_master(self, plot: bool):
        """
        Main method to run NES master
        """
        logging.info('run_master: {}'.format(locals()))

        config, experiment, rs, master, policy, stats, it, optimizer = \
            self.config, self.experiment, self.rs, self.master, self.policy, self.stats, self.it, self.optimizer

        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        try:
            while not config.max_nb_iterations or it.iteration() < config.max_nb_iterations:
                it.incr_epoch()

                for batch_data in experiment.get_trainloader():
                    gc.collect()
                    it.incr_iteration()
                    stats.set_step_tstart()

                    # publish task to relay so workers can take it
                    data = copy.deepcopy(batch_data)
                    curr_task_id = master.declare_task(NESTask(
                        current=it.current_model(),
                        batch_data=data,
                        noise_stdev=it.get_noise_stdev(),
                        ref_batch=experiment.get_ref_batch(),
                        batch_size=it.batch_size()
                    ))

                    logging.info('********** Iteration {} **********'.format(it.iteration()))
                    logging.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                    stats.reset_it_mem_usages()

                    while it.models_left_to_evolve() or it.models_left_to_eval():

                        # this is just for logging
                        it.warn_waiting_for_evaluations()

                        # wait for a result
                        task_id, result = master.pop_result()
                        assert isinstance(task_id, int) and isinstance(result, NESResult)

                        # some memory usage tracking
                        # https://psutil.readthedocs.io/en/latest/#memory
                        master_mem_usage = psutil.Process(os.getpid()).memory_info().rss
                        stats.record_it_master_mem_usage(master_mem_usage)
                        stats.record_it_worker_mem_usage(result.worker_id, result.mem_usage)

                        it.record_worker_id(result.worker_id)

                        if result.eval_score is not None:
                            # this was an evaluation job, store the result only for current tasks
                            if task_id == curr_task_id:
                                it.record_eval_result(result)

                        elif result.fitness is not None:
                            # this was an evolution job, so contains mutation and score
                            # store results only for current tasks
                            if task_id == curr_task_id:
                                it.record_task_result(result)

                    it.process_evaluated_elites()

                    # compute a gradient estimate from the mutations and the scores
                    grad_estimate = self.gradient_estimate(it.fitnesses(), it.noise_vecs())
                    update_ratio, theta = optimizer.update(
                        # caution l2 * theta is correct because L2 regularization adds a (1/2)* l2 * sum(theta^2) term
                        # to the loss function, the derivative of this w.r.t. theta = l2 * theta
                        - grad_estimate + config.l2coeff * policy.parameter_vector().numpy()
                    )

                    # set the policy to the resulting parameters
                    policy.set_from_parameter_vector(vector=theta)
                    it.set_model(policy.get_model())

                    if it.patience_reached() or it.schedule_reached():
                        experiment.increase_loader_batch_size(it.batch_size())
                        optimizer.stepsize /= config.stepsize_divisor

                    stats.record_update_ratio(update_ratio)
                    stats.record_score_stats(it.flat_fitnesses())
                    stats.record_bs_stats(it.batch_size())
                    stats.record_step_time_stats()
                    stats.record_norm_stats(policy.parameter_vector())
                    stats.record_acc_stats(it.score())
                    stats.record_best_acc_stats(it.best_elites()[0][1])
                    stats.record_std_stats(it.noise_stdev())
                    stats.update_mem_stats()

                    stats.log_stats()
                    it.log_stats()

                    if it.patience_reached() or it.schedule_reached():
                        # to use new trainloader when increased batch size!
                        break

                    if config.snapshot_freq != 0 and it.iteration() % config.snapshot_freq == 0:
                        save_snapshot(stats, it, experiment)
                        if plot:
                            stats.plot_stats(experiment.snapshot_dir())

        except KeyboardInterrupt:
            save_snapshot(stats, it, experiment)
            if plot:
                stats.plot_stats(experiment.snapshot_dir())

    def gradient_estimate(self, fitnesses, noise_vecs):
        """
        :param fitnesses: numpy array with size (F, 2)
                          contains two scores per mutation vector in noise_vecs
                          --> one for theta + delta and one for theta - delta (mirrored sampling)
        :param noise_vecs: numpy array with size (F, dim(theta))
        :return: numpy array of dimension (dim(theta),)
        """
        ranked_fitnesses = self.compute_centered_ranks(fitnesses)
        weights = ranked_fitnesses[:, 0] - ranked_fitnesses[:, 1]
        gradient_est, _ = self.batched_weighted_sum(weights, noise_vecs)
        gradient_est /= ranked_fitnesses.size
        return gradient_est

    def compute_centered_ranks(self, x):
        """
        Taken from https://github.com/openai/evolution-strategies-starter
        Return a np.array with an element per element in x that indicates the corresponding
            elements rank with a float between -0.5 and 0.5
        E.g. [[101, 200], [2, 100]] --> [[0.16666667, 0.5], [-0.5, -0.16666667]]
        """
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float)
        y /= (x.size - 1)
        y -= .5
        return y

    def compute_ranks(self, x):
        """
        Taken from https://github.com/openai/evolution-strategies-starter
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    @staticmethod
    def batched_weighted_sum(weights, vecs, batch_size=500):
        """
        Taken from https://github.com/openai/evolution-strategies-starter

        Used to compute a dot product in batches to reduce memory usage
        """
        total = 0.
        num_items_summed = 0
        for batch_weights, batch_vecs in zip(NESMaster.itergroups(weights, batch_size),
                                             NESMaster.itergroups(vecs, batch_size)):
            assert len(batch_weights) == len(batch_vecs) <= batch_size
            total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
            num_items_summed += len(batch_weights)
        return total, num_items_summed

    @staticmethod
    def itergroups(items, group_size):
        """
        Taken from https://github.com/openai/evolution-strategies-starter
        """
        assert group_size >= 1
        group = []
        for x in items:
            group.append(x)
            if len(group) == group_size:
                yield tuple(group)
                del group[:]
        if group:
            yield tuple(group)
