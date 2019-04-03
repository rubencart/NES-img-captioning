import copy
import gc
import logging
import os
import time

import psutil

import numpy as np
import torch
# from memory_profiler import profile

from dist import MasterClient
from algorithm.tools.experiment import Experiment
from algorithm.tools.iteration import Iteration
from algorithm.policies import Policy
from algorithm.tools.setup import setup_master, Config
from algorithm.tools.snapshot import save_snapshot
from algorithm.tools.statistics import Statistics
from algorithm.tools.utils import GATask, Result, remove_all_files_from_dir, IterationFailedException

logger = logging.getLogger(__name__)


# todo cpu profile_exp on server! --> a lot of waiting in workers??
# - snelste lijkt 2 workers --> hele master / worker setup nog nodig?
# - memory problemen door CompressedModel? Misschien zonder hele serializatie proberen?
# - meer CPUs wel veel sneller? Misschien toch GPU overwegen voor snellere FW?

# from meeting
# - VCS --> slides graham
# - start workers from CL

# - multiple elites!
# - safe mutations
# - beam search
# - scheduled sampling (ss_prob in self-critical repo)
# - different models from paper, att, resnet feats,...
# - try self critical reward instead of pure cider
# - improved exploration --> NS/RS?
# - check if we are using cider right (https://github.com/ruotianluo/cider/)

# next things:
# x implement test on val set!
# x keep overall best elite ( a la early stopping )
# x init from SINGLE pretrained
# - PROFILE run on server
# - cococaption uses CIDEr and not CIDErD
# x improve eval run: entire valid set?
# x snapshot in SEPARATE FOLDER! plots etc
# - FIX SNAPSHOT --> now: snapshot saves paths to parents in infos
#                    next generation: parents are deleted
#                    until next snapshot: infos still points to non existing files!
#   --> fix by using generic names for parents/elite? {i}_parent.pt
#       easy fix and works but technically not correct (stats in infos about other generation than
#       param files)
#   --> save snapshot of parents to other loc, eg snapshot/ dir in log_dir
#       --> adjust snapshot code to make copies of files AND to save paths to these files
#           instead of files in models/parents/ !!!
# - num elites instead of 1 elite!
# - PROFILE WORKER ON SERVER!
# - look at (i, parent) --> index we always keep --> NECESSARY??
# - MSCocoExperiment class from experiment.py to captioning module
# x add code to worker that checks if already too many files in dir and breaks if so!
# - add some copy.deepcopy() !!
# - options for capt models --> find nicer way! spaghetti with setup now
# - lstm/gru/...?
# - label axes (name + units) in plots!!!!
# - get rid of tlogger (just use logger but also dump to file like tlogger)
# - assertions, param checks,...


class GAMaster(object):

    def run_master(self, master_redis_cfg, exp, plot):
        logger.info('run_master: {}'.format(locals()))

        setup_tuple = setup_master(exp)
        config: Config = setup_tuple[0]
        policy: Policy = setup_tuple[1]
        stats: Statistics = setup_tuple[2]
        it: Iteration = setup_tuple[3]
        experiment: Experiment = setup_tuple[4]

        from algorithm.tools import tabular_logger as tlogger
        logger.info('Tabular logging to {}'.format(experiment.snapshot_dir()))
        tlogger.start(experiment.snapshot_dir())

        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        # redis master
        master = MasterClient(master_redis_cfg)

        rs = np.random.RandomState()

        # this puts up a redis key, value pair with the experiment
        master.declare_experiment(exp)

        max_nb_epochs = config.max_nb_epochs if config.max_nb_epochs else 0

        # elite = it.elite()
        try:
            # todo max epochs?
            while True or it.epoch() < max_nb_epochs:
                it.incr_epoch()

                # todo max generations
                for _, batch_data in enumerate(experiment.trainloader, 0):
                    try:
                        gc.collect()
                        it.incr_iteration()
                        stats.set_step_tstart()

                        # logging.info('declaring task')
                        curr_task_id = master.declare_task(GATask(
                            elite=it.elite(),
                            # val_data=next(iter(experiment.valloader)),
                            # val_loader=copy.deepcopy(experiment.valloader),
                            # val_loader=experiment.valloader,
                            parents=it.parents(),
                            # todo batch & val data to disk as well
                            batch_data=batch_data,
                            noise_stdev=it.get_noise_stdev(),
                            # log_dir=experiment.log_dir()
                        ))
                        # logging.info('declared task')

                        tlogger.log('********** Iteration {} **********'.format(it.iteration()))
                        logger.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                        it.reset_task_results()
                        it.reset_eval_returns()
                        it.reset_worker_ids()

                        it.set_nb_models_to_evaluate(experiment.population_size())
                        it.set_waiting_for_eval_run(False)
                        it.set_waiting_for_elite_eval(False)
                        it.set_elite_evaluated(False)
                        stats.reset_it_mem_usages()

                        # logging.info('going into while true loop')
                        while it.models_left_to_evaluate() or not it.elite_evaluated() or not it.eval_ran():
                            # if it.models_left_to_evaluate():
                            #     logging.info('models left')
                            # if not it.elite_evaluated():
                            #     logging.info('elite not evaluated')
                            # if it.eval_ran():
                            #     logging.info('no eval runs')

                            # this is just for logging
                            it.warn_elite_evaluated()
                            it.warn_eval_run()

                            # if len(os.listdir(experiment.offspring_dir())) > 2 * experiment.population_size():
                            #     time.sleep(2)
                            #     try:
                            #         remove_all_files_from_dir(experiment.offspring_dir())
                            #     finally:
                            #         raise IterationFailedException()

                            # wait for a result
                            task_id, result = master.pop_result()
                            assert isinstance(task_id, int) and isinstance(result, Result)

                            # https://psutil.readthedocs.io/en/latest/#memory
                            master_mem_usage = psutil.Process(os.getpid()).memory_info().rss
                            stats.record_it_master_mem_usage(master_mem_usage)
                            stats.record_it_worker_mem_usage(result.worker_id, result.mem_usage)

                            it.record_worker_id(result.worker_id)

                            if result.eval_return is not None:
                                # this was an eval job, store the result only for current tasks
                                if task_id == curr_task_id and not it.eval_ran():
                                    it.record_eval_return(result.eval_return)
                                    it.set_waiting_for_eval_run(False)

                                # elif task_id != curr_task_id:
                                #     logging.info('ER, NOT EQ: task id {} - curr {}'.format(task_id, curr_task_id))

                            elif result.evaluated_model_id is not None:
                                # assert result.returns_n2.dtype == np.float32
                                assert result.fitness.dtype == np.float32

                                # store results only for current tasks
                                if task_id == curr_task_id:

                                    it.decr_nb_models_to_evaluate()
                                    it.record_task_result(result)

                                    if result.evaluated_model_id == 0:
                                        it.set_elite_evaluated(True)
                                        it.set_waiting_for_elite_eval(False)

                                # else:
                                #     logging.info('EV, NOT EQ: task id {} - curr {}'.format(task_id, curr_task_id))

                        # logging.info('Out of while true loop')

                        elite_acc = it.max_eval_return()
                        it.record_elite(elite_acc)

                        parents, scores = self._selection(it.task_results(), experiment.truncation())

                        elite = parents[0][1]
                        policy.set_model(elite)
                        it.set_elite(elite)

                        # elite twice in parents: once to have an unmodified copy in next gen,
                        # once for possible offspring
                        parents.append((len(parents), copy.copy(elite)))

                        reset_parents = it.record_parents(parents, scores.max())
                        if reset_parents:
                            parents = reset_parents
                            experiment.increase_loader_batch_size(it.batch_size())

                        stats.record_score_stats(scores)
                        stats.record_bs_stats(it.batch_size())
                        stats.record_step_time_stats()
                        stats.record_norm_stats(policy.parameter_vector())
                        stats.record_acc_stats(elite_acc)
                        stats.record_std_stats(it.noise_stdev())
                        stats.update_mem_stats()

                        stats.log_stats(tlogger)
                        it.log_stats(tlogger)
                        tlogger.dump_tabular()

                        # logging.info('saving snap')
                        if config.snapshot_freq != 0 and it.iteration() % config.snapshot_freq == 0:
                            save_snapshot(stats, it, experiment, policy)
                            if plot:
                                stats.plot_stats(experiment.snapshot_dir())
                        # logging.info('saved snap')

                    except IterationFailedException:
                        pass

        except KeyboardInterrupt:
            save_snapshot(stats, it, experiment, policy)
            if plot:
                stats.plot_stats(experiment.snapshot_dir())

    def _selection(self, curr_task_results, truncation):
        scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                         for result in curr_task_results]
        elite_scored_models = [t for t in scored_models if t[0] == 0]
        other_scored_models = [t for t in scored_models if t[0] != 0]

        best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))

        scored_models = [best_elite_score] + other_scored_models

        scored_models.sort(key=lambda x: x[2], reverse=True)
        scores = np.array([fitness for (_, _, fitness) in scored_models])

        # pick parents for next generation, give new index              todo -1 or not
        parents = [(i, model) for (i, (_, model, _)) in enumerate(scored_models[:truncation - 1])]
        # rest = [model for (_, model, _) in scored_models[truncation - 1:]]

        logger.info('Best 5: {}'.format([(i, round(f, 2)) for (i, _, f) in scored_models[:5]]))
        return parents, scores  # , rest

    # def _remove_truncated(self, parents, elite, directory, exp):
    #     if exp.mode() == 'seeds':
    #         return
    #
    #     to_keep = [parent for _, parent in parents] + [elite]
    #     remove_all_files_but(from_dir=directory, but_list=to_keep)
