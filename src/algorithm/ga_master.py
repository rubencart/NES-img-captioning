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
from algorithm.tools.utils import GATask, Result, IterationFailedException

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
# - torchvision.datasets.MSCOCO instead of hacky own version?
# - tournament selection instead of truncation?
# - consider excluding FC layer params?

# - https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/11 !!!
#   --> prints currently alive Tensors and Variables
#   --> no_grad blocks
#   --> placeholder tensor
#   --> computation graph references
# - https://discuss.pytorch.org/t/very-consitent-memory-leak/21038/5
#   --> disabled eval mode:
#       https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/9
#       use no_grad everywhere and not just .eval()!!!!
#   --> loss.detach() was creating a memory leakage
# - https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory
# - https://discuss.pytorch.org/t/how-to-check-memory-leak-in-a-model/22903
#   -->  torch.cuda.empty_cache()
#
# https://github.com/pytorch/pytorch/issues/13246 !!
#   --> use panda dicts and numpy arrays instead of python lists in dataloaders
#       when multiprocessing!!!!!!!

# next things:
# - PROFILE run on server
# - https://stackoverflow.com/questions/11328958/save-the-plots-into-a-pdf
#   --> save figs as pdf and not png
# - make bad generation smarter than loss
# - fix trainloader generator etc, nicer solution to incr bs than breaking loop
# - cococaption uses CIDEr and not CIDErD
# - leave unused BLEU / METEOR / ... scores out of validation run
# - model.eval() / .train() --> so no dropout is used
# - FIX SNAPSHOT --> now: snapshot saves paths to parents in infos
#                    next generation: parents are deleted
#                    until next snapshot: infos still points to non existing files!
#   --> fix by using generic names for parents/elite? {i}_parent.pt
#       easy fix and works but technically not correct (stats in infos about other generation than
#       param files)
#   --> save snapshot of parents to other loc, eg snapshot/ dir in log_dir
#       --> adjust snapshot code to make copies of files AND to save paths to these files
#           instead of files in models/parents/ !!!
# - look at (i, parent) --> index we always keep --> NECESSARY??
# - MSCocoExperiment class from experiment.py to captioning module
# - add some copy.deepcopy() !!
# - options for capt models --> find nicer way! spaghetti with setup now
# - lstm/gru/...?
# - label axes (name + units) in plots!!!!
# - get rid of tlogger (just use logger but also dump to file like tlogger)
# - assertions, param checks,...
# - split Result in EvalResult and EvolveResult
# - subclass experiment with masterexp and workerexp?

class GAMaster(object):

    # @profile(stream=open('../output/memory_profile_worker.txt', 'w+'))
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
                for batch_data in experiment.get_trainloader():
                    try:
                        gc.collect()
                        # it.incr_iteration(batch_data[1].size(0))
                        it.incr_iteration(it.times_orig_bs())
                        stats.set_step_tstart()

                        data = copy.deepcopy(batch_data)
                        curr_task_id = master.declare_task(GATask(
                            elites=it.elites_to_evaluate(),
                            # val_data=next(iter(experiment.valloader)),
                            # val_loader=copy.deepcopy(experiment.valloader),
                            # val_loader=experiment.valloader,
                            parents=it.parents(),
                            batch_data=data,
                            noise_stdev=it.get_noise_stdev(),
                        ))

                        tlogger.log('********** Iteration {} **********'.format(it.iteration()))
                        logger.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

                        # it.reset_task_results()
                        # it.reset_eval_results()
                        # it.reset_worker_ids()
                        #
                        # it.set_nb_models_to_evaluate(experiment.population_size())
                        # it.set_waiting_for_elite_ev(False)
                        stats.reset_it_mem_usages()

                        while it.models_left_to_evaluate() or it.elite_cands_left_to_evaluate():

                            # this is just for logging
                            it.warn_waiting_for_elite_evaluations()

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

                            if result.evaluated_cand_id is not None:
                                # this was an eval job, store the result only for current tasks
                                if task_id == curr_task_id:
                                    it.record_evaluated_elite_cand(result)

                            elif result.evaluated_model_id is not None:
                                # assert result.returns_n2.dtype == np.float32
                                assert result.fitness.dtype == np.float32

                                # store results only for current tasks
                                if task_id == curr_task_id:

                                    it.record_task_result(result)
                                    if rs.rand() < 0.02:
                                        logger.info('Incoming result: %.2f' % result.fitness.item())

                        best_ev_acc, best_ev_elite = it.process_evaluated_elites()
                        policy.set_model(best_ev_elite)

                        parents, scores = self._selection(it.task_results(), experiment.truncation(),
                                                          experiment.num_elites())

                        best_individuals = parents[:experiment.num_elite_cands()]
                        it.set_next_elites_to_evaluate(best_individuals)

                        # # elite twice in parents: once to have an unmodified copy in next gen,
                        # # once for possible offspring
                        # elites = it.elites()
                        # parents.extend(
                        #     [(len(parents) + i, copy.copy(elite)) for i, elite in enumerate(elites)]
                        # )

                        it.record_parents(parents, scores.max())
                        if it.patience_reached():
                            # parents = reset_parents
                            experiment.increase_loader_batch_size(it.batch_size())

                        # it.add_elites_to_parents()
                        # it.clean_offspring_dir()

                        stats.record_score_stats(scores)
                        stats.record_bs_stats(it.batch_size())
                        stats.record_step_time_stats()
                        stats.record_norm_stats(policy.parameter_vector())
                        stats.record_acc_stats(best_ev_acc)
                        stats.record_std_stats(it.noise_stdev())
                        stats.update_mem_stats()

                        stats.log_stats(tlogger)
                        it.log_stats(tlogger)
                        tlogger.dump_tabular()

                        if it.patience_reached():
                            # to use new trainloader!
                            break

                        if config.snapshot_freq != 0 and it.iteration() % config.snapshot_freq == 0:
                            save_snapshot(stats, it, experiment, policy)
                            if plot:
                                stats.plot_stats(experiment.snapshot_dir())

                    except IterationFailedException:
                        pass

        except KeyboardInterrupt:
            save_snapshot(stats, it, experiment, policy)
            if plot:
                stats.plot_stats(experiment.snapshot_dir())

    def _selection(self, curr_task_results, truncation, num_elites):
        scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                         for result in curr_task_results]
        # elite_scored_models = [t for t in scored_models if t[0] == 0]
        # other_scored_models = [t for t in scored_models if t[0] != 0]
        #
        # best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))
        #
        # scored_models = [best_elite_score] + other_scored_models

        scored_models.sort(key=lambda x: x[2], reverse=True)
        scores = np.array([fitness for (_, _, fitness) in scored_models])

        # pick parents for next generation                            todo - num_elites or not
        parents = [model for (_, model, _) in scored_models[:truncation]]
        # rest = [model for (_, model, _) in scored_models[truncation - 1:]]

        logger.info('Best 5: {}'.format([(i, round(f, 2)) for (i, _, f) in scored_models[:5]]))
        return parents, scores  # , rest
