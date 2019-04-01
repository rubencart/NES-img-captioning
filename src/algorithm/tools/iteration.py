import logging
import os

import torch
from collections import namedtuple

from algorithm.nets import SerializableModel
from algorithm.tools.podium import Podium
from algorithm.policies import Policy
from algorithm.tools.utils import mkdir_p

logger = logging.getLogger(__name__)

Checkpoint = namedtuple('Checkpoint', ['elite', 'best_elite', 'parents', 'best_parents'])


class Iteration(object):

    def __init__(self, config, exp):
        # CAUTION: todo maybe get these from infos as well when available?

        # ACROSS SOME ITERATIONS
        self._noise_stdev = config.noise_stdev
        self._batch_size = config.batch_size
        self._bad_generations = 0
        self._epoch = 0
        self._iteration = 0

        # WITHIN ONE ITERATION
        self._nb_models_to_evaluate = 0
        self._task_results = []
        self._eval_returns = []
        self._worker_ids = []
        self._waiting_for_eval_run = False
        self._elite_evaluated = False
        self._waiting_for_elite_eval = False

        # todo to experiment.py?
        # ENTIRE EXPERIMENT
        self._stdev_decr_divisor = config.stdev_decr_divisor
        self._patience = config.patience

        self._log_dir = exp['log_dir']
        self._parents_dir = os.path.join(self._log_dir, 'tmp')
        mkdir_p(self._parents_dir)
        self._elite_path = os.path.join(self._parents_dir, 'elite_params.pth')
        # self._parent_path = os.path.join(_parents_dir, 'i{i}_parent_params.pth')

        # MODELS
        # [(int, ABCModel),]
        self._parents = []
        self._elite = None
        self._podium = Podium(config.patience, self._parents_dir)

    def init_from_infos(self, infos: dict, models_checkpt: Checkpoint, policy: Policy):
        # self.__init__(config)

        self._epoch = infos['epoch'] - 1 if 'epoch' in infos else self._epoch
        self._iteration = infos['iter'] - 1 if 'iter' in infos else self._iteration
        self._bad_generations = infos['bad_gens'] if 'bad_gens' in infos else self._bad_generations
        self._noise_stdev = infos['noise_stdev'] if 'noise_stdev' in infos else self._noise_stdev
        self._batch_size = infos['batch_size'] if 'batch_size' in infos else self._batch_size

        # todo! check if still works
        self._elite = policy.from_serialized(models_checkpt.elite)
        self._parents = [(i, policy.from_serialized(parent)) for i, parent in models_checkpt.parents]

        self._podium.init_from_checkpt(models_checkpt, policy)

    def init_parents(self, truncation, policy):
        # important that this stays None:
        # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
        #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
        # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
        #       TRUNC random parents == less random!
        self._parents = [(model_id, None) for model_id in range(truncation)]
        # self._elite = policy.generate_model()
        self._elite = policy.generate_model().serialize(path=self._elite_path)

    def init_from_single(self, param_file_name, truncation, policy):
        # todo! check if still works
        self._parents = [(i, policy.generate_model(from_param_file=param_file_name))
                         for i in range(truncation)]
        self._elite = policy.generate_model(from_param_file=param_file_name)

    def to_dict(self):
        return {
            'iter': self._iteration,
            'epoch': self._epoch,
            'noise_stdev': self._noise_stdev,
            'batch_size': self._batch_size,
            'bad_generations': self._bad_generations,

            # TODO when saving a snapshot maybe we do want the param files and
            # not just the path to them...
            'parents': self._parents,
            'elite': self._elite,

            'best_elite': self.best_elite(),
            'best_parents': self.best_parents(),
        }

    # def serialized_parents(self):
    #     assert self._elite is not None
    #     assert len(self._parents) > 0
    #
    #     to_save = {
    #         'elite': self._elite.serialize(path=self._elite_path),
    #         'parents': [(i, parent.serialize(path=self._parent_path.format(i=i))) for i, parent in self._parents]
    #     }
    #     return to_save

    def serialized_best_parents(self):
        # TODO!
        return self._podium.serialized(self._log_dir)

    # def parents_for_redis(self):
    #     result = []
    #     for i, parent in self._parents:
    #         if parent:
    #             result.append((i, parent.serialize(path=self._parent_path.format(i=i))))
    #         else:
    #             result.append((i, None))
    #     return result
    #
    # def elite_for_redis(self):
    #     # elite is never None right?
    #     return self._elite.serialize(path=self._elite_path)

    # def save_parents_to_disk(self):
    #     paths = []
    #     for (i, parent) in self._parents:
    #         if parent:
    #             parent_filename = 'parents_params_i{i}.pth'
    #             path_to_parent = os.path.join(self._log_dir, self._parents_dir, parent_filename)
    #             torch.save(parent.serialize(), path_to_parent)
    #             paths.append((i, path_to_parent))
    #         else:
    #             paths.append((i, None))
    #
    #     return paths

    # def save_elite_to_disk(self):
    #     elite_filename = 'elite_params_i{i}.pth'
    #     path_to_elite = os.path.join(self._log_dir, self._parents_dir, elite_filename)
    #
    #     torch.save(self._elite.serialize(), path_to_elite)
    #     return path_to_elite

    def log_stats(self, tlogger):
        tlogger.record_tabular('NoiseStd', self._noise_stdev)
        tlogger.record_tabular('BatchSize', self._batch_size)

        if self._eval_returns:
            tlogger.record_tabular('EliteAcc', self._eval_returns[0])

        if self._patience:
            tlogger.record_tabular('BadGen', str(self._bad_generations) + '/' + str(self._patience))

        tlogger.record_tabular('UniqueWorkers', len(self._worker_ids))
        tlogger.record_tabular('UniqueWorkersFrac', len(self._worker_ids) / len(self._task_results))

    def record_elite(self, elite, acc):
        self._podium.record_elite(elite, acc)

    def record_parents(self, parents, score):
        if not self._podium.record_parents(parents, score):

            self._bad_generations += 1

            if self._bad_generations > self._patience:
                # todo tlogger like logger
                logger.warning('Max patience reached; setting lower noise stdev & bigger batch_size')

                self._noise_stdev /= self._stdev_decr_divisor
                self._batch_size *= 2

                self._bad_generations = 0
                return self._podium.best_parents()  # best_parents_so_far[1]
        else:
            self._bad_generations = 0
            return None

    def set_elite(self, elite):
        self._elite = elite

    def set_parents(self, parents):
        self._parents = parents

    def warn_elite_evaluated(self):
        if not self.models_left_to_evaluate() and not self.elite_evaluated():
            if not self.was_already_waiting_for_elite_eval():
                logger.warning('Waiting for elite to be evaluated')
                self.set_waiting_for_elite_eval(True)

    def warn_eval_run(self):
        if not self.models_left_to_evaluate() and not self.eval_ran():
            if not self.was_already_waiting_for_eval_run():
                logger.warning('Waiting for eval runs')
                self.set_waiting_for_eval_run(True)

    def reset_bad_gens(self):
        self._bad_generations = 0

    def incr_bad_gens(self):
        self._bad_generations += 1

    def incr_epoch(self):
        self._epoch += 1

    def incr_iteration(self):
        self._iteration += 1

    def set_batch_size(self, value):
        self._batch_size = value

    def set_noise_stdev(self, value):
        self._noise_stdev = value

    def set_nb_models_to_evaluate(self, nb_models):
        self._nb_models_to_evaluate = nb_models

    def decr_nb_models_to_evaluate(self):
        self._nb_models_to_evaluate -= 1

    def reset_task_results(self):
        self._task_results = []

    def reset_eval_returns(self):
        self._eval_returns = []

    def reset_worker_ids(self):
        self._worker_ids = []

    def set_waiting_for_eval_run(self, value):
        self._waiting_for_eval_run = value

    def set_elite_evaluated(self, value):
        self._elite_evaluated = value

    def set_waiting_for_elite_eval(self, value):
        self._waiting_for_elite_eval = value

    def record_worker_id(self, worker_id):
        self._worker_ids.append(worker_id)
        self._worker_ids = list(set(self._worker_ids))

    def record_eval_return(self, eval_return):
        self._eval_returns.append(eval_return)

    def record_task_result(self, result):
        self._task_results.append(result)

    def models_left_to_evaluate(self):
        return self._nb_models_to_evaluate > 0

    def epoch(self):
        return self._epoch

    def noise_stdev(self):
        return self._noise_stdev

    def batch_size(self):
        return self._batch_size

    def bad_gens(self):
        return self._bad_generations

    def iteration(self):
        return self._iteration

    def get_noise_stdev(self):
        return self._noise_stdev

    def elite_evaluated(self):
        return self._elite_evaluated

    def eval_ran(self):
        return len(self._eval_returns) > 0

    def was_already_waiting_for_eval_run(self):
        return self._waiting_for_eval_run

    def eval_returns(self):
        return self._eval_returns

    def max_eval_return(self):
        return max(self._eval_returns) if self._eval_returns else float('-inf')

    def was_already_waiting_for_elite_eval(self):
        return self._waiting_for_elite_eval

    def task_results(self):
        return self._task_results

    def best_elite(self):
        return self._podium.best_elite()

    def best_parents(self):
        return self._podium.best_parents()

    def elite(self):
        return self._elite

    def parents(self):
        return self._parents

    def parents_dir(self):
        return self._parents_dir
