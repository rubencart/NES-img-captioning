import copy
import logging
import os

from algorithm.tools.podium import Podium
from algorithm.tools.utils import copy_file_from_to, remove_all_files_from_dir

logger = logging.getLogger(__name__)


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
        self._parents_dir = os.path.join(self._log_dir, exp['parents_dir'])
        self._elite_dir = os.path.join(self._log_dir, exp['elite_dir'])
        self._offspring_dir = os.path.join(self._log_dir, exp['offspring_dir'])
        # mkdir_p(self._parents_dir)
        self._new_elite_path = os.path.join(self._elite_dir, '0_elite_params.pth')
        self._new_parent_path = os.path.join(self._parents_dir, '0_{i}_parent_params.pth')

        # MODELS
        self._parents = []
        self._elite = None
        self._podium = Podium(config.patience, os.path.join(self._log_dir, exp['best_dir']))

    def init_from_infos(self, infos: dict):

        self._epoch = infos['epoch'] - 1 if 'epoch' in infos else self._epoch
        self._iteration = infos['iter'] - 1 if 'iter' in infos else self._iteration
        self._bad_generations = infos['bad_gens'] if 'bad_gens' in infos else self._bad_generations
        self._noise_stdev = infos['noise_stdev'] if 'noise_stdev' in infos else self._noise_stdev
        self._batch_size = infos['batch_size'] if 'batch_size' in infos else self._batch_size

        copy_file_from_to(infos['elite'], self._new_elite_path)
        self._elite = self._new_elite_path

        for (i, parent_path) in infos['parents']:
            copy_file_from_to(parent_path, self._new_parent_path.format(i=i))
        self._parents = [(i, self._new_parent_path.format(i=i)) for i, _ in enumerate(infos['parents'])]

        self._podium.init_from_infos(infos)

    def init_parents(self, truncation, policy):
        # important that this stays None:
        # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
        #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
        # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
        #       TRUNC random parents == less random!
        self._parents = [(model_id, None) for model_id in range(truncation)]
        # self._elite = policy.generate_model()
        self._elite = policy.generate_model().serialize(path=self._new_elite_path)

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
            # 'models': {
            #     'parents': self._parents,
            #     'elite': self._elite,
            #
            #     'best_elite': self.best_elite(),
            #     'best_parents': self.best_parents(),
            # },
            'parents': self._parents,
            'elite': self._elite,

            'best_elite': self.best_elite(),
            'best_parents': self.best_parents(),
        }

    def log_stats(self, tlogger):
        tlogger.record_tabular('NoiseStd', self._noise_stdev)
        tlogger.record_tabular('BatchSize', self._batch_size)

        if self._eval_returns:
            tlogger.record_tabular('EliteAcc', self._eval_returns[0])

        if self._patience:
            tlogger.record_tabular('BadGen', str(self._bad_generations) + '/' + str(self._patience))

        tlogger.record_tabular('UniqueWorkers', len(self._worker_ids))
        tlogger.record_tabular('UniqueWorkersFrac', len(self._worker_ids) / len(self._task_results))

    def record_elite(self, acc):
        self._podium.record_elite(self.elite(), acc)
        # self._elite = self._copy_and_clean_elite(elite)
        # return self._elite

    def set_elite(self, elite):
        self._elite = self._copy_and_clean_elite(elite)

    def record_parents(self, parents, score):
        if not self._podium.record_parents(parents, score):

            self._bad_generations += 1

            if self._bad_generations > self._patience:
                # todo tlogger like logger
                logger.warning('Max patience reached; setting lower noise stdev & bigger batch_size')

                self._noise_stdev /= self._stdev_decr_divisor
                self._batch_size *= 2
                self._bad_generations = 0

                new_parents = self._new_parents_from_best()
                self._parents = self._copy_and_clean_parents(new_parents)
                return self._parents
        else:
            self._bad_generations = 0

        new_parents = copy.deepcopy(parents)
        self._parents = self._copy_and_clean_parents(new_parents)
        return None

    def _new_parents_from_best(self):
        _, prev_best_parents = self._podium.best_parents()

        # todo necessary to discard existing index?
        new_parents = [(i, prev) for i, (_, prev) in enumerate(prev_best_parents)]

        # for i, (_, prev_best_parent_path) in enumerate(prev_best_parents):
        #     new_parent = self._new_parent_path.format(i=i)
        #     new_parents.append((i, new_parent))

        return copy.deepcopy(new_parents)

    def _copy_and_clean_parents(self, parents):
        """
        :param parents: List<Tuple<int, str: path to offspring dir>>
        :return: List<Tuple<int, str: path to parents in parents dir>>
        """
        # remove all parents previously stored in parent dir
        remove_all_files_from_dir(self.parents_dir())

        # copy new parents from offspring dir to parent dir and rename them
        new_parents = []
        for i, parent in parents:

            _, new_parent_filename = os.path.split(parent)
            new_parent_path = os.path.join(self.parents_dir(), new_parent_filename)
            new_parents.append((i, new_parent_path))

            copy_file_from_to(parent, new_parent_path)

        # clean offspring dir
        remove_all_files_from_dir(self.offspring_dir())

        return copy.deepcopy(new_parents)

    def _copy_and_clean_elite(self, elite):
        # remove previous elite
        remove_all_files_from_dir(self.elite_dir())

        # name & copy new elite
        _, new_elite_filename = os.path.split(elite)
        new_elite_path = os.path.join(self.elite_dir(), new_elite_filename)
        copy_file_from_to(elite, new_elite_path)

        return new_elite_path

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

    def elite_dir(self):
        return self._elite_dir

    def offspring_dir(self):
        return self._offspring_dir
