import logging

from podium import Podium

logger = logging.getLogger(__name__)


class Iteration(object):

    def __init__(self, config):
        # CAUTION: todo maybe get these from infos as well when available?
        self._noise_stdev = config.noise_stdev
        self._batch_size = config.batch_size

        self._bad_generations = 0
        self._epoch = 0
        self._iteration = 0
        self._nb_models_to_evaluate = 0

        self._task_results = []
        self._eval_returns = []
        self._worker_ids = []
        self._waiting_for_eval_run = False
        self._elite_evaluated = False
        self._waiting_for_elite_eval = False

        # todo to experiment.py?
        self._stdev_decr_divisor = config.stdev_decr_divisor
        self._patience = config.patience

        self._parents = None  # [(i, None) for i in range(truncation)], todo to init_from_infos?
        self._elite = None
        self._podium = Podium(config.patience)

    def init_from_infos(self, infos, config):
        self.__init__(config)

        self._epoch = infos['epoch'] - 1 if 'epoch' in infos else self._epoch
        self._iteration = infos['iter'] - 1 if 'iter' in infos else self._iteration
        self._bad_generations = infos['bad_gens'] if 'bad_gens' in infos else self._bad_generations
        self._noise_stdev = infos['noise_stdev'] if 'noise_stdev' in infos else self._noise_stdev
        self._batch_size = infos['batch_size'] if 'batch_size' in infos else self._batch_size

        # todo deserialize! now they are set as dicts, see also setup.py
        best_elite = infos['best_elite'] if 'best_elite' in infos else None
        self._podium.set_best_elite(best_elite)
        best_parents = infos['best_parents'] if 'noise_std_stats' in infos else None
        self._podium.set_best_parents(best_parents)

        # todo parents, elite from dict, now this is done in setup.py

    def to_dict(self):
        return {
            'iter': self._iteration,
            'epoch': self._epoch,
            'noise_stdev': self._noise_stdev,
            'batch_size': self._batch_size,

            # todo state_dicts not json serializable?
            # 'parents': [parent.__dict__ for (_, parent) in self._parents if parent],
            #
            # 'best_elite': (self._podium.best_elite()[0], self._podium.best_elite()[1].__dict__),
            # 'best_parents': (self._podium.best_parents()[0],
            #                  [parent.__dict__ for (_, parent) in self._podium.best_parents()[1] if parent]),
        }

    # todo here or in experiment or in policy or...?
    # def serialized_parents(self):
    #     return [(i, parent.serialize() if parent else None) for i, parent in self._parents]
    #
    # def serialized_elite(self):
    #     return self._elite.serialize() if self._elite else None

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
