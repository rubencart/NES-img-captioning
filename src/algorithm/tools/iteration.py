import logging
import os

from abc import ABC

from algorithm.policies import Policy
from algorithm.tools.podium import Podium
from algorithm.tools.utils import check_if_filepath_exists, Config, log


class Iteration(ABC):
    """
    Abstract helper class for bookkeeping stuff of an iteration
    """

    def __init__(self, config: Config, exp: dict):
        # ACROSS SOME ITERATIONS
        self._noise_stdev = config.noise_stdev
        self._batch_size = config.batch_size
        self._times_orig_bs = 1
        self._nb_samples_used = 0
        self._bad_generations = 0
        self._patience_reached = False
        self._epoch = 0
        self._iteration = 0

        # self._schedule = 0
        self._schedule_limit = config.schedule_limit
        self._schedule_start = config.schedule_start if config.schedule_start else 0
        self._schedule_reached = False

        # WITHIN ONE ITERATION
        self._nb_models_to_evaluate = 0
        self._task_results = []
        self._eval_results = {}
        self._worker_ids = []
        self._waiting_for_eval_run = False

        # ENTIRE EXPERIMENT
        self._stdev_divisor = config.stdev_divisor
        self._bs_multiplier = config.bs_multiplier
        self._patience = config.patience
        self._nb_offspring = exp['nb_offspring']

        self._log_dir = exp['log_dir']
        self._models_dir = os.path.join(self._log_dir, 'models')

        # the podium keeps track of the E elites (the E best individuals so far)
        self._podium = Podium(config.patience, os.path.join(self._models_dir, 'best'),
                              num_elites=exp['num_elites'])

    def to_dict(self):
        return {
            'iter': self._iteration,
            'epoch': self._epoch,
            'noise_stdev': self._noise_stdev,
            'batch_size': self._batch_size,
            'bad_generations': self._bad_generations,
            'times_orig_bs': self._times_orig_bs,
            'nb_samples_used': self._nb_samples_used,
            'best_elites': self.best_elites(),
        }

    def init_from_infos(self, infos: dict):

        self._epoch = infos['epoch'] - 1 if 'epoch' in infos else self._epoch
        self._iteration = infos['iter'] - 1 if 'iter' in infos else self._iteration
        self._bad_generations = (
            infos['bad_generations'] if 'bad_generations' in infos else self._bad_generations
        )
        self._noise_stdev = infos['noise_stdev'] if 'noise_stdev' in infos else self._noise_stdev

        self._batch_size = infos['batch_size'] if 'batch_size' in infos else self._batch_size
        self._times_orig_bs = infos['times_orig_bs'] if 'times_orig_bs' in infos else self._times_orig_bs
        self._nb_samples_used = infos['nb_samples_used'] if 'nb_samples_used' in infos \
            else self._nb_samples_used

        self._podium.init_from_infos(infos)

    def init_from_zero(self, exp, policy: Policy):
        raise NotImplementedError

    def init_from_single(self, param_file_name, exp, policy: Policy):
        raise NotImplementedError

    def log_stats(self):
        log('NoiseStd', self._noise_stdev)
        log('BatchSize', self._batch_size)
        log('NbSamplesUsed', self._nb_samples_used)

        if self._patience:
            log('BadGen', str(self._bad_generations) + '/' + str(self._patience))
        elif self._schedule_limit:
            if self._iteration <= self._schedule_start:
                part, full = self._iteration, self._schedule_start
            else:
                part = (self._iteration - self._schedule_start) % self._schedule_limit
                full = self._schedule_limit
            log('Schedule', str(part) + '/' + str(full))

        log('UniqueWorkers', len(self._worker_ids))
        log('UniqueWorkersFrac', len(self._worker_ids) / len(self._task_results))

    def patience_reached(self):
        return self._patience_reached

    def schedule_reached(self):
        return self._schedule_reached

    def record_task_result(self, result):
        self._task_results.append(result)
        self.decr_nb_models_to_evaluate()

    def record_eval_result(self, result):
        raise NotImplementedError

    def process_evaluated_elites(self):
        """
        Process the so far in this iteration received results of elite candidate evaluations:
            hand them to the podium, and handle patience stuff
        :return:
        """
        best_sc, best_ind = float('-inf'), None

        elite_candidates = []
        logging.info('Eval results: {}'.format(self._eval_results))
        for (ind, sc) in self._eval_results.values():
            if check_if_filepath_exists(ind):
                elite_candidates.append((ind, sc))
                if sc > best_sc:
                    best_sc, best_ind = sc, ind

        self._podium.record_elites(elite_candidates)

        if self._patience and self._podium.is_bad_generation():
            self._bad_generations += 1

            if self._bad_generations > self._patience:

                logging.warning('Max patience reached; old std {}, bs: {}'.format(self._noise_stdev, self.batch_size()))
                self.next_curriculum_step()
                self._patience_reached = True
                self._bad_generations = 0
                logging.warning('Max patience reached; new std {}, bs: {}'.format(self._noise_stdev, self.batch_size()))

        else:
            self._bad_generations = 0
        return best_sc, best_ind

    def next_curriculum_step(self):
        # Anneal the noise std and batch size according to the factors in the experiment.json
        self._noise_stdev /= self._stdev_divisor
        self._batch_size = int(self._batch_size * self._bs_multiplier)
        self._times_orig_bs *= self._bs_multiplier

    def warn_waiting_for_evaluations(self):
        if not self.models_left_to_evolve() and self.models_left_to_eval():
            if not self.was_already_waiting_for_eval_run():
                logging.warning('Waiting for evaluations')
                self.set_waiting_for_elite_ev(True)

    def incr_bad_gens(self):
        self._bad_generations += 1

    def incr_epoch(self):
        self._epoch += 1

    def incr_iteration(self):
        self.reset_task_results()
        self.reset_eval_results()
        self.reset_worker_ids()

        self.set_nb_models_to_evaluate(self._nb_offspring)
        self.set_waiting_for_elite_ev(False)
        self._patience_reached = False
        self._schedule_reached = False

        self._iteration += 1
        self._nb_samples_used += self._batch_size

        if self.check_schedule_limit():
            logging.warning('Next curriculum step reached; old std {}, bs: {}'
                            .format(self._noise_stdev, self.batch_size()))
            self._schedule_reached = True
            self.next_curriculum_step()
            logging.warning('Next curriculum step reached; new std {}, bs: {}'
                            .format(self._noise_stdev, self.batch_size()))

    def check_schedule_limit(self):
        return self._schedule_limit and \
               self._iteration >= self._schedule_start and \
               (self._iteration - self._schedule_start) % self._schedule_limit == 0

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

    def reset_eval_results(self):
        self._eval_results = {}

    def reset_worker_ids(self):
        self._worker_ids = []

    def set_waiting_for_elite_ev(self, value):
        self._waiting_for_eval_run = value

    def record_worker_id(self, worker_id):
        self._worker_ids.append(worker_id)
        self._worker_ids = list(set(self._worker_ids))

    def models_left_to_evolve(self):
        return self._nb_models_to_evaluate > 0

    def models_left_to_eval(self):
        raise NotImplementedError

    def epoch(self):
        return self._epoch

    def noise_stdev(self):
        return self._noise_stdev

    def batch_size(self):
        return self._batch_size

    def times_orig_bs(self):
        return self._times_orig_bs

    def iteration(self):
        return self._iteration

    def nb_samples_used(self):
        return self._nb_samples_used

    def get_noise_stdev(self):
        return self._noise_stdev

    def was_already_waiting_for_eval_run(self):
        return self._waiting_for_eval_run

    def eval_returns(self):
        return self._eval_results

    def task_results(self):
        return self._task_results

    def best_elites(self):
        return self._podium.best_elites()

    def best_elite(self):
        return self._podium.best_elites()[0][0]


class IterationFactory:
    @staticmethod
    def create(config, exp):
        if exp['algorithm'] == 'nic_es':
            from algorithm.nic_es.iteration import ESIteration
            return ESIteration(config, exp)
        elif exp['algorithm'] == 'nic_nes':
            from algorithm.nic_nes.iteration import NESIteration
            return NESIteration(config, exp)
