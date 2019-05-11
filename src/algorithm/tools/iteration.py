import copy
import logging
import os

import numpy as np
from abc import ABC

from algorithm.nets import SerializableModel
from algorithm.tools.podium import Podium
from algorithm.tools.utils import copy_file_from_to, remove_all_files_from_dir, GAResult, remove_all_files_but, \
    check_if_filepath_exists, mkdir_p, remove_file_if_exists

logger = logging.getLogger(__name__)


class Iteration(ABC):

    def __init__(self, config, exp):
        # ACROSS SOME ITERATIONS
        self._noise_stdev = config.noise_stdev
        self._batch_size = config.batch_size
        self._times_orig_bs = 1
        self._nb_samples_used = 0
        self._bad_generations = 0
        self._patience_reached = False
        self._epoch = 0
        self._iteration = 0

        # WITHIN ONE ITERATION
        self._nb_models_to_evaluate = 0
        self._task_results = []
        self._eval_results = {}
        self._worker_ids = []
        self._waiting_for_eval_run = False

        # todo to experiment.py?
        # ENTIRE EXPERIMENT
        self._stdev_divisor = config.stdev_divisor
        self._bs_multiplier = config.bs_multiplier
        self._patience = config.patience
        self._population_size = exp['population_size']

        self._log_dir = exp['log_dir']
        self._models_dir = os.path.join(self._log_dir, 'models')

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

    def init_from_zero(self, exp, policy):
        raise NotImplementedError

    def init_from_single(self, param_file_name, exp, policy):
        raise NotImplementedError

    def log_stats(self, tlogger):
        tlogger.record_tabular('NoiseStd', self._noise_stdev)
        tlogger.record_tabular('BatchSize', self._batch_size)
        tlogger.record_tabular('NbSamplesUsed', self._nb_samples_used)

        if self._patience:
            tlogger.record_tabular('BadGen', str(self._bad_generations) + '/' + str(self._patience))

        tlogger.record_tabular('UniqueWorkers', len(self._worker_ids))
        tlogger.record_tabular('UniqueWorkersFrac', len(self._worker_ids) / len(self._task_results))

    def patience_reached(self):
        # status = self._patience_reached
        # self._patience_reached = False
        return self._patience_reached

    def record_task_result(self, result):
        self._task_results.append(result)
        self.decr_nb_models_to_evaluate()

    def record_eval_result(self, result):
        raise NotImplementedError

    def process_evaluated_elites(self):
        best_sc, best_ind = float('-inf'), None

        elite_candidates = []
        for (ind, sc) in self._eval_results.values():
            if check_if_filepath_exists(ind):
                elite_candidates.append((ind, sc))
                if sc > best_sc:
                    best_sc, best_ind = sc, ind

        self._podium.record_elites(elite_candidates)

        if self._patience and self._podium.is_bad_generation():
            self._bad_generations += 1

            if self._bad_generations > self._patience:

                logger.warning('Max patience reached; old std {}, bs: {}'.format(self._noise_stdev, self.batch_size()))
                self._noise_stdev /= self._stdev_divisor
                self._batch_size *= self._bs_multiplier
                self._bad_generations = 0
                self._times_orig_bs *= self._bs_multiplier
                self._patience_reached = True
                logger.warning('Max patience reached; new std {}, bs: {}'.format(self._noise_stdev, self.batch_size()))

        else:
            self._bad_generations = 0
        return best_sc, best_ind

    def warn_waiting_for_evaluations(self):
        if not self.models_left_to_evolve() and self.models_left_to_eval():
            if not self.was_already_waiting_for_eval_run():
                logger.warning('Waiting for evaluations')
                self.set_waiting_for_elite_ev(True)

    def incr_bad_gens(self):
        self._bad_generations += 1

    def incr_epoch(self):
        self._epoch += 1

    def incr_iteration(self):
        self.reset_task_results()
        self.reset_eval_results()
        self.reset_worker_ids()

        self.set_nb_models_to_evaluate(self._population_size)
        self.set_waiting_for_elite_ev(False)
        self._patience_reached = False

        self._iteration += 1
        self._nb_samples_used += self._batch_size

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

    # def bad_gens(self):
    #     return self._bad_generations

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

    # def fitnesses(self):
    #     return np.concatenate([r.fitness for r in self._task_results])
    #
    # def flat_fitnesses(self):
    #     return np.concatenate([r.fitness for r in self._task_results]).ravel()

    # def score(self):
    #     return self._eval_results.get(self._model)

    # def _copy_and_clean_elites(self, elites_to_ev):
    #     raise NotImplementedError


class ESIteration(Iteration):

    def __init__(self, config, exp):
        super().__init__(config, exp)

        self._current_dir = os.path.join(self._models_dir, 'current')
        mkdir_p(self._current_dir)
        self._current_path = os.path.join(self._current_dir, '{i}_current_params.pth')

        self._model = None

    def init_from_infos(self, infos: dict):
        super().init_from_infos(infos)
        copy_file_from_to(infos['current_model'], self._current_path.format(i=0))
        self._model = self._current_path.format(i=0)

    def init_from_zero(self, exp, policy):
        self._model = policy.generate_model().serialize(path=self._current_path.format(i=0))
        # self._elites_to_evaluate = [(0, self._model)]

    def init_from_single(self, param_file_name, exp, policy):
        self._model = (policy
                       .generate_model(from_param_file=param_file_name)
                       .serialize(path=self._current_path.format(i=0)))
        # self._elites_to_evaluate = [(0, self._model)]

    def to_dict(self):
        return {
            **super().to_dict(),
            'current_model': self._model,
        }

    def record_eval_result(self, result):
        prev = self._eval_results.get(0, ('', None))[1] or float('-inf')
        self._eval_results.update({
            0: (self._model, max(result.eval_score, prev))
        })

    def models_left_to_eval(self):
        # print(bool(self._eval_results))
        return not bool(self._eval_results)

    def set_model(self, model: SerializableModel):
        assert isinstance(model, SerializableModel)
        remove_all_files_from_dir(self._current_dir)
        self._model = model.serialize(path=self._current_path.format(i=self.iteration()))

    def current_model(self):
        return self._model

    def score(self):
        return self._eval_results.get(0, ('', None))[1] or float(0)

    def fitnesses(self):
        # print([r.fitness for r in self._task_results])
        return np.stack([r.fitness for r in self._task_results])

    def flat_fitnesses(self):
        return np.concatenate([r.fitness for r in self._task_results])  # .ravel()

    def noise_vecs(self):
        return np.stack([r.evolve_noise for r in self._task_results])


class GAIteration(Iteration):
    def __init__(self, config, exp):
        super().__init__(config, exp)

        self._offspring_dir = os.path.join(self._models_dir, 'offspring')
        self._elite_dir = os.path.join(self._models_dir, 'elite')
        self._parents_dir = os.path.join(self._models_dir, 'parents')
        self._new_elite_path = os.path.join(self._elite_dir, '0_{i}_elite_params.pth')
        self._new_parent_path = os.path.join(self._parents_dir, '0_{i}_parent_params.pth')

        self._new_parent_off_path = os.path.join(self._offspring_dir, '0_{i}_parent_params.pth')

        self._truncation = exp['truncation'] if 'truncation' in exp else self._population_size
        self._num_elite_cands = exp['num_elite_cands']

        self._parents = []
        self._elites_to_evaluate = []

    def to_dict(self):
        return {
            **super().to_dict(),

            # TODO when saving a snapshot maybe we do want the param files and
            # not just the path to them...
            # 'models': {
            #     'parents': self._parents,
            #     'elite': self._elite,
            #
            #     'best_elite': self.best_elite(),
            #     'best_parents': self.best_parents(),
            # },
            'elites_to_evaluate': self._elites_to_evaluate,
            'parents': self._parents,
            'best_parents': self.best_parents(),
        }

    def init_from_infos(self, infos: dict):

        super().init_from_infos(infos)

        for (i, elite_path) in infos['elites_to_evaluate']:
            copy_file_from_to(elite_path, self._new_elite_path.format(i=i))
        self._elites_to_evaluate = [(i, self._new_elite_path.format(i=i))
                                    for i, _ in enumerate(infos['elites_to_evaluate'])]

        for (i, parent_path) in infos['parents']:
            copy_file_from_to(parent_path, self._new_parent_off_path.format(i=i))
        self._parents = [(i, self._new_parent_off_path.format(i=i)) for i, _ in enumerate(infos['parents'])]

    def init_from_zero(self, exp, policy):
        # important that this stays None:
        # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
        #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
        # - if ComprModels: workers get CM as parent ==> first gen: POP_SIZE descendants of
        #       TRUNC random parents == less random!
        self._parents = [(model_id, None) for model_id in range(self._truncation)]

        # self._elites_to_evaluate = policy.generate_model().serialize(path=self._new_elite_path)
        self._elites_to_evaluate = []
        for i in range(self._num_elite_cands):
            cand = policy.generate_model().serialize(path=self._new_elite_path.format(i=i))
            self._elites_to_evaluate.append((i, cand))

    def init_from_single(self, param_file_name, exp, policy):

        parent_path = policy \
            .generate_model(from_param_file=param_file_name) \
            .serialize(path=self._new_parent_off_path.format(i=0))
        elite_path = policy \
            .generate_model(from_param_file=param_file_name) \
            .serialize(path=self._new_elite_path.format(i=0))

        self._parents = [(0, parent_path)]
        self._elites_to_evaluate = [(0, elite_path)]

    def record_parents(self, parents, score):
        # self._podium.record_parents(parents, score)

        new_parents = [(i, p) for i, p in enumerate(parents)]
        self._parents = self._copy_and_clean_parents(new_parents)
        # self._parents = copy.copy(new_parents)
        self._add_elites_to_parents()
        self._clean_offspring_dir()
        return None

    # def _new_parents_from_best(self):
    #     _, prev_best_parents = self._podium.best_parents()
    #     new_parents = [(i, prev) for i, prev in enumerate(prev_best_parents)]
    #     return copy.deepcopy(new_parents)

    def _add_elites_to_parents(self):
        elites = [e for (e, sc) in self.best_elites()]
        parents = [p for (i, p) in self._parents]
        self._parents = [(i, m) for i, m in enumerate(elites + parents)]

    def _copy_and_clean_parents(self, parents):
        """
        :param parents: List<Tuple<int, str: path to offspring dir>>
        :return: List<Tuple<int, str: path to parents in parents dir>>
        """
        # remove all parents previously stored in parent dir
        # remove_all_files_from_dir(self.parents_dir())

        # copy new parents from offspring dir to parent dir and rename them
        new_parents = []
        for i, parent in parents:

            # _, new_parent_filename = os.path.split(parent)
            # new_parent_path = os.path.join(self.parents_dir(), new_parent_filename)
            new_parent_path = self._new_parent_off_path.format(i=i)
            new_parents.append((i, new_parent_path))

            # copy_file_from_to(parent, new_parent_path)
            remove_file_if_exists(new_parent_path)
            os.rename(parent, new_parent_path)

        return copy.deepcopy(new_parents)

    def set_next_elites_to_evaluate(self, best_individuals):
        elites_to_evaluate = [(i, ind) for i, ind in enumerate(best_individuals)]
        self._elites_to_evaluate = self._copy_and_clean_elites(elites_to_evaluate)

    def _copy_and_clean_elites(self, elites):
        # copy new elite cands from offspring dir to elite dir and rename them
        new_elites_to_ev = []
        new_elite_filenames = []
        for i, elite in elites:

            # _, new_elite_filename = os.path.split(elite)
            # new_elite_path = os.path.join(self.elite_dir(), new_elite_filename)
            new_elite_path = self._new_elite_path.format(i=i)
            new_elites_to_ev.append((i, new_elite_path))
            new_elite_filenames.append(new_elite_path)
            copy_file_from_to(elite, new_elite_path)

        # remove previous elite
        remove_all_files_but(self.elite_dir(), new_elite_filenames)

        return copy.deepcopy(new_elites_to_ev)

    def record_eval_result(self, eval_return: GAResult):
        prev = self._eval_results.get(eval_return.evaluated_cand_id, ('', None))[1] or float('-inf')
        self._eval_results.update({
            eval_return.evaluated_cand_id:
                (eval_return.evaluated_cand, max(eval_return.score, prev))
        })

    def models_left_to_eval(self):
        evaluated = set(self._eval_results.keys())
        # return not all([idx in evaluated for idx, _ in self._elites_to_evaluate])
        return len(evaluated) < len(self._elites_to_evaluate) / 2.0

    def _clean_offspring_dir(self):
        # clean offspring dir
        # remove_all_files_from_dir(self.offspring_dir())
        remove_all_files_but(self._offspring_dir,
                             [parent for _, parent in self._parents])

    def parents_dir(self):
        return self._parents_dir

    def offspring_dir(self):
        return self._offspring_dir

    def parents(self):
        return self._parents

    def elites_to_evaluate(self):
        return self._elites_to_evaluate

    def best_parents(self):
        return self._podium.best_parents()

    def elite_dir(self):
        return self._elite_dir


class IterationFactory:
    @staticmethod
    def create(config, exp):
        if exp['algorithm'] == 'ga':
            return GAIteration(config, exp)

        elif exp['algorithm'] == 'es':
            return ESIteration(config, exp)
