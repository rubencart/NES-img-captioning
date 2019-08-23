import copy
import os

from algorithm.policies import Policy
from algorithm.tools.iteration import Iteration
from algorithm.tools.utils import Config, copy_file_from_to, remove_file_if_exists, remove_all_files_but


class ESIteration(Iteration):
    """
    Subclass for NIC-ES iteration
        Parents are kept together with offspring in offspring dir: models/offspring
        elite are kept in models/elite
    """

    def __init__(self, config: Config, exp: dict):
        super().__init__(config, exp)

        self._offspring_dir = os.path.join(self._models_dir, 'offspring')
        self._elite_dir = os.path.join(self._models_dir, 'elite')
        self._new_elite_path = os.path.join(self._elite_dir, '0_{i}_elite_params.pth')
        self._new_parent_path = os.path.join(self._offspring_dir, '0_{i}_parent_params.pth')

        self._pop_size = exp['population_size'] if 'population_size' in exp else self._nb_offspring
        self._num_elite_cands = exp['num_elite_cands']

        self._parents = []
        self._elites_to_evaluate = []

    def to_dict(self):
        return {
            **super().to_dict(),
            'elites_to_evaluate': self._elites_to_evaluate,
            'parents': self._parents,
        }

    def init_from_infos(self, infos: dict):

        super().init_from_infos(infos)

        for (i, elite_path) in infos['elites_to_evaluate']:
            copy_file_from_to(elite_path, self._new_elite_path.format(i=i))
        self._elites_to_evaluate = [(i, self._new_elite_path.format(i=i))
                                    for i, _ in enumerate(infos['elites_to_evaluate'])]

        for (i, parent_path) in infos['parents']:
            copy_file_from_to(parent_path, self._new_parent_path.format(i=i))
        self._parents = [(i, self._new_parent_path.format(i=i)) for i, _ in enumerate(infos['parents'])]

    def init_from_zero(self, exp: dict, policy: Policy):
        # important that this stays None:
        # - if None: workers get None as parent to evaluate so initialize POP_SIZE random initial parents,
        #       of which the best TRUNC get selected ==> first gen: POP_SIZE random parents
        # - if Models: workers get CM as parent ==> first gen: POP_SIZE descendants of
        #       TRUNC random parents == less random!
        self._parents = [(model_id, None) for model_id in range(self._pop_size)]

        self._elites_to_evaluate = []
        for i in range(self._num_elite_cands):
            cand = policy.generate_model().serialize(path=self._new_elite_path.format(i=i))
            self._elites_to_evaluate.append((i, cand))

    def init_from_singles(self, param_file_names: list, exp: dict, policy: Policy):
        if isinstance(param_file_names, str):
            param_file_names = [param_file_names]

        self._parents = []
        self._elites_to_evaluate = []

        for i, param_file_name in enumerate(param_file_names):
            parent_path = policy \
                .generate_model(from_param_file=param_file_name) \
                .serialize(path=self._new_parent_path.format(i=i))
            elite_path = policy \
                .generate_model(from_param_file=param_file_name) \
                .serialize(path=self._new_elite_path.format(i=i))

            self._parents.append((i, parent_path))
            self._elites_to_evaluate.append((i, elite_path))

        self._elites_to_evaluate = self._elites_to_evaluate[:self._num_elite_cands]

    def record_parents(self, parents: list):
        new_parents = [(i, p) for i, p in enumerate(parents)]
        self._parents = self._copy_and_clean_parents(new_parents)
        self._add_elites_to_parents()
        self._clean_offspring_dir()
        return None

    def _add_elites_to_parents(self):
        elites = [e for (e, sc) in self.best_elites()]
        parents = [p for (i, p) in self._parents]
        self._parents = [(i, m) for i, m in enumerate(elites + parents)]

    def _copy_and_clean_parents(self, parents: list):
        """
        :param parents: List<Tuple<int, str: path to offspring dir>>
        :return: List<Tuple<int, str: path to parents in parents dir>>
        """
        # Keep parents in offspring dir but rename them
        new_parents = []
        for i, parent in parents:

            new_parent_path = self._new_parent_path.format(i=i)
            new_parents.append((i, new_parent_path))

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

            new_elite_path = self._new_elite_path.format(i=i)
            new_elites_to_ev.append((i, new_elite_path))
            new_elite_filenames.append(new_elite_path)
            copy_file_from_to(elite, new_elite_path)

        # remove previous elite
        remove_all_files_but(self.elite_dir(), new_elite_filenames)

        return copy.deepcopy(new_elites_to_ev)

    def record_eval_result(self, eval_return):
        prev = self._eval_results.get(eval_return.evaluated_cand_id, ('', None))[1] or float('-inf')
        self._eval_results.update({
            eval_return.evaluated_cand_id:
                (eval_return.evaluated_cand, max(eval_return.score, prev))
        })

    def models_left_to_eval(self):
        evaluated = set(self._eval_results.keys())
        return len(evaluated) < len(self._elites_to_evaluate)

    def _clean_offspring_dir(self):
        remove_all_files_but(self._offspring_dir,
                             [parent for _, parent in self._parents])

    def offspring_dir(self):
        return self._offspring_dir

    def parents(self):
        return self._parents

    def elites_to_evaluate(self):
        return self._elites_to_evaluate

    def elite_dir(self):
        return self._elite_dir
