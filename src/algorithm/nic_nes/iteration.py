import os

import numpy as np

from algorithm.nets import PolicyNet
from algorithm.tools.iteration import Iteration
from algorithm.tools.utils import mkdir_p, copy_file_from_to, remove_all_files_from_dir


class NESIteration(Iteration):
    """
    Subclass for NIC-NES iteration
    """

    def __init__(self, config, exp):
        super().__init__(config, exp)

        self._current_dir = os.path.join(self._models_dir, 'current')
        mkdir_p(self._current_dir)
        self._current_path = os.path.join(self._current_dir, '0_current_params.pth')

        self._model = None

    def init_from_infos(self, infos: dict):
        super().init_from_infos(infos)
        copy_file_from_to(infos['current_model'], self._current_path)
        self._model = self._current_path.format(i=0)

    def init_from_zero(self, exp, policy):
        self._model = policy.generate_model().serialize(path=self._current_path)

    def init_from_single(self, param_file_name, exp, policy):
        self._model = (policy
                       .generate_model(from_param_file=param_file_name)
                       .serialize(path=self._current_path))

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
        return not bool(self._eval_results)

    def set_model(self, model: PolicyNet):
        assert isinstance(model, PolicyNet)
        remove_all_files_from_dir(self._current_dir)
        self._model = model.serialize(path=self._current_path)

    def current_model(self):
        return self._model

    def score(self):
        return self._eval_results.get(0, ('', None))[1] or float(0)

    def fitnesses(self):
        return np.stack([r.fitness for r in self._task_results])

    def flat_fitnesses(self):
        return np.concatenate([r.fitness for r in self._task_results])

    def noise_vecs(self):
        return np.stack([r.evolve_noise.astype(np.float32) for r in self._task_results])
