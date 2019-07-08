import os
from abc import ABC

from algorithm.tools.experiment import Experiment
from algorithm.tools.utils import Config, mkdir_p


class ESExperiment(Experiment, ABC):
    """
    Subclass for NIC-ES experiment
    """

    def __init__(self, exp, config: Config, master=True):
        super(ESExperiment, self).__init__(exp, config, master)

        self._population_size = exp['population_size'] if 'population_size' in exp else exp['nb_offspring']
        self._num_elites = exp['num_elites']
        self._num_elite_cands = exp['num_elite_cands']
        self._tournament_size = exp['tournament_size'] if 'tournament_size' in exp else None
        self._selection = exp['selection'] if 'selection' in exp else 'population_size'

        if master:

            _models_dir = os.path.join(self._log_dir, 'models')
            self._parents_dir = os.path.join(_models_dir, 'parents')
            self._offspring_dir = os.path.join(_models_dir, 'offspring')
            self._elite_dir = os.path.join(_models_dir, 'elite')

            mkdir_p(self._parents_dir)
            mkdir_p(self._offspring_dir)
            mkdir_p(self._elite_dir)

    def population_size(self):
        return self._population_size

    def selection(self):
        return self._selection

    def tournament_size(self):
        return self._tournament_size

    def parents_dir(self):
        assert self._master
        return self._parents_dir

    def offspring_dir(self):
        assert self._master
        return self._offspring_dir

    def elite_dir(self):
        assert self._master
        return self._elite_dir

    def num_elites(self):
        return self._num_elites

    def num_elite_cands(self):
        return self._num_elite_cands
