import os
from abc import ABC

import numpy as np

from algorithm.nic_nes.optimizers import SGD, Adam
from algorithm.tools.experiment import Experiment, logger
from algorithm.tools.utils import Config, mkdir_p


class NESExperiment(Experiment, ABC):
    """
    Subclass for NIC-NES experiment
    """

    def __init__(self, exp, config: Config, master=True):
        super(NESExperiment, self).__init__(exp, config, master)

        if master:
            self.Optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer_options']['type']]
            self.optimizer = self.Optimizer(np.zeros(1), **exp['optimizer_options']['args'])
            mkdir_p(os.path.join(self.log_dir(), 'optimizer'))
            self.optimizer_path = os.path.join(self.log_dir(), 'optimizer', 'optimizer.tar')

    def init_optimizer(self, params):
        self.optimizer.set_theta(params)
        return self.optimizer

    def get_optimizer(self):
        return self.optimizer

    def init_from_infos(self, infos: dict):
        super(NESExperiment, self).init_from_infos(infos)
        if 'optimizer_state' in infos and infos['optimizer_state'] is not None:
            logger.info('loading optimizer state from {}'.format(infos['optimizer_state']))
            self.optimizer.load_from_file(infos['optimizer_state'])

    def to_dict(self):
        self.optimizer.save_to_file(self.optimizer_path)
        return {
            **super(NESExperiment, self).to_dict(),
            'optimizer_state': self.optimizer_path
        }
