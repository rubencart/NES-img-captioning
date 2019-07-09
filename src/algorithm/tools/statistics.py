import logging
import time

import numpy as np
import psutil

import matplotlib.pyplot as plt

from algorithm.tools.utils import log


class Statistics(object):
    """
    Wrapper class for a bunch of running statistics
    """

    def __init__(self):
        self._score_stats = [[], [], []]
        self._score_stds = []
        self._time_stats = []
        self._acc_stats = []
        self._norm_stats = []
        self._std_stats = []
        self._bs_stats = []
        self._mem_stats = [[], [], []]
        self._best_acc_so_far_stats = []

        self._step_tstart = 0
        self._tstart = time.time()
        self._time_elapsed = 0

        self._it_worker_mem_usages = {}
        self._it_master_mem_usages = []

        self._update_ratio_stats = []

    def init_from_infos(self, infos):

        self._score_stats = infos['score_stats'] if 'score_stats' in infos else self._score_stats
        self._score_stds = infos['score_stds'] if 'score_stds' in infos else self._score_stds
        self._time_stats = infos['time_stats'] if 'time_stats' in infos else self._time_stats
        self._acc_stats = infos['acc_stats'] if 'acc_stats' in infos else self._acc_stats
        self._norm_stats = infos['norm_stats'] if 'norm_stats' in infos else self._norm_stats
        self._std_stats = infos['noise_std_stats'] if 'noise_std_stats' in infos else self._std_stats
        self._bs_stats = infos['bs_stats'] if 'bs_stats' in infos else self._bs_stats
        self._mem_stats = infos['mem_stats'] if 'mem_stats' in infos else self._mem_stats
        self._update_ratio_stats = infos['update_ratio_stats'] if 'update_ratio_stats' in infos \
            else self._update_ratio_stats
        self._time_elapsed = infos['time_elapsed'] if 'time_elapsed' in infos else self._time_elapsed
        self._best_acc_so_far_stats = infos['best_acc_so_far_stats'] \
            if 'best_acc_so_far_stats' in infos else self._best_acc_so_far_stats

    def to_dict(self):
        return {
            'score_stats': self._score_stats,
            'score_stds': self._score_stds,
            'time_stats': self._time_stats,
            'acc_stats': self._acc_stats,
            'norm_stats': self._norm_stats,
            'noise_std_stats': self._std_stats,
            'bs_stats': self._bs_stats,
            'mem_stats': self._mem_stats,
            'update_ratio_stats': self._update_ratio_stats,
            'time_elapsed': self._time_elapsed,
            'best_acc_so_far_stats': self._best_acc_so_far_stats,
        }

    def plot_stats(self, log_dir):
        kwargs = {
            'time': (self._time_stats, 'Time per gen'),
            'norm': (self._norm_stats, 'Norm of params'),
            'acc': (self._acc_stats, 'Elite score'),
            'best_acc': (self._best_acc_so_far_stats, 'Best elite score'),
            'master_mem': (self._mem_stats[0], 'Master mem usage'),
            'worker_mem': (self._mem_stats[2], 'Worker mem usage'),
            'virtmem': (self._mem_stats[1], 'Virt mem usage'),
            'batch_size': (self._bs_stats, 'Batch size'),
            'noise_std': (self._std_stats, 'Noise stdev'),
            'reward_std': (self._score_stds, 'Score stdev'),
        }
        if self._update_ratio_stats:
            kwargs.update({'update_ratio': (self._update_ratio_stats, 'Update ratio')})
        self._plot(log_dir, self._score_stats, **kwargs)

    @staticmethod
    def _plot(log_dir, score_stats=None, **kwargs):
        if score_stats:
            fig = plt.figure()
            x = np.arange(len(score_stats[1]))
            plt.fill_between(x=x, y1=score_stats[0], y2=score_stats[2], facecolor='blue', alpha=0.3)
            plt.plot(x, score_stats[1], color='blue')
            plt.title('Training score')
            plt.savefig(log_dir + '/loss_plot.pdf', format='pdf')
            plt.close(fig)

        for (name, (lst, label)) in kwargs.items():
            fig = plt.figure()
            plt.plot(np.arange(len(lst)), lst)
            plt.title(label)
            plt.savefig(log_dir + '/{}_plot.pdf'.format(name), format='pdf')
            plt.close(fig)

    def log_stats(self):
        logging.info('---------------- STATS ----------------')
        log('RewMax', self._score_stats[2][-1])
        log('RewMean', self._score_stats[1][-1])
        log('RewMin', self._score_stats[0][-1])
        log('RewStd', self._score_stds[-1])
        log('EliteAcc', self._acc_stats[-1])
        log('BestEliteAcc', self._best_acc_so_far_stats[-1])
        log('NormMean', self._norm_stats[-1])

        if self._update_ratio_stats:
            log('UpdateRatio', self._update_ratio_stats[-1])

        step_tend = time.time()
        log('TimeElapsedThisIter', step_tend - self._step_tstart)
        log('TimeElapsed', self._time_elapsed)
        log('MemUsage', self._mem_stats[1][-1])

    def record_score_stats(self, scores: np.ndarray):
        """
        :param scores: np.ndarray
        """
        self._score_stats[0].append(scores.min())
        self._score_stats[1].append(scores.mean())
        self._score_stats[2].append(scores.max())
        self._score_stds.append(scores.std())

    def record_time_stats(self, value):
        self._time_stats.append(value)

    def record_acc_stats(self, value):
        self._acc_stats.append(value)

    def record_best_acc_stats(self, value):
        self._best_acc_so_far_stats.append(value)

    def record_norm_stats(self, param_vector):
        norm = float(param_vector.abs().sum() / param_vector.numel())
        self._norm_stats.append(norm)

    def record_std_stats(self, value):
        self._std_stats.append(value)

    def record_bs_stats(self, value):
        self._bs_stats.append(value)

    def reset_it_mem_usages(self):
        self._it_master_mem_usages, self._it_worker_mem_usages = [], {}

    def record_it_worker_mem_usage(self, worker_id, worker_mem_usage):
        value = max(worker_mem_usage, self._it_worker_mem_usages[worker_id]) \
            if worker_id in self._it_worker_mem_usages else worker_mem_usage

        self._it_worker_mem_usages.update({worker_id: value})

    def record_it_master_mem_usage(self, master_mem_usage):
        self._it_master_mem_usages.append(master_mem_usage)

    def update_mem_stats(self):
        self._mem_stats[0].append(max(self._it_master_mem_usages) if self._it_master_mem_usages else 0)
        self._mem_stats[1].append(psutil.virtual_memory().percent)
        self._mem_stats[2].append(sum(self._it_worker_mem_usages.values()) / len(self._it_worker_mem_usages.values()))

    def record_step_time_stats(self):
        step_tend = time.time()
        self._time_elapsed += step_tend - self._step_tstart
        self._time_stats.append(step_tend - self._step_tstart)

    def record_update_ratio(self, update_ratio):
        self._update_ratio_stats.append(update_ratio)

    def set_step_tstart(self):
        self._step_tstart = time.time()

    def score_stats(self):
        return self._score_stats

    def time_stats(self):
        return self._time_stats

    def acc_stats(self):
        return self._acc_stats

    def norm_stats(self):
        return self._norm_stats

    def std_stats(self):
        return self._std_stats

    def bs_stats(self):
        return self._bs_stats

    def mem_stats(self):
        return self._mem_stats

    def step_tstart(self):
        return self._step_tstart
