
import copy
import logging
import os
import shutil

from algorithm.tools.utils import mkdir_p

logger = logging.getLogger(__name__)


class Podium(object):
    DEFAULT_BEST_ELITE = (float('-inf'), None)
    DEFAULT_BEST_PARENTS = (float('-inf'), None)

    def __init__(self, patience, directory):
        self._best_elite = self.DEFAULT_BEST_ELITE
        self._best_parents = self.DEFAULT_BEST_PARENTS

        self._patience = patience

        self._best_directory = os.path.join(directory, 'best')
        self._best_elite_path = os.path.join(self._best_directory, 'elite.pth')
        self._best_parent_path = os.path.join(self._best_directory, '{i}_parent.pth')
        mkdir_p(self._best_directory)

    def init_from_checkpt(self, models_checkpt, policy):
        # TODO!!!!
        self._best_elite = (models_checkpt.best_elite[0],
                            policy.from_serialized(models_checkpt.best_elite[1]))
        self._best_parents = (
            models_checkpt.best_parents[0],
            [(i, policy.from_serialized(parent)) for i, parent in models_checkpt.best_parents[1]]
        )

    def serialized(self, log_dir):
        # TODO
        assert self._best_elite[1] is not None
        assert self._best_parents[1] is not None

        score, best_parents = self._best_parents
        return {
            'best_elite': (self._best_elite[0], self._best_elite[1].serialize(log_dir=log_dir)),
            'best_parents': (score, [(i, parent.serialize(log_dir=log_dir)) for i, parent in best_parents])
        }

    def record_elite(self, elite, acc):
        _prev_acc, _prev_elite = self._best_elite
        if not _prev_elite or acc > _prev_acc:
            self._best_elite = (acc, copy.deepcopy(elite))
            shutil.copy(src=elite,
                        dst=self._best_elite_path)

    def record_parents(self, parents, score):
        best_parent_score, best_parents = self._best_parents

        if not best_parents or (score > best_parent_score):
            self._best_parents = (score, copy.deepcopy(parents))
            for i, (_, parent) in enumerate(self._best_parents[1]):
                shutil.copy(src=parent,
                            dst=self._best_parent_path.format(i=i))
            logger.info('GOOD GENERATION')
            return True

        elif self._patience:
            logger.info('BAD GENERATION')
            return False
        return True

    def set_best_elite(self, best_elite):
        assert not self._best_elite[1]
        self._best_elite = best_elite if best_elite else self.DEFAULT_BEST_ELITE

    def set_best_parents(self, best_parents):
        assert not self._best_parents[1]
        self._best_parents = best_parents if best_parents else self.DEFAULT_BEST_PARENTS

    def best_elite(self):
        return self._best_elite

    def best_parents(self):
        return self._best_parents
