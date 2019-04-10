
import logging
import os

from algorithm.tools.utils import mkdir_p, copy_file_from_to, remove_files, remove_file_if_exists, \
    remove_all_files_from_dir

logger = logging.getLogger(__name__)


class Podium(object):
    DEFAULT_BEST_ELITE = (float('-inf'), None)
    DEFAULT_BEST_PARENTS = (float('-inf'), None)

    def __init__(self, patience, directory):
        self._best_elite = self.DEFAULT_BEST_ELITE
        self._best_parents = self.DEFAULT_BEST_PARENTS

        self._patience = patience

        _best_directory = directory
        self._best_elite_dir = os.path.join(_best_directory, 'best_elite')
        self._best_parents_dir = os.path.join(_best_directory, 'best_parents')
        self._new_best_elite_path = os.path.join(self._best_elite_dir, '0_elite.pth')
        self._new_best_parent_path = os.path.join(self._best_parents_dir, '0_{i}_parent.pth')
        # mkdir_p(_best_directory)
        mkdir_p(self._best_elite_dir)
        mkdir_p(self._best_parents_dir)

    def init_from_infos(self, infos):

        copy_file_from_to(infos['best_elite'][1], self._new_best_elite_path)
        self._best_elite = (infos['best_elite'][0],
                            self._new_best_elite_path)

        for (i, parent_path) in infos['best_parents'][1]:
            copy_file_from_to(parent_path, self._new_best_parent_path.format(i=i))

        self._best_parents = (
            infos['best_parents'][0],
            [(i, self._new_best_parent_path.format(i=i)) for i, _ in infos['best_parents'][1]]
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
        # todo record epoch/iteration as well!

        _prev_acc, _prev_elite = self._best_elite
        if not _prev_elite or acc > _prev_acc:
            if _prev_elite:
                remove_all_files_from_dir(self._best_elite_dir)

            new_elite_path = self._new_best_elite_path

            self._best_elite = (acc, new_elite_path)
            copy_file_from_to(elite, new_elite_path)

    def record_parents(self, parents, score):
        best_parent_score, best_parents = self._best_parents

        if not best_parents or (score > best_parent_score):
            if best_parents:
                remove_all_files_from_dir(self._best_parents_dir)

            new_best_parents = []
            for (i, parent) in parents:

                new_parent_path = self._new_best_parent_path.format(i=i)
                new_best_parents.append((i, new_parent_path))
                copy_file_from_to(parent, new_parent_path)

            self._best_parents = (score, new_best_parents)

            logger.info('GOOD GENERATION: {}'.format(self._best_parents[0]))
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
