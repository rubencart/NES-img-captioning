
import logging
import os
import shutil

from algorithm.tools.utils import mkdir_p, copy_file_from_to, remove_files, remove_file_if_exists

logger = logging.getLogger(__name__)


class Podium(object):
    DEFAULT_BEST_ELITE = (float('-inf'), None)
    DEFAULT_BEST_PARENTS = (float('-inf'), None)

    def __init__(self, patience, directory):
        self._best_elite = self.DEFAULT_BEST_ELITE
        self._best_parents = self.DEFAULT_BEST_PARENTS

        self._patience = patience

        self._best_directory = os.path.join(directory, 'best')
        self._new_best_elite_path = os.path.join(self._best_directory, '0_elite.pth')
        self._new_best_parent_path = os.path.join(self._best_directory, '0_{i}_parent.pth')
        mkdir_p(self._best_directory)

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
            if _prev_elite and self._best_parents[1] \
                    and _prev_elite not in [p for _, p in self._best_parents[1]]:
                remove_file_if_exists(_prev_elite)

            _, new_elite_filename = os.path.split(elite)
            new_elite_path = os.path.join(self._best_directory, new_elite_filename)

            self._best_elite = (acc, new_elite_path)
            shutil.copy(src=elite,
                        dst=new_elite_path)

    def record_parents(self, parents, score):
        best_parent_score, best_parents = self._best_parents

        if not best_parents or (score - 5 > best_parent_score):
            if best_parents:
                to_remove = [p for _, p in best_parents if p != self._best_elite[1]]
                remove_files(from_dir=self._best_directory, rm_list=to_remove)

            new_best_parents = []
            for (i, parent) in parents:

                # todo is this really necessary? we can also just store them under std name
                _, new_parent_filename = os.path.split(parent)
                new_parent_path = os.path.join(self._best_directory, new_parent_filename)
                new_best_parents.append((i, new_parent_path))

                shutil.copy(src=parent,
                            dst=new_parent_path)

            self._best_parents = (score, new_best_parents)

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
