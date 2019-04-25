
import logging
import os

from algorithm.tools.utils import mkdir_p, copy_file_from_to, remove_all_files_from_dir, remove_all_files_but

logger = logging.getLogger(__name__)


class Podium(object):
    DEFAULT_BEST_ELITE = [('', float('-inf')), ]
    DEFAULT_BEST_PARENTS = (float('-inf'), [])

    def __init__(self, patience, directory, num_elites):
        self._best_elites = [('', float('-inf')) for _ in range(num_elites)]
        self._best_parents = self.DEFAULT_BEST_PARENTS

        self._num_elites = num_elites
        self._patience = patience

        _best_directory = directory
        self._best_elite_dir = os.path.join(_best_directory, 'best_elite')
        self._best_parents_dir = os.path.join(_best_directory, 'best_parents')
        self._new_best_elite_path = os.path.join(self._best_elite_dir, '0_{i}_elite.pth')
        self._new_best_parent_path = os.path.join(self._best_parents_dir, '0_{i}_parent.pth')
        # mkdir_p(_best_directory)
        mkdir_p(self._best_elite_dir)
        mkdir_p(self._best_parents_dir)

        self._bad_generation = True

    def init_from_infos(self, infos):
        # TODO !!!!! make parents optional

        self._best_elites = []
        for i, (elite_path, sc) in enumerate(infos['best_elites']):
            new_elite_path = self._new_best_elite_path.format(i=i)
            copy_file_from_to(elite_path, new_elite_path)
            self._best_elites.append((new_elite_path, sc))

        if 'best_parents' in infos and infos['best_parents'] is not None:
            for (i, parent_path) in enumerate(infos['best_parents'][1]):
                copy_file_from_to(parent_path, self._new_best_parent_path.format(i=i))

            self._best_parents = (
                infos['best_parents'][0],
                [self._new_best_parent_path.format(i=i) for i in range(len(infos['best_parents'][1]))]
            )

    # def serialized(self, log_dir):
    #     assert self._best_elites[0][0]
    #     assert self._best_parents[1]
    #
    #     pscore, best_parents = self._best_parents
    #     # escore, best_elites = self._best_elites
    #     return {
    #         'best_elites': (escore, [elite.serialize(log_dir=log_dir) for elite in best_elites]),
    #         'best_parents': (pscore, [parent.serialize(log_dir=log_dir) for parent in best_parents])
    #     }

    def record_elites(self, elites_and_scores):
        # todo record epoch/iteration as well!

        all_cands = self._best_elites + list(elites_and_scores)
        sorted_cands = sorted(all_cands, key=lambda cand: cand[1], reverse=True)
        best_cands = sorted_cands[:self._num_elites]

        new_best_elites = []
        new_best_el_filenames = []
        for i, (elite, sc) in enumerate(best_cands):
            if elite:
                new_elite_path = self._new_best_elite_path.format(i=i)
                new_best_elites.append((new_elite_path, sc))
                new_best_el_filenames.append(new_elite_path)

                if elite != new_elite_path:
                    try:
                        copy_file_from_to(elite, new_elite_path)
                        # this means a new elite is added to the podium
                        self._bad_generation = False
                    except OSError:
                        logging.error('[Podium]: tried to copy non existing elite')

        self._best_elites = new_best_elites
        remove_all_files_but(self._best_elite_dir, new_best_el_filenames)

    def is_bad_generation(self):
        status = self._bad_generation
        if status:
            logger.info('BAD GENERATION')
        else:
            logger.info('GOOD GENERATION')
        self._bad_generation = True
        return status

    def record_parents(self, parents, score):
        best_parent_score, best_parents = self._best_parents

        if not best_parents or (score > best_parent_score):
            if best_parents:
                remove_all_files_from_dir(self._best_parents_dir)

            new_best_parents = []
            for (i, parent) in enumerate(parents):

                new_parent_path = self._new_best_parent_path.format(i=i)
                new_best_parents.append(new_parent_path)
                copy_file_from_to(parent, new_parent_path)

            self._best_parents = (score, new_best_parents)

            # logger.info('GOOD GENERATION: {}'.format(self._best_parents[0]))
            # return True

        # elif self._patience:
        #     logger.info('BAD GENERATION')
        #     return False
        # return True

    # def set_best_elites(self, best_elites):
    #     assert not self._best_elites[0][0]
    #     self._best_elites = best_elites if best_elites else self.DEFAULT_BEST_ELITE

    # def set_best_parents(self, best_parents):
    #     assert not self._best_parents[1]
    #     self._best_parents = best_parents if best_parents else self.DEFAULT_BEST_PARENTS

    def best_elites(self):
        return self._best_elites

    def best_parents(self):
        return self._best_parents
