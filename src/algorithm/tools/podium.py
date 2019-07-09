
import logging
import os

from algorithm.tools.utils import mkdir_p, copy_file_from_to, remove_all_files_but

logger = logging.getLogger(__name__)


class Podium(object):
    """
    Keeps track of best individuals so far
    And determines based on the elite results if a generation was 'good' or 'bad' --> for the patience
        and annealing schedule. Good means a new elite is added to the podium.
    """

    def __init__(self, patience, directory, num_elites):
        self._best_elites = [('', float('-inf')) for _ in range(num_elites)]

        self._num_elites = num_elites
        self._patience = patience

        _best_directory = directory
        self._best_elite_dir = os.path.join(_best_directory, 'best_elite')
        self._new_best_elite_path = os.path.join(self._best_elite_dir, '0_{i}_elite.pth')

        mkdir_p(self._best_elite_dir)

        self._bad_generation = True

    def init_from_infos(self, infos):

        self._best_elites = []
        for i, (elite_path, sc) in enumerate(infos['best_elites']):
            new_elite_path = self._new_best_elite_path.format(i=i)
            copy_file_from_to(elite_path, new_elite_path)
            self._best_elites.append((new_elite_path, sc))

    def record_elites(self, elites_and_scores):

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

    def best_elites(self):
        return self._best_elites
