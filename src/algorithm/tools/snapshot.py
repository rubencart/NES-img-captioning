import json
import logging
import os


from algorithm.tools.experiment import Experiment
from algorithm.tools.iteration import Iteration
from algorithm.policies import Policy
from algorithm.tools.statistics import Statistics
from algorithm.tools.utils import mkdir_p, remove_file_with_pattern

logger = logging.getLogger(__name__)


def save_snapshot(stats: Statistics, it: Iteration, experiment: Experiment, policy: Policy):

    mkdir_p(experiment.snapshot_dir())

    filename = save_infos(experiment, it, stats)

    logger.info('Saved snapshot {}'.format(filename))


def save_infos(experiment, it, stats):
    directory = experiment.snapshot_dir()

    infos_pattern = r'z_info_e[0-9]*?_i[0-9]*?-[0-9]*?.json'
    remove_file_with_pattern(infos_pattern, directory)

    filename = 'z_info_e{e}_i{i}-{n}.json'.format(e=it.epoch(), i=it.iteration(),
                                                  n=experiment.orig_trainloader_lth())
    assert not os.path.exists(os.path.join(directory, filename))
    infos = {
        **stats.to_dict(),
        **it.to_dict(),
        **experiment.to_dict(),
    }
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(infos, f)
    return filename
