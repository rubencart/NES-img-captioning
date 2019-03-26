import json
import logging
import os

from experiment import Experiment
from iteration import Iteration
from policies import Policy
from statistics import Statistics
from utils import mkdir_p

logger = logging.getLogger(__name__)


def save_snapshot(stats: Statistics, it: Iteration, experiment: Experiment, policy: Policy):

    # snapshot_dir = 'snapshots/es_{}_master_{}'.format(experiment.mode(), os.getpid())
    filename = 'info_e{e}_i{i}:{n}.json'.format(e=it.epoch(), i=it.iteration(),
                                                n=experiment.orig_trainloader_lth())
    mkdir_p(experiment.snapshot_dir())
    assert not os.path.exists(os.path.join(experiment.snapshot_dir(), filename))

    infos = {
        **stats.to_dict(),
        **it.to_dict(),
        **experiment.to_dict(),
    }

    with open(os.path.join(experiment.snapshot_dir(), filename), 'w') as f:
        json.dump(infos, f)

    net_filename = 'elite_params_e{e}_i{i}:{n}_r{r}.pth' \
        .format(e=it.epoch(), i=it.iteration(), n=experiment.orig_trainloader_lth(),
                r=round(stats.acc_stats()[-1], 2))

    # todo necessary?
    policy.save(path=experiment.snapshot_dir(), filename=net_filename)

    logger.info('Saved snapshot {}'.format(filename))
    # return os.path.join(snapshot_dir, filename)
