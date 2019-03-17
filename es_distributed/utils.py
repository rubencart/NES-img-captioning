import json
import os

import numpy as np

from es_distributed.main import mkdir_p


def save_snapshot(acc_stats, epoch, iteration, parents, policy, trainloader_length):
    snapshot_dir = 'snapshots/es_master_{}'.format(os.getpid())
    filename = 'info_e{e}_i{i}:{n}.json'.format(e=epoch, i=iteration, n=trainloader_length)
    mkdir_p(snapshot_dir)
    assert not os.path.exists(os.path.join(snapshot_dir, filename))

    infos = {
        'rewards': acc_stats[0],
        'iter': iteration,
        'epoch': epoch,
        'parents': [parent.__dict__ for (_, parent) in parents],
    }

    with open(os.path.join(snapshot_dir, filename), 'w') as f:
        json.dump(infos, f)

    net_filename = 'elite_params_e{e}_i{i}:{n}_r{r}.pth' \
        .format(e=epoch, i=iteration, n=trainloader_length, r=acc_stats[0][-1])
    policy.save(path=snapshot_dir, filename=net_filename)

    return os.path.join(snapshot_dir, filename)



def plot_stats(log_dir, score_stats=None, **kwargs):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    if score_stats:
        fig = plt.figure()
        x = np.arange(len(score_stats[1]))
        plt.fill_between(x=x, y1=score_stats[0], y2=score_stats[2], facecolor='blue', alpha=0.3)
        plt.plot(x.copy(), score_stats[1], label='Training loss', color='blue')
        # plt.savefig(log_dir + '/loss_plot_{i}.png'.format(i=i))
        plt.savefig(log_dir + '/loss_plot.png')
        plt.close(fig)

    for (name, (lst, label)) in kwargs.items():
        fig = plt.figure()
        plt.plot(np.arange(len(lst)), lst, label=label)
        # plt.savefig(log_dir + '/time_plot_{i}.png'.format(i=i))
        plt.savefig(log_dir + '/{}_plot.png'.format(name))
        plt.close(fig)

