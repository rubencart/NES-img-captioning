import errno
import json
import os
import sys
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.pyplot

# import matplotlib
# matplotlib.use("TkAgg")
#
# from matplotlib import pyplot as plt

import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_snapshot(acc_stats, time_stats, norm_stats, score_stats,
                  epoch, iteration, parents, policy, trainloader_length):
    snapshot_dir = 'snapshots/es_master_{}'.format(os.getpid())
    filename = 'info_e{e}_i{i}:{n}.json'.format(e=epoch, i=iteration, n=trainloader_length)
    mkdir_p(snapshot_dir)
    assert not os.path.exists(os.path.join(snapshot_dir, filename))

    infos = {
        'acc_stats': acc_stats,
        'time_stats': time_stats,
        'norm_stats': norm_stats,
        'score_stats': score_stats,
        'iter': iteration,
        'epoch': epoch,
        'parents': [parent.__dict__ for (_, parent) in parents if parent],
    }

    with open(os.path.join(snapshot_dir, filename), 'w') as f:
        json.dump(infos, f)

    net_filename = 'elite_params_e{e}_i{i}:{n}_r{r}.pth' \
        .format(e=epoch, i=iteration, n=trainloader_length, r=round(acc_stats[0][-1], 2))
    policy.save(path=snapshot_dir, filename=net_filename)

    return os.path.join(snapshot_dir, filename)


def plot_stats(log_dir, plt, score_stats=None, **kwargs):
    # import matplotlib
    # import matplotlib.pyplot as plt
    # if sys.platform == 'darwin':
    #     matplotlib.use('TkAgg')
    mkdir_p(log_dir)

    if score_stats:
        fig = plt.figure()
        x = np.arange(len(score_stats[1]))
        plt.fill_between(x=x, y1=score_stats[0], y2=score_stats[2], facecolor='blue', alpha=0.3)
        plt.plot(x, score_stats[1], color='blue')
        plt.title('Training loss')
        # plt.savefig(log_dir + '/loss_plot_{i}.png'.format(i=i))
        plt.savefig(log_dir + '/loss_plot.png')
        plt.close(fig)

    for (name, (lst, label)) in kwargs.items():
        fig = plt.figure()
        plt.plot(np.arange(len(lst)), lst)
        plt.title(label)
        # plt.savefig(log_dir + '/time_plot_{i}.png'.format(i=i))
        plt.savefig(log_dir + '/{}_plot.png'.format(name))
        plt.close(fig)


def extract_stds_from_log(filename):
    # eg 'logs/logs/es_master_16799/log.txt'
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # print(content[:15])
    # --> ['********** Iteration 1 **********',
    #  '----------------------------------',
    #  '| RewMax              | -2.26    |',
    #  '| RewMean             | -2.42    |',
    #  '| RewMin              | -2.77    |',
    #  '| RewStd              | 0.0852   |',
    #  '| Norm                | 13.6     |',
    #  '| NoiseStd            | 0.01     |',
    #  '| MaxAcc              | 0.146    |',
    #  '| UniqueWorkers       | 2        |',
    #  '| UniqueWorkersFrac   | 0.00388  |',
    #  '| TimeElapsedThisIter | 6.9      |',
    #  '| TimeElapsed         | 7.05     |',
    #  '| MemUsage            | 4.7      |',
    #  '----------------------------------']
    stds = [c for (i, c) in enumerate(content) if (i - 7) % 15 == 0]
    numstds = [float(s[24:-1].strip()) for s in stds]
    return numstds


def readable_bytes(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return '%3.1f%s%s' % (num, unit, suffix)
        num /= 1024.0
    return '%.1f%s%s' % (num, 'Yi', suffix)
