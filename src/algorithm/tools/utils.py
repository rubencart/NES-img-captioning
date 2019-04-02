
import errno
import os
import shutil
import sys
from collections import namedtuple

import numpy as np

ga_task_fields = ['elite', 'population', 'val_data', 'batch_data', 'parents', 'noise_stdev',
                  'log_dir']  # , 'policy'
GATask = namedtuple('GATask', field_names=ga_task_fields, defaults=(None,) * len(ga_task_fields))


result_fields = ['worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
                 'eval_return', 'mem_usage']
Result = namedtuple('Result', field_names=result_fields, defaults=(None,) * len(result_fields))

config_fields = [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch', 'stdev_decr_divisor',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq', 'num_dataloader_workers', 'log_dir',
    'return_proc_mode', 'episode_cutoff_mode', 'batch_size', 'max_nb_epochs', 'patience'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# To plot against scores:
# fig = plt.figure()
# plt.plot(x[:150], score_stats[1][:150], color='blue')
# plt.plot(np.arange(150), np.array(numstds[:150])*80 - 2.4, color='red')
# plt.savefig('tmp/1/both')
# plt.close(fig)
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


def random_state():
    rs = np.random.RandomState()
    return rs.randint(0, 2 ** 31 - 1)


def copy_file_from_to(path_to_old, path_to_new):
    shutil.copy(src=path_to_old, dst=path_to_new)


def remove_all_files_but(from_dir, but_list):
    for file in os.listdir(from_dir):
        path = os.path.join(from_dir, file)

        if os.path.isfile(path) and path not in but_list:
            os.remove(path)


def remove_files(from_dir, rm_list):
    for file in os.listdir(from_dir):
        path = os.path.join(from_dir, file)

        if os.path.isfile(path) and path in rm_list:
            os.remove(path)


def remove_all_files_from_dir(from_dir):
    for file in os.listdir(from_dir):
        path = os.path.join(from_dir, file)

        if os.path.isfile(path):
            os.remove(path)


def remove_file_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]
