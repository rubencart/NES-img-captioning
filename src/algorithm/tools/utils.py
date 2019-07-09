import copy
import errno
import gc
import logging
import os
import re
import shutil
import sys
import torch
from collections import namedtuple

import numpy as np

config_fields = [
    'l2coeff', 'noise_stdev', 'stdev_divisor', 'eval_prob', 'snapshot_freq', 'log_dir',
    'batch_size', 'patience', 'val_batch_size', 'num_val_batches',
    'num_val_items', 'cuda', 'max_nb_iterations', 'ref_batch_size', 'bs_multiplier', 'stepsize_divisor',
    'single_batch', 'schedule_limit', 'schedule_start'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))


def log(name, result):
    try:
        # result = round(result, 2)
        result = '{:g}'.format(float('{:.{p}g}'.format(result, p=3)))
    except Exception:
        pass
    logging.info('| %s: %s | %s %s |', name,
                 ' ' * (max(19 - len(name), 0)), ' ' * (max(10 - len(str(result)), 0)),
                 result)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def write_alive_tensors(self):
    # from pytorch forum
    fn = os.path.join(self.eval_dir, 'alive_tensors.txt')

    to_write = '***************************\n'
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                to_write += 'type: {}, size: {} \n'.format(type(obj), obj.size())
        except:
            pass

    with open(fn, 'a+') as f:
        f.write(to_write)


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


def remove_file_with_pattern(pattern, directory):
    for file in os.listdir(directory):
        if re.search(pattern, file):
            os.remove(os.path.join(directory, file))


def find_file_with_pattern(pattern, directory):
    # pattern like r'z_info_e[0-9]*?_i[0-9]*?-[0-9]*?.json'
    for file in os.listdir(directory):
        if re.search(pattern, file):
            return file


def check_if_filepath_exists(path):
    return os.path.isfile(path)


def get_platform():
    """
    From https://www.webucator.com/how-to/how-check-the-operating-system-with-python.cfm
    """
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


def get_ciders_from_sc(hist, infos):
    # with sc: 0.75 s/batch (16)
    # with xent: 0.065 s/batch (16)
    # nb samples in Ds: 113287
    # params 2 865 808

    # with open('../instance_logs/logs_xent_fc_128/checkpt/histories_test_0.pkl', 'rb') as f:
    #      hist = pickle.load(f)

    # with open('../instance_logs/logs_xent_fc_128/checkpt/infos_test_0.pkl', 'rb') as f:
    #      infos = pickle.load(f)

    # with open('../instance_logs/instance3_logs_2/test_0_sc_128_checkpt_cont_2/histories_test_0.pkl', 'rb') as f:
    #      histsc2 = pickle.load(f)

    # with open('../instance_logs/instance3_logs_2/test_0_sc_128_checkpt_cont_2/infos_test_0-best.pkl', 'rb') as f:
    #      infossc2 = pickle.load(f)

    # plt.plot(*get_ciders(histsc2, infossc2), label='self-critical RL')
    # plt.legend()
    # plt.savefig('...')

    keys = list(hist['val_result_history'].keys())
    ciders = [hist['val_result_history'][nb]['lang_stats']['CIDEr'] for nb in keys]
    bs = infos['opt'].batch_size
    keys = np.asarray(keys) * bs
    return keys, np.asarray(ciders)


def plot_ciders_vs_something_nicely(time_xent, ciders_xent, time_sc, ciders_sc):
    from matplotlib import pyplot as plt

    plt.plot(time_xent, ciders_xent, label='XENT')
    plt.plot(time_sc, ciders_sc, label='Self-critical RL')
    plt.axhline(ciders_sc.max(), linestyle='dashed', color='green', lw=0.5)
    plt.text(-2, 0.93, round(ciders_sc.max(), 3), color='green')
    plt.legend(loc='lower right')
    plt.xlabel('Aantal uur')
    plt.ylabel('CIDEr')
    plt.savefig('./logs/sc_xent_time.pdf')
    plt.close()


def cst_from_infos(infos):
    if 'best_acc_so_far_stats' in infos:
        ciders = np.asarray(infos['best_acc_so_far_stats'])
    else:
        ciders = np.maximum.accumulate(infos['acc_stats'])
    samples = np.cumsum(infos['bs_stats'])
    times = np.cumsum(infos['time_stats'])
    return ciders, samples, times


def combine_diff_lengths(*arrays):
    # will give jumps in output at points where one of the arrays ends --> better padding
    sorted_arrays = sorted(arrays, key=lambda a: len(a), reverse=False)
    lengths = [len(a) for a in sorted_arrays]
    result = np.zeros(lengths[-1])

    for nb, (lower, upper) in enumerate(zip([0] + lengths, lengths)):
        for j in range(lower, upper):
            result[j] = np.asarray([a[j] for a in sorted_arrays[nb:]]).mean()

    return result


def combine_diff_lengths_pad(*arrays):
    length = max([len(a) for a in arrays])
    result = np.zeros(length)
    copied_arrays = copy.deepcopy(arrays)
    padded_arrays = []
    for a in copied_arrays:
        padded_arrays.append(np.concatenate((a, [a[-1] for _ in range(length - len(a))])))
    for i in range(length):
        result[i] = np.asarray([a[i] for a in padded_arrays]).mean()
    return result


def sample_at(raster, axis, values):
    result = []
    for i, rast_pt in enumerate(raster):

        upper = lower = 0
        if rast_pt > axis[-1]:
            break
        for k, ax in enumerate(axis):
            if ax == rast_pt:
                upper = lower = k
                break
            elif ax > rast_pt:
                upper = k
                lower = max(k - 1, lower)
                break
        result.append((values[lower] + values[upper]) / 2)
    return np.asarray(result)


def rasterize(*coords):
    # coords like [ [ (1, 10), (2, 20) ] , [...] ]

    axes = [[a for (a, _) in arr] for arr in coords]
    values = [[v for (_, v) in arr] for arr in coords]
    minim = int(min(a[0] for a in axes))
    maxim = int(max(a[-1] for a in axes))
    step = int(min([a[1] - a[0] for a in axes]))

    raster = np.arange(minim, maxim, step)

    rasterized = []
    for i in range(len(axes)):
        rasterized.append(sample_at(raster, axes[i], values[i]))

    return [raster[:len(rized)] for rized in rasterized], rasterized


def tournament(pop, t, offspr):
    rs = np.random.RandomState()
    return [min(rs.choice(np.arange(pop), t, replace=False)) for _ in range(offspr)]


def count_in_tournament(p, t, o):
    tourn = tournament(p, t, o)
    counts = [tourn.count(c) for c in range(p)]
    return counts


def avg_c_in_t(p, t, o, x):
    cts = np.empty([x, p], dtype=float)
    for i in range(x):
        cts[i] = count_in_tournament(p, t, o)
    return cts.mean(0)
