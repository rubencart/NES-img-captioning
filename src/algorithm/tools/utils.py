import copy
import errno
import os
import re
import shutil
import sys
from collections import namedtuple

import numpy as np


result_fields = ['worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
                 'eval_return', 'mem_usage', 'evaluated_cand', 'evaluated_cand_id',
                 'score']
GAResult = namedtuple('GAResult', field_names=result_fields, defaults=(None,) * len(result_fields))

config_fields = [
    'l2coeff', 'noise_stdev', 'stdev_divisor', 'eval_prob', 'snapshot_freq', 'log_dir',
    'return_proc_mode', 'batch_size', 'patience', 'val_batch_size', 'num_val_batches',
    'num_val_items', 'cuda', 'max_nb_epochs', 'ref_batch_size', 'bs_multiplier', 'stepsize_divisor',
    'single_batch', 'schedule_limit'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))


class IterationFailedException(Exception):
    pass


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
    # print('[pid {pid}] random state: {rs}'.format(pid=os.getpid(), rs=rs.randint(0, 2 ** 31 - 1)))
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

    # plt.plot(nes_times / 3600, smooth(nes_ciders, 6), color='blue', label='NIC-NES')
    # plt.plot(es_times_59[:650] / 3600, smooth(np.maximum.accumulate(es_ciders_59[:650]), 10), color='green',
    #          label='NIC-ES')
    # plt.plot(times_sc[:70] / 3600, smooth(np.maximum.accumulate(ciders_sc[:70]), 2), color='orange',
    #          label='Self-critical RL')
    #
    # plt.axhline(ciders_sc.max(), linestyle='dashed', color='orange', lw=0.5)
    # plt.text(-1, ciders_sc.max() - 0.01, round(ciders_sc.max(), 3), color='orange')
    #
    # plt.axhline(nes_ciders.max(), linestyle='dashed', color='blue', lw=0.5)
    # plt.text(-1, nes_ciders.max() - 0.01, round(nes_ciders.max(), 3), color='blue')
    #
    # plt.axhline(es_ciders_59.max(), linestyle='dashed', color='green', lw=0.5)
    # plt.text(-1, es_ciders_59.max() - 0.01, round(es_ciders_59.max(), 3), color='green')
    #
    # plt.legend(loc='lower right')
    # plt.xlabel('Aantal uur')
    # plt.ylabel('CIDEr')
    # plt.savefig('./logs/sc_nes_es_time_smooth.pdf')
    # plt.close()


def smooth(x, n):
    tmp = np.array(x, copy=True)
    for i in range(n, len(x) - n):
        tmp[i] = x[i - n:i + n].mean()
    return tmp


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

        # upper = next(i for (i, a) in enumerate(axes) if a >= rast_pt)
        upper = lower = 0
        if rast_pt > axis[-1]:
            # upper = lower = len(axes) - 1
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
    # coords = [ [ (1, 10), (2, 20) ] , [...] ]
    # print(coords)

    axes = [[a for (a, _) in arr] for arr in coords]
    values = [[v for (_, v) in arr] for arr in coords]
    minim = int(min(a[0] for a in axes))
    maxim = int(max(a[-1] for a in axes))
    step = int(min([a[1] - a[0] for a in axes]))

    raster = np.arange(minim, maxim, step)

    rasterized = []
    for i in range(len(axes)):
        # print(raster, axes[i], values[i])
        rasterized.append(sample_at(raster, axes[i], values[i]))

    return [raster[:len(rized)] for rized in rasterized], rasterized


# ciders_93, samples_93, times_93 = cst_from_infos(infos_93)
# ciders_3486, samples_3486, times_3486 = cst_from_infos(infos_3486)
# ciders_5506, samples_5506, times_5506 = cst_from_infos(infos_5506)
# ciders_3566, samples_3566, times_3566 = cst_from_infos(infos_3566)
# ciders_4067, samples_4067, times_4067 = cst_from_infos(infos_4067)
# ciders_5945, samples_5945, times_5945 = cst_from_infos(infos_5945)
# ciders_19695, samples_19695, times_19695 = cst_from_infos(infos_19695)
#
# tssamples, tsciders = rasterize(list(zip(samples_93, ciders_93)),
#                                 list(zip(samples_3486, ciders_3486)),
#                                 list(zip(samples_5506, ciders_5506)))
#
# rssamples, rsciders = rasterize(list(zip(samples_3566, ciders_3566)),
#                                 list(zip(samples_4067, ciders_4067)),
#                                 list(zip(samples_5945, ciders_5945)),
#                                 list(zip(samples_19695, ciders_19695)))
#
# tstimes, tstciders = rasterize(list(zip(times_93, ciders_93)),
#                                 list(zip(times_3486, ciders_3486)),
#                                 list(zip(times_5506, ciders_5506)))
#
# rstimes, rstciders = rasterize(list(zip(times_3566, ciders_3566)),
#                                 list(zip(times_4067, ciders_4067)),
#                                 list(zip(times_5945, ciders_5945)),
#                                 list(zip(times_19695, ciders_19695)))
#
# plt.plot(times_93, ciders_93, color='red', label='93')
# plt.plot(times_3486, ciders_3486, color='red', label='3486')
# plt.plot(times_5506, ciders_5506, color='red', label='5506')
# plt.plot(times_3566, ciders_3566, color='blue', label='3566')
# plt.plot(times_4067, ciders_4067, color='blue', label='4067')
# plt.plot(times_5945, ciders_5945, color='blue', label='5945')
# plt.plot(times_19695, ciders_19695, color='blue', label='19695')
#
#
# plt.plot(samples_93, ciders_93, color='red', label='93')
# plt.plot(samples_3486, ciders_3486, color='red', label='3486')
# plt.plot(samples_5506, ciders_5506, color='red', label='5506')
# plt.plot(samples_3566, ciders_3566, color='blue', label='3566')
# plt.plot(samples_4067, ciders_4067, color='blue', label='4067')
# plt.plot(samples_5945, ciders_5945, color='blue', label='5945')
# plt.plot(samples_19695, ciders_19695, color='blue', label='19695')
#
# plt.plot(tssamples[1], ctsciders, color='red', label='TS')
# plt.plot(rssamples[2], crsciders, color='blue', label='UWS')
#
# plt.legend(loc='lower right')
# plt.xlabel('Aantal s')
# plt.ylabel('CIDEr')
# plt.savefig('./logs/uws_vs_ts_samples_comb.pdf')
# plt.close()
#
# xtimes_2540 = np.concatenate((times_2540, [times_2540[-1] + i*(times_2540[-1]-times_2540[-2]) for i in range(250) ]))
# xciders_2540 = np.concatenate((ciders_2540, [ciders_2540[-1] for _ in range(250)]))
