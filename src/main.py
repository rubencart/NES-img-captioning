import os

# this should go before everything else!
# tells the OS to not multithread low-level lin alg operations, so every worker stays nicely
# on 1 cpu
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import json
import logging
import os
import time
import psutil

from algorithm.tools.utils import get_platform
from dist import RelayClient


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('who', type=str, choices=['master', 'workers'])

    # MASTER
    parser.add_argument('--algo', type=str, default='nic_es', help='')
    parser.add_argument('--exp_file', type=str, default='experiments/mnist_es.json', help='')
    parser.add_argument('--master_socket_path', type=str, default='/tmp/es_redis_master_6379.sock', help='')
    parser.add_argument('--plot', action='store_true', default=True)

    # WORKER
    parser.add_argument('--master_host', type=str, default='localhost', help='')
    parser.add_argument('--master_port', type=int, default=6379, help='')
    parser.add_argument('--relay_socket_path', type=str, default='/tmp/es_redis_relay_6379.sock', help='')
    parser.add_argument('--num_workers', type=int, help='')

    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )

    if args.who == 'master':
        master(args.algo, args.exp_file, args.master_socket_path, args.plot)
    elif args.who == 'workers':
        workers(args.algo, args.master_host, args.master_port, args.relay_socket_path, args.num_workers)


def master(algo, exp_file, master_socket_path, plot):
    # start the master

    if exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False

    if algo == 'nic_es':
        from algorithm.nic_es.nic_es_master import ESMaster
        logging.info('RUNNING NIC-ES')
        master_alg = ESMaster(exp, {'unix_socket_path': master_socket_path})
    else:
        # algo == 'nic_nes':
        logging.info('RUNNING NIC-NES')
        from algorithm.nic_nes.nic_nes_master import NESMaster
        master_alg = NESMaster(exp, {'unix_socket_path': master_socket_path})

    master_alg.run_master(plot=plot)


def workers(algo, master_host, master_port, relay_socket_path, num_workers):

    # start the relay process
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return

    if algo == 'nic_es':
        from algorithm.nic_es.nic_es_worker import start_and_run_worker
        run_func = start_and_run_worker
    else:
        # algo == 'nic_nes':
        from algorithm.nic_nes.nic_nes_worker import start_and_run_worker
        run_func = start_and_run_worker

    # wait for master process to have uploaded tasks, otherwise we get errors
    # because workers start on old tasks that were cached by redis
    time.sleep(5)
    if num_workers == -1:
        # for testing purposes if num_workers = -1 the current process just acts as only worker
        run_func(0, master_redis_cfg, relay_redis_cfg)
    else:

        num_workers = num_workers if num_workers else os.cpu_count() - 2
        processes = spawn_workers(num_workers, run_func, master_redis_cfg, relay_redis_cfg)
        counter = 0
        try:
            while True:
                # sometimes workers die because of parallel computing issues with pytorch
                # dirty workaround here: count alive workers at intervals and relaunch some if necessary
                alive_procs = [p for p in processes if p.is_alive()]
                nb_alive = len(alive_procs)
                if num_workers > nb_alive:
                    logging.warning('****************************************************')
                    logging.warning('SPAWNING {} NEW WORKERS'.format(num_workers - nb_alive))
                    logging.warning('****************************************************')
                    new_procs = spawn_workers(num_workers - nb_alive, run_func, master_redis_cfg, relay_redis_cfg)
                    processes = alive_procs + new_procs

                if psutil.virtual_memory().percent > 90.0:
                    # a memory leak occurs when sampling fitness function is used, since I couldn't find
                    # the reason this is a dirty workaround again: just kill all workers and relaunch them
                    logging.warning('****************************************************')
                    logging.warning('!!!!! ---  KILLING ALL WORKERS   --- !!!!!')
                    logging.warning('****************************************************')

                    [p.kill() for p in processes]
                    processes = spawn_workers(num_workers, run_func, master_redis_cfg, relay_redis_cfg)
                    counter += 1
                if counter > 20:
                    [p.kill() for p in processes]
                    break
                if nb_alive == 0:
                    break
                time.sleep(60)
        except KeyboardInterrupt:
            [p.kill() for p in processes]


def spawn_workers(num_workers, run_func, master_redis_cfg, relay_redis_cfg):
    logging.info('Spawning {} workers'.format(num_workers))
    worker_ids = []
    for _id in range(num_workers):

        p = mp.Process(target=run_func, args=(0, master_redis_cfg, relay_redis_cfg))
        p.start()
        worker_ids += [p]

    return worker_ids


if __name__ == '__main__':
    import torch.multiprocessing as mp

    if get_platform() == 'OS X':
        mp.set_start_method('forkserver', force=True)

    run()
