import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# print('importing rest')
import argparse
import json
import logging
import os
import signal
import sys
import time

# from memory_profiler import profile_exp
import psutil

from dist import RelayClient
from algorithm.ga_master import GAMaster
from algorithm.ga_worker import GAWorker


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('who', type=str, choices=['master', 'workers'])

    # MASTER
    parser.add_argument('--algo', type=str, default='ga', help='')
    parser.add_argument('--exp_file', type=str, default='experiments/mnist_ga.json', help='')
    parser.add_argument('--master_socket_path', type=str, default='/tmp/es_redis_master.sock', help='')
    parser.add_argument('--log_dir', type=str, default='', help='')
    parser.add_argument('--plot', action='store_true', default=True)

    # WORKER
    parser.add_argument('--master_host', type=str, default='localhost', help='')
    parser.add_argument('--master_port', type=int, default=6379, help='')
    parser.add_argument('--relay_socket_path', type=str, default='/tmp/es_redis_relay.sock', help='')
    parser.add_argument('--num_workers', type=int, default=2, help='')

    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stdout)

    if args.who == 'master':
        master(args.algo, args.exp_file, args.master_socket_path, args.log_dir, args.plot)
    elif args.who == 'workers':
        workers(args.algo, args.master_host, args.master_port, args.relay_socket_path, args.num_workers)


def import_algo(name):
    # todo add support for more ES algorithms like Novelty Search/...
    # if name == 'es':
    #     from . import es as algo
    # elif name == 'ns-es' or name == "nsr-es":
    #     from . import nses as algo
    if name == 'ga':
        from algorithm import ga_master as algo
    # elif name == 'rs':
    #     from . import rs as algo
    else:
        raise NotImplementedError()
    return algo


# @profile_exp(stream=open('memory_profiler.log', 'w+'))
def master(algo, exp_file, master_socket_path, log_dir, plot):
    # Start the master
    # assert exp_file is not None, 'Must provide exp_file to the master'

    # import mkl
    # mkl.set_num_threads(1)

    if exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False

    # todo
    # algo = import_algo(algo)
    ga_master = GAMaster()
    ga_master.run_master({'unix_socket_path': master_socket_path}, exp=exp, plot=plot)  # log_dir=log_dir


# @profile_exp(stream=open('memory_profiler.log', 'w+'))
def workers(algo, master_host, master_port, relay_socket_path, num_workers):
    # Start the relay
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}

    if os.fork() == 0:
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return

    # import mkl
    # mkl.set_num_threads(1)

    # Start the workers
    # todo
    # algo = import_algo(algo)

    # wait for master process to have uploaded tasks, otherwise errors
    # because workers start on cached tasks and files don't exist
    time.sleep(10)

    num_workers = num_workers if num_workers else os.cpu_count() - 2
    worker_ids = spawn_workers(num_workers, algo, master_redis_cfg, relay_redis_cfg)
    while True:
        if psutil.virtual_memory().percent > 90.0:
            logging.warning('!!!!! ---  Killing all workers   --- !!!!!')
            [os.kill(pid, signal.SIGKILL) for pid in worker_ids]
            worker_ids = spawn_workers(num_workers, algo, master_redis_cfg, relay_redis_cfg)
        else:
            time.sleep(60)
    # os.wait()


def spawn_workers(num_workers, algo, master_redis_cfg, relay_redis_cfg):
    logging.info('Spawning {} workers'.format(num_workers))
    worker_ids = []
    for _ in range(num_workers):
        new_pid = os.fork()
        if new_pid == 0:

            # print('importing mkl, setting num threads')
            # import mkl
            # mkl.set_num_threads(1)

            # todo
            ga_worker = GAWorker()

            # todo pass along worker id to ensure unique
            ga_worker.run_worker(master_redis_cfg, relay_redis_cfg)
            return
        else:
            worker_ids.append(new_pid)
    return worker_ids


if __name__ == '__main__':
    run()
