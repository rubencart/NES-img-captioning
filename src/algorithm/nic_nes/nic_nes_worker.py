import copy
import gc
import logging
import os
import psutil
import time

import numpy as np
import torch
# from memory_profiler import profile
from algorithm.nic_nes.nic_nes_master import NESTask, NESResult
from algorithm.nic_nes.experiment import NESExperiment
from dist import WorkerClient
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import mkdir_p


class NESWorker(object):

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.rs = np.random.RandomState()
        self.worker_id = os.getpid()

        # redis client
        self.worker = WorkerClient(master_redis_cfg, relay_redis_cfg)
        self.exp = self.worker.get_experiment()

        self.eval_dir = os.path.join(self.exp['log_dir'], 'eval_{}'.format(os.getpid()))
        self.sensitivity_dir = os.path.join(self.exp['log_dir'], 'models', 'current')
        mkdir_p(self.eval_dir)

        setup_tuple = setup_worker(self.exp)
        self.config: Config = setup_tuple[0]
        self.policy: Policy = setup_tuple[1]
        self.experiment: NESExperiment = setup_tuple[2]

        self.placeholder = torch.FloatTensor(1)
        self.loader = self.experiment.get_trainloader()
        self.iloader = iter(self.experiment.get_trainloader())

    # @profile(stream=open('output/memory_profile_worker.txt', 'w+'))
    def run_worker(self):
        logger = logging.getLogger(__name__)

        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        exp, config, experiment, rs, worker, policy = \
            self.exp, self.config, self.experiment, self.rs, self.worker, self.policy

        it_id = 0

        while True:

            it_id += 1
            torch.set_grad_enabled(False)
            time.sleep(0.01)

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, NESTask)

            self.policy.set_ref_batch(task_data.ref_batch)

            if rs.rand() < config.eval_prob:
                logger.info('EVAL RUN')
                try:
                    result = self.accuracy(task_id, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    # Happens sometimes because master process is cleaning up files between iterations
                    logger.error(e)

            else:
                # logging.info('EVOLVE RUN')
                try:

                    result = self.fitness(task_id, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    # Happens sometimes because master process is cleaning up files between iterations
                    logging.error(e)

            del task_data
            gc.collect()
            # uncomment to write alive tensors to a file, for memory debugging.
            # (Nothing suspicious was found)
            # self.write_alive_tensors()

    def accuracy(self, task_id, policy, task_data):
        """
        Compute the accuracy of the current individual on the validation set
        """

        # keep track of memory usage
        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        current_path = task_data.current
        policy.set_model(current_path)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        score = policy.accuracy_on(self.experiment.valloader, self.config, self.eval_dir)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        logging.info('Iteration: {}, Val score: {}'.format(task_id, score))

        return NESResult(
            worker_id=self.worker_id,
            eval_score=score,
            mem_usage=max(mem_usages)
        )

    def fitness(self, task_id, policy, task_data):

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        # NIC-NES with vs without single batch:
        # take published batch or take random batch from own dataloader
        if self.config.single_batch:
            logging.debug('taking published batch')
            batch_data = copy.deepcopy(task_data.batch_data)
        else:
            # loader = self.experiment.get_trainloader()
            if self.loader.batch_size != task_data.batch_size:
                self.experiment.increase_loader_batch_size(task_data.batch_size)
            batch_data = next(self.iloader)

        current_path = task_data.current
        policy.set_model(current_path)
        current_params = torch.empty_like(policy.parameter_vector()).copy_(policy.parameter_vector())
        # copy.deepcopy(policy.parameter_vector())

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # compute sensitivity is necessary
        policy.calc_sensitivity(task_id, 0, batch_data, self.experiment.orig_batch_size(), self.sensitivity_dir)
        # theta <-- theta + noise
        noise_vector = policy.evolve_model(task_data.noise_stdev)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # compute fitness
        pos_fitness = policy.rollout(placeholder=self.placeholder,
                                     data=batch_data, config=self.config)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # theta <-- theta - noise (mirrored sampling) and compute fitness again
        policy.set_from_parameter_vector(current_params - torch.from_numpy(noise_vector))
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)
        neg_fitness = policy.rollout(placeholder=self.placeholder,
                                     data=batch_data, config=self.config)
        del current_params
        return NESResult(
            worker_id=self.worker_id,
            evolve_noise=noise_vector,
            fitness=np.stack((pos_fitness, neg_fitness)),
            mem_usage=max(mem_usages)
        )

    def write_alive_tensors(self):
        # from pytorch forum

        fn = os.path.join(self.eval_dir, 'alive_tensors.txt')
        to_write = '***************************\n'
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    to_write += 'type: {}, size: {} \n'.format(type(obj), obj.size())
            except Exception:
                pass
        with open(fn, 'a+') as f:
            f.write(to_write)


def start_and_run_worker(i, master_redis_cfg, relay_redis_cfg):
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )

    nes_worker = NESWorker(master_redis_cfg, relay_redis_cfg)
    nes_worker.run_worker()
