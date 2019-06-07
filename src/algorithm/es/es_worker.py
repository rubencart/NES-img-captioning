import copy
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch
# from memory_profiler import profile
from algorithm.es.es_master import ESTask, ESResult
from algorithm.tools.experiment import ESExperiment
from dist import WorkerClient
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import mkdir_p

logger = logging.getLogger(__name__)


class ESWorker(object):

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.rs = np.random.RandomState()
        self.worker_id = self.rs.randint(2 ** 31)

        # redis client
        self.worker = WorkerClient(master_redis_cfg, relay_redis_cfg)
        self.exp = self.worker.get_experiment()
        assert self.exp['mode'] in ['seeds', 'nets'], '{}'.format(self.exp['mode'])

        self.eval_dir = os.path.join(self.exp['log_dir'], 'eval_{}'.format(os.getpid()))
        self.sensitivity_dir = os.path.join(self.exp['log_dir'], 'models', 'current')
        mkdir_p(self.eval_dir)

        setup_tuple = setup_worker(self.exp)
        self.config: Config = setup_tuple[0]
        self.policy: Policy = setup_tuple[1]
        self.experiment: ESExperiment = setup_tuple[2]

        self.placeholder = torch.FloatTensor(1)

    # @profile(stream=open('output/memory_profile_worker.txt', 'w+'))
    # @profile
    def run_worker(self):
        logger = logging.getLogger(__name__)

        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        exp, config, experiment, rs, worker, policy = \
            self.exp, self.config, self.experiment, self.rs, self.worker, self.policy

        _it_id = 0

        while True:

            _it_id += 1
            torch.set_grad_enabled(False)
            time.sleep(0.01)

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, ESTask)

            self.policy.set_ref_batch(task_data.ref_batch)

            if rs.rand() < config.eval_prob:
                logger.info('EVAL RUN')
                try:
                    result = self.accuracy(task_id, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logger.error(e)

                except Exception as e:
                    raise Exception

            else:
                # logging.info('EVOLVE RUN')
                try:

                    result = self.fitness(_it_id, policy, task_data, task_id)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logging.error(e)

                except Exception as e:
                    raise Exception

            # del policy, task_data
            del task_data
            gc.collect()
            # self.write_alive_tensors()

    def accuracy(self, task_id, policy, task_data):

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        current_path = task_data.current
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        policy.set_model(current_path)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        score = policy.accuracy_on(self.experiment.valloader, self.config, self.eval_dir)
        # print(score)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        logger.info('Iteration: {}, CIDEr: {}'.format(task_id, score))

        # del task_data, cand, score
        return ESResult(
            worker_id=self.worker_id,
            eval_score=score,
            mem_usage=max(mem_usages)
        )

    def fitness(self, it_id, policy, task_data, task_id):

        # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?
        # todo   --> also 'rollouts with noise': do they mean added noise in param space?
        #            or in action space? bc we don't do this, this would be like sampling vs greedy
        # "If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        #         Otherwise, no action noise will be added."

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        if self.config.single_batch:
            batch_data = copy.deepcopy(task_data.batch_data)
        else:
            loader = self.experiment.get_trainloader()
            if loader.batch_size != task_data.batch_size:
                self.experiment.increase_loader_batch_size(task_data.batch_size)
            batch_data = next(iter(loader))

        current_path = task_data.current
        policy.set_model(current_path)
        current_params = policy.parameter_vector()

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        policy.calc_sensitivity(task_id, 0, batch_data, self.experiment.orig_batch_size(), self.sensitivity_dir)
        # theta <-- theta + noise
        noise_vector = policy.evolve_model(task_data.noise_stdev)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        pos_fitness = policy.rollout(placeholder=self.placeholder,
                                     data=batch_data, config=self.config)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # theta <-- theta - noise (mirrored sampling)
        policy.set_from_parameter_vector(current_params - torch.from_numpy(noise_vector))

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        neg_fitness = policy.rollout(placeholder=self.placeholder,
                                     data=batch_data, config=self.config)
        del current_params
        return ESResult(
            worker_id=self.worker_id,
            evolve_noise=noise_vector,
            fitness=np.array([pos_fitness, neg_fitness]),   # , dtype=np.float32
            mem_usage=max(mem_usages)
        )

    def write_alive_tensors(self):
        fn = os.path.join(self.eval_dir, 'alive_tensors.txt')

        to_write = '***************************\n'
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # print(type(obj), obj.size())
                    # print(reduce(torch.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
                    to_write += 'type: {}, size: {} \n'.format(type(obj), obj.size())
            except:
                pass

        with open(fn, 'a+') as f:
            f.write(to_write)


def start_and_run_worker(i, master_redis_cfg, relay_redis_cfg):
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )  # stream=sys.stdout)

    es_worker = ESWorker(master_redis_cfg, relay_redis_cfg)
    es_worker.run_worker()
