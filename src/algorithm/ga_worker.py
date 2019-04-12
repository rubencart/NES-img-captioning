
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch

from algorithm.tools.experiment import Experiment
from dist import WorkerClient
from algorithm.policies import Policy, PolicyFactory, SuppDataset, Net
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import GATask, Result, mkdir_p


class GAWorker(object):

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.rs = np.random.RandomState()
        self.worker_id = self.rs.randint(2 ** 31)
        self.offspring_dir = ''
        self.offspring_path = ''
        self.eval_dir = ''

        # redis client
        self.worker = WorkerClient(master_redis_cfg, relay_redis_cfg)
        self.exp = self.worker.get_experiment()
        assert self.exp['mode'] in ['seeds', 'nets'], '{}'.format(self.exp['mode'])

        self.offspring_dir = os.path.join(self.exp['log_dir'], 'models', 'offspring')
        mkdir_p(self.offspring_dir)
        self.offspring_path = os.path.join(self.offspring_dir, '{w}_{i}_offspring_params.pth')

        self.eval_dir = os.path.join(self.exp['log_dir'], 'eval_{}'.format(os.getpid()))
        mkdir_p(self.eval_dir)

        setup_tuple = setup_worker(self.exp)
        self.config: Config = setup_tuple[0]
        # policy: Policy = setup_tuple[1]
        self.experiment: Experiment = setup_tuple[2]

    # @profile_exp(stream=open('profile_exp/memory_profile_worker.log', 'w+'))
    def run_worker(self):
        logger = logging.getLogger(__name__)

        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        # self.exp = self.worker.get_experiment()
        exp, config, experiment, rs, worker = self.exp, self.config, self.experiment, self.rs, self.worker

        _it_id = 0

        while True:

            _it_id += 1
            torch.set_grad_enabled(False)
            time.sleep(0.01)
            mem_usages = []

            # deadlocks!!!! if elite hasn't been evaluated when nb files > 2 pop
            # if len(os.listdir(offspring_dir)) > 2 * exp['population_size']:
            #     time.sleep(10)
            #     continue

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, GATask)

            policy: Policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']), mode=exp['mode'],
                                                  net=Net(exp['net']), exp=exp)

            if rs.rand() < config.eval_prob:
                logger.info('EVAL RUN')
                try:
                    result = self.accuracy(config, experiment, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logger.error(e)

                except Exception as e:
                    raise Exception

            else:
                # logging.info('EVOLVE RUN')
                try:

                    result = self.fitness(config, _it_id, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logging.error(e)

                except Exception as e:
                    raise Exception

            del policy, task_data
            gc.collect()

    def accuracy(self, config, experiment, policy, task_data):

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        index = self.rs.randint(len(task_data.elites))
        cand_id, cand = task_data.elites[index]
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        policy.set_model(cand)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        score = policy.accuracy_on(experiment.valloader, config, self.eval_dir)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # del task_data, cand, score
        return Result(
            worker_id=self.worker_id,
            score=score,
            evaluated_cand_id=cand_id,
            evaluated_cand=cand,
            mem_usage=max(mem_usages)
        )

    def fitness(self, config, it_id, policy, task_data):

        # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?

        index = self.rs.randint(len(task_data.parents))
        # if len(os.listdir(os.path.join(exp['log_dir'], 'tmp'))) > 2 * exp['population_size']:
        #   time.sleep(1)
        #   continue
        #   index = 0
        parent_id, parent = task_data.parents[index]

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        if parent is None:
            # 0'th iteration: first models still have to be generated
            model = policy.generate_model()
            policy.set_model(model)
        else:
            policy.set_model(parent)
            # todo unmodified or not?
            # elite at idx 0 doesn't have to be evolved (last elem of parents list is an
            # exact copy of the elite, which will be evolved)
            # if index < experiment.num_elites():
            #    policy.evolve_model(task_data.noise_stdev)
            policy.evolve_model(task_data.noise_stdev)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        fitness = policy.rollout(data=task_data.batch_data, config=config)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        return Result(
            worker_id=self.worker_id,
            evaluated_model_id=parent_id,
            evaluated_model=policy.serialized(path=self.offspring_path.format(w=self.worker_id,
                                                                              i=it_id)),
            fitness=np.array([fitness], dtype=np.float32),
            mem_usage=max(mem_usages)
        )


def start_and_run_worker(i, master_redis_cfg, relay_redis_cfg):
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )  # stream=sys.stdout)

    ga_worker = GAWorker(master_redis_cfg, relay_redis_cfg)
    ga_worker.run_worker()
