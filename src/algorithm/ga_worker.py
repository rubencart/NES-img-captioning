
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch

from dist import WorkerClient
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import GATask, Result, mkdir_p

logger = logging.getLogger(__name__)


class GAWorker(object):
    # @profile_exp(stream=open('profile_exp/memory_profile_worker.log', 'w+'))
    def run_worker(self, master_redis_cfg, relay_redis_cfg):
        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        # redis client
        worker = WorkerClient(master_redis_cfg, relay_redis_cfg)

        exp = worker.get_experiment()
        assert exp['mode'] in ['seeds', 'nets'], '{}'.format(exp['mode'])

        setup_tuple = setup_worker(exp)
        config: Config = setup_tuple[0]
        policy: Policy = setup_tuple[1]
        # policy = None
        # experiment: Experiment = setup_tuple[6]

        rs = np.random.RandomState()
        worker_id = rs.randint(2 ** 31)
        # todo worker_id random int???? what if two get the same?

        offspring_dir = os.path.join(exp['log_dir'], 'models', 'offspring')
        mkdir_p(offspring_dir)
        offspring_path = os.path.join(offspring_dir, '{w}_{i}_offspring_params.pth')

        _it_id = 0

        # logging.info('going into while true loop')
        while True:

            _it_id += 1
            gc.collect()
            time.sleep(0.01)
            mem_usages = []

            # deadlocks!!!! if elite hasn't been evaluated when nb files > 2 pop
            if len(os.listdir(offspring_dir)) > 2 * exp['population_size']:
                time.sleep(10)
                break

            # logging.info('Getting task')
            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, GATask)
            # logging.info('Got task')

            # parent_paths = task_data.parents
            # elite_path = task_data.elite

            # input('...')
            # if not policy:
            #     policy = task_data.policy

            if rs.rand() < config.eval_prob:
                # logging.info('EVAL RUN')
                try:
                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)
                    # logging.info('Elite: %s', task_data.elite)

                    # setup_tuple = setup_worker(exp)
                    # config: Config = setup_tuple[0]
                    # policy: Policy = setup_tuple[1]

                    policy.init_model(policy.generate_model())
                    policy.set_model(task_data.elite)

                    # logging.info('Elite set in policy')
                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                    # logging.info('Calculating acc')
                    accuracy = policy.accuracy_on(data=task_data.val_data)
                    # accuracy = policy.rollout(data=task_data.val_data)
                    # logging.info('Accuracy: %d', round(accuracy, 2))

                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                    # logging.info('EVAL pushing result')
                    worker.push_result(task_id, Result(
                        worker_id=worker_id,
                        eval_return=accuracy,
                        mem_usage=max(mem_usages)
                    ))
                except FileNotFoundError as e:
                    # logging.error(e)
                    pass

                except Exception as e:
                    raise Exception

                # del model
            else:
                # logging.info('EVOLVE RUN')
                try:
                    # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?

                    index = rs.randint(len(task_data.parents))
                    # if len(os.listdir(os.path.join(exp['log_dir'], 'tmp'))) > 2 * exp['population_size']:
                    #   time.sleep(1)
                    #   break
                    #   index = 0
                    parent_id, parent = task_data.parents[index]

                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                    if parent is None:
                        # 0'th iteration: first models still have to be generated
                        model = policy.generate_model()
                        policy.set_model(model)
                    else:
                        policy.set_model(parent)
                        # elite at idx 0 doesn't have to be evolved (last elem of parents list is an
                        # exact copy of the elite, which will be evolved)
                        if index != 0:
                            policy.evolve_model(task_data.noise_stdev)

                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                    fitness = policy.rollout(data=task_data.batch_data)

                    mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                    # logging.info('EVOLVE pushing result')
                    worker.push_result(task_id, Result(
                        worker_id=worker_id,
                        evaluated_model_id=parent_id,
                        # evaluated_model=policy.get_model(),
                        evaluated_model=policy.serialized(path=offspring_path.format(w=worker_id,
                                                                                     i=_it_id)),
                        fitness=np.array([fitness], dtype=np.float32),
                        mem_usage=max(mem_usages)
                    ))
                except FileNotFoundError as e:
                    # logging.error(e)
                    pass

                except Exception as e:
                    raise Exception

                # del model

            # del task_data
            gc.collect()
            # logging.info('going out of while true loop')


def start_and_run_worker(i, master_redis_cfg, relay_redis_cfg):
    ga_worker = GAWorker()
    ga_worker.run_worker(master_redis_cfg, relay_redis_cfg)
