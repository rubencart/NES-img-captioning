import copy
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch

from dist import WorkerClient
from policies import Policy, CompressedModel
from setup import setup, Config
from utils import GATask, Result

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
        setup_tuple = setup(exp)
        config: Config = setup_tuple[0]
        policy: Policy = setup_tuple[1]

        rs = np.random.RandomState()
        worker_id = rs.randint(2 ** 31)
        # todo worker_id random int???? what if two get the same?

        while True:

            gc.collect()
            time.sleep(0.01)
            mem_usages = []

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, GATask)

            if rs.rand() < config.eval_prob:
                model = copy.deepcopy(task_data.elite)

                mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                accuracy = policy.accuracy_on(data=task_data.val_data, model=model, mode=exp['mode'])

                mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                worker.push_result(task_id, Result(
                    worker_id=worker_id,
                    eval_return=accuracy,
                    mem_usage=max(mem_usages)
                ))
            else:
                # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?

                index = rs.randint(len(task_data.parents))
                parent_id, compressed_parent = task_data.parents[index]

                if compressed_parent is None:
                    # 0'th iteration: first models still have to be generated
                    model = CompressedModel()
                else:
                    model = copy.deepcopy(compressed_parent)
                    # elite doesn't have to be evolved
                    # todo THIS IS WRONG!!!!!
                    if index != 0:
                        model.evolve(config.noise_stdev)
                        assert isinstance(model, CompressedModel)

                mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                # policy.set_model(model)

                mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                fitness = policy.rollout(data=task_data.batch_data, model=model, mode=exp['mode'])

                mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

                worker.push_result(task_id, Result(
                    worker_id=worker_id,
                    evaluated_model_id=parent_id,
                    evaluated_model=model,
                    fitness=np.array([fitness], dtype=np.float32),
                    mem_usage=max(mem_usages)
                ))

            # del task_data
