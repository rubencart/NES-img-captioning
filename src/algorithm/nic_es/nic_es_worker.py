import copy
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch
# from memory_profiler import profile
from algorithm.nic_es.nic_es_master import ESTask, ESResult
from algorithm.nic_es.experiment import ESExperiment
from dist import WorkerClient
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import mkdir_p

logger = logging.getLogger(__name__)


class ESWorker(object):

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.rs = np.random.RandomState()
        self.worker_id = self.rs.randint(2 ** 31)
        self.offspring_dir = ''
        self.offspring_path = ''
        self.eval_dir = ''

        # redis client
        self.worker = WorkerClient(master_redis_cfg, relay_redis_cfg)
        self.exp = self.worker.get_experiment()

        self.offspring_dir = os.path.join(self.exp['log_dir'], 'models', 'offspring')
        mkdir_p(self.offspring_dir)
        self.offspring_path = os.path.join(self.offspring_dir, '{w}_{i}_offspring_params.pth')

        self.eval_dir = os.path.join(self.exp['log_dir'], 'eval_{}'.format(os.getpid()))
        mkdir_p(self.eval_dir)

        setup_tuple = setup_worker(self.exp)
        self.config: Config = setup_tuple[0]
        self.policy: Policy = setup_tuple[1]
        self.experiment: ESExperiment = setup_tuple[2]

        self.placeholder = torch.FloatTensor(1)

    # @profile(stream=open('output/memory_profile_worker.txt', 'w+'))
    # @profile
    def run_worker(self):
        # logger = logging.getLogger(__name__)

        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        # self.exp = self.worker.get_experiment()
        exp, config, experiment, rs, worker, policy = \
            self.exp, self.config, self.experiment, self.rs, self.worker, self.policy

        it_id = 0

        while True:

            it_id += 1
            time.sleep(0.01)
            mem_usages = []

            eval_or_evolve = rs.rand()
            # this is here to make workers stop when they have already generated 3 times the population size
            # within one generation because master is running behind on tasks or waiting for evaluations
            # (as opposed to evolutions)
            if len(os.listdir(self.offspring_dir)) > 3 * self.experiment.nb_offspring():
                time.sleep(2)
                logging.warning('Too many files in offspring dir')
                if eval_or_evolve >= config.eval_prob:
                    continue

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, ESTask)

            # uncomment to calculate sensitivities for current policy on entire training set
            # policy.calculate_all_sensitivities(task_data, self.experiment.trainloader,
            #                                    self.offspring_dir, self.experiment.orig_batch_size())
            # break

            self.policy.set_ref_batch(task_data.ref_batch)

            if eval_or_evolve < config.eval_prob:
                logger.info('EVAL RUN')
                try:
                    result = self.accuracy(task_id, policy, task_data)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logger.error(e)

            else:
                # logging.info('EVOLVE RUN')
                try:

                    result = self.fitness(it_id, policy, task_data, task_id)
                    worker.push_result(task_id, result)

                except FileNotFoundError as e:
                    logging.error(e)

            del task_data
            gc.collect()
            # self.write_alive_tensors()

    def accuracy(self, task_id, policy, task_data):
        """
        Compute accuracy of one of the elite candidates of last generation on validation set
        """
        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        index = self.rs.randint(len(task_data.elites))
        cand_id, cand = task_data.elites[index]
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        policy.set_model(cand)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        score = policy.accuracy_on(self.experiment.valloader, self.config, self.eval_dir)
        logger.info('Iteration: {}, Val score: {}'.format(task_id, score))

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        return ESResult(
            worker_id=self.worker_id,
            score=score,
            evaluated_cand_id=cand_id,
            evaluated_cand=cand,
            mem_usage=max(mem_usages)
        )

    def fitness(self, it_id, policy, task_data, task_id):
        """
        Pick a parent, mutate it and compute the fitness of the resulting individual on
            the current minibatch (that is published by the master)
        """
        batch_data = copy.deepcopy(task_data.batch_data)

        # parent selection
        if self.experiment.selection() == 'tournament':
            tournament = self.rs.choice(len(task_data.parents),
                                        min(len(task_data.parents), self.experiment.tournament_size()),
                                        replace=False)
            logging.info(tournament)
            # parents are sorted highest fitness first so individual winning the tournament
            # is simply the lowest index sampled
            index = tournament.min()
            parent_id, parent = task_data.parents[index]
        else:
            # uniform random selection
            index = self.rs.randint(len(task_data.parents))
            parent_id, parent = task_data.parents[index]

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        if parent is None:
            # 0'th iteration: first models still have to be generated
            model = policy.generate_model()
            policy.set_model(model)
        else:
            # calculate sensitivity if necessary and evolve model
            policy.set_model(parent)
            policy.calc_sensitivity(task_id, parent_id, batch_data, self.experiment.orig_batch_size(),
                                    self.offspring_dir)
            policy.evolve_model(task_data.noise_stdev)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # compute fitness
        fitness = policy.rollout(placeholder=self.placeholder,
                                 data=batch_data, config=self.config)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        return ESResult(
            worker_id=self.worker_id,
            evaluated_model_id=parent_id,
            # this saves the params to a file on disk
            evaluated_model=policy.serialized(path=self.offspring_path.format(w=self.worker_id,
                                                                              i=it_id)),
            fitness=np.array([fitness], dtype=np.float),
            mem_usage=max(mem_usages)
        )


def start_and_run_worker(i, master_redis_cfg, relay_redis_cfg):
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
    )

    es_worker = ESWorker(master_redis_cfg, relay_redis_cfg)
    es_worker.run_worker()
