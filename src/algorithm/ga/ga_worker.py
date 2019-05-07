import copy
import gc
import logging
import os

import psutil
import time

import numpy as np
import torch
# from memory_profiler import profile
from algorithm.ga.ga_master import GATask
from algorithm.tools.experiment import GAExperiment
from dist import WorkerClient
from algorithm.policies import Policy
from algorithm.tools.setup import Config, setup_worker
from algorithm.tools.utils import GAResult, mkdir_p


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
        self.policy: Policy = setup_tuple[1]
        self.experiment: GAExperiment = setup_tuple[2]

        self.placeholder = torch.FloatTensor(1)

    # @profile(stream=open('output/memory_profile_worker.txt', 'w+'))
    # @profile
    def run_worker(self):
        logger = logging.getLogger(__name__)

        logger.info('run_worker: {}'.format(locals()))
        torch.set_grad_enabled(False)
        torch.set_num_threads(0)

        # self.exp = self.worker.get_experiment()
        exp, config, experiment, rs, worker, policy = \
            self.exp, self.config, self.experiment, self.rs, self.worker, self.policy

        _it_id = 0

        while True:

            _it_id += 1
            torch.set_grad_enabled(False)
            time.sleep(0.01)
            mem_usages = []

            eval_or_evolve = rs.rand()
            if len(os.listdir(self.offspring_dir)) > 2 * exp['population_size']:
                time.sleep(30)
                eval_or_evolve = config.eval_prob - 0.05

            task_id, task_data = worker.get_current_task()
            task_tstart = time.time()
            assert isinstance(task_id, int) and isinstance(task_data, GATask)

            # policy: Policy = PolicyFactory.create(dataset=SuppDataset(exp['dataset']),
            #                                       mode=exp['mode'], exp=exp)
            # policy.init_model(policy.generate_model())

            # policy.calculate_all_sensitivities(task_data, self.experiment.trainloader,
            #                                    self.offspring_dir, self.experiment.orig_batch_size())

            # break

            if eval_or_evolve < config.eval_prob:
                logger.info('EVAL RUN')
                try:
                    result = self.accuracy(policy, task_data)
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
            self.write_alive_tensors()

    def accuracy(self, policy, task_data):

        mem_usages = [psutil.Process(os.getpid()).memory_info().rss]

        index = self.rs.randint(len(task_data.elites))
        # index = os.getpid() % len(task_data.elites)
        cand_id, cand = task_data.elites[index]
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        policy.set_model(cand)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        score = policy.accuracy_on(self.experiment.valloader, self.config, self.eval_dir)
        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        # del task_data, cand, score
        return GAResult(
            worker_id=self.worker_id,
            score=score,
            evaluated_cand_id=cand_id,
            evaluated_cand=cand,
            mem_usage=max(mem_usages)
        )

    def fitness(self, it_id, policy, task_data, task_id):

        # todo, see SC paper: during training: picking ARGMAX vs SAMPLE! now argmax?

        batch_data = copy.deepcopy(task_data.batch_data)

        if self.config.selection == 'tournament':
            tournament = self.rs.randint(0, len(task_data.parents), self.config.tournament_size)
            # parents are sorted highest fitness first so individual winning the tournament
            # is simply the lowest index sampled
            index = tournament.min()
            parent_id, parent = task_data.parents[index]
        else:
            # truncation selection, select one uniformly
            index = self.rs.randint(len(task_data.parents))
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
            if index > self.experiment.num_elites():
                policy.calc_sensitivity(task_id, parent_id, batch_data, self.experiment.orig_batch_size(),
                                        self.offspring_dir)
                policy.evolve_model(task_data.noise_stdev)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        fitness = policy.rollout(placeholder=self.placeholder,
                                 data=batch_data, config=self.config)

        mem_usages.append(psutil.Process(os.getpid()).memory_info().rss)

        return GAResult(
            worker_id=self.worker_id,
            evaluated_model_id=parent_id,
            evaluated_model=policy.serialized(path=self.offspring_path.format(w=self.worker_id,
                                                                              i=it_id)),
            fitness=np.array([fitness], dtype=np.float),
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

    ga_worker = GAWorker(master_redis_cfg, relay_redis_cfg)
    ga_worker.run_worker()
