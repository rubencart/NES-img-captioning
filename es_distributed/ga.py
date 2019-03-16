import copy
import logging
import time
from collections import namedtuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from es_distributed.dist import MasterClient, WorkerClient
from es_distributed.policies import CompressedModel


logger = logging.getLogger(__name__)

ga_task_fields = ['params', 'population', 'ob_mean', 'ob_std', 'timestep_limit', 'batch_data', 'parents']
GATask = namedtuple('GATask', field_names=ga_task_fields, defaults=(None,) * len(ga_task_fields))

config_fields = [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq', 'num_dataloader_workers',
    'return_proc_mode', 'episode_cutoff_mode', 'batch_size'
]
Config = namedtuple('Config', field_names=config_fields, defaults=(None,) * len(config_fields))
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])

result_fields = [
    'worker_id', 'evaluated_model_id', 'fitness', 'evaluated_model',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
]
Result = namedtuple('Result', field_names=result_fields, defaults=(None,) * len(result_fields))


# def setup(exp, single_threaded):
def setup(exp):
    # import gym
    # gym.undo_logger_setup()
    # from . import policies, tf_util
    #
    # config = Config(**exp['config'])
    # env = gym.make(exp['env_id'])
    # if exp['env_id'].endswith('NoFrameskip-v4'):
    #     from .atari_wrappers import wrap_deepmind
    #     env = wrap_deepmind(env)
    # sess = make_session(single_threaded=single_threaded)
    # policy = getattr(policies, exp['policy']['type'])
    # (env.observation_space, env.action_space, **exp['policy']['args'])
    # tf_util.initialize()
    # return config, env, sess, policy

    # todo
    from . import policies
    config = Config(**exp['config'])
    policy = getattr(policies, exp['policy']['type'])(**exp['policy']['args'])
    return config, policy


# def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob):
#     if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
#         rollout_rews, rollout_len, obs = policy.rollout(
#             env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs)
#         task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
#     else:
#         rollout_rews, rollout_len, rollout_nov = policy.rollout(env, timestep_limit=timestep_limit, random_stream=rs)
#     return rollout_rews, rollout_len


# def run_master(master_redis_cfg, log_dir, exp):
def run_master(master_redis_cfg, log_dir, exp, plot):
    logger.info('run_master: {}'.format(locals()))
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)

    # config, env, sess, policy = setup(exp, single_threaded=False)
    config, policy = setup(exp)

    # redis master
    master = MasterClient(master_redis_cfg)

    # noise = SharedNoiseTable()
    rs = np.random.RandomState()

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # MNIST has 60k training images
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    # batch_size = config.batch_size if config.batch_size else 256
    batch_size = config.batch_size

    # todo num_workers?
    num_workers = config.num_dataloader_workers

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=num_workers)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.set_grad_enabled(False)

    # todo cutoff mode: necessary? could be like starting with shorter (parts of) sentences,
    # but lets assume this is not necessary since in SC paper they don't seem to have problems
    # defines tslimit, incr_tslimit_threshold, tslimit_incr_ratio, adaptive_tslimit

    # if isinstance(config.episode_cutoff_mode, int):
    #     tslimit, incr_tslimit_threshold, tslimit_incr_ratio = config.episode_cutoff_mode, None, None
    #     adaptive_tslimit = False
    # elif config.episode_cutoff_mode.startswith('adaptive:'):
    #     _, args = config.episode_cutoff_mode.split(':')
    #     arg0, arg1, arg2 = args.split(',')
    #     tslimit, incr_tslimit_threshold, tslimit_incr_ratio = int(arg0), float(arg1), float(arg2)
    #     adaptive_tslimit = True
    #     logger.info(
    #         'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'
    #         .format(
    #             tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio))
    # elif config.episode_cutoff_mode == 'env_default':
    #     tslimit, incr_tslimit_threshold, tslimit_incr_ratio = None, None, None
    #     adaptive_tslimit = False
    # else:
    #     raise NotImplementedError(config.episode_cutoff_mode)

    # episodes_so_far = 0
    # timesteps_so_far = 0
    tstart = time.time()

    # this puts up a redis key, value pair with the experiment
    master.declare_experiment(exp)

    best_score = float('-inf')
    population_size = exp['population_size']
    truncation = exp['truncation']

    num_elites = exp['num_elites']
    # population_score = np.array([])

    # fill population
    # population = []
    # todo continue_from
    # population = [(model_id, CompressedModel()) for model_id in range(population_size)]
    parents = [(model_id, None) for model_id in range(truncation)]
    # parent_dict = {model_id: (model_id, model) for (model_id, model) in parents}

    score_stats = [[], [], []]
    time_stats = []

    # todo max epochs?
    while True:
        # todo max generations
        # todo check how many times training set has been gone through
        for i, batch_data in enumerate(trainloader, 0):

            step_tstart = time.time()
            # theta = policy.get_trainable_flat()
            # assert theta.dtype == np.float32

            # todo what is ob_stat?
            # if policy.needs_ob_stat:
            #     ob_stat = RunningStat(
            #         env.observation_space.shape,
            #         eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
            #     )

            curr_task_id = master.declare_task(GATask(
                # params=theta,
                # population=population,
                parents=parents,
                batch_data=batch_data,
                ob_mean=None,  # ob_stat.mean if policy.needs_ob_stat else None,
                ob_std=None,  # ob_stat.std if policy.needs_ob_stat else None,
                timestep_limit=None  # tslimit
            ))

            tlogger.log('********** Iteration {} **********'.format(curr_task_id))
            logger.info('Searching {nb} params for NW'.format(nb=policy.nb_learnable_params()))

            # Pop off results for the current task
            # curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
            curr_task_results, eval_rets, worker_ids = [], [], []
            # num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
            num_results_skipped = 0

            # models_to_evaluate = sorted([i for i, _ in population])
            nb_models_to_evaluate = population_size
            elite_evaluated = False
            # evaluated_models = []

            # while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
            # todo population
            while nb_models_to_evaluate > 0 or not elite_evaluated:
                # if nb_models_to_evaluate % 50 == 0 and nb_models_to_evaluate != population_size:
                #     logger.info('Nb of models left to evaluate: {nb}. Elite evaluated: {el}'
                #                 .format(nb=nb_models_to_evaluate, el=elite_evaluated))

                # if nb_models_to_evaluate <= 0 and not elite_evaluated:
                #     logger.warning('Only the elite still has to be evaluated')

                # Wait for a result
                task_id, result = master.pop_result()
                assert isinstance(task_id, int) and isinstance(result, Result)
                assert (result.eval_return is None) == (result.eval_length is None)

                worker_ids.append(result.worker_id)

                # todo difference between eval and not eval? --> paper
                if result.eval_length is not None:
                    # This was an eval job
                    # episodes_so_far += 1
                    # timesteps_so_far += result.eval_length
                    # Store the result only for current tasks
                    if task_id == curr_task_id:
                        eval_rets.append(result.eval_return)
                        # eval_lens.append(result.eval_length)
                else:
                    # assert result.returns_n2.dtype == np.float32
                    assert result.fitness.dtype == np.float32

                    # Store results only for current tasks
                    if task_id == curr_task_id:
                        # Update counts
                        # result_num_eps = result.lengths_n2.size
                        # result_num_timesteps = result.lengths_n2.sum()
                        # episodes_so_far += result_num_eps
                        # timesteps_so_far += result_num_timesteps

                        # curr_task_results.append(result)
                        # num_episodes_popped += result_num_eps
                        # num_timesteps_popped += result_num_timesteps

                        # if result.evaluated_model_id in models_to_evaluate:
                        # models_to_evaluate.remove(result.evaluated_model_id)
                        nb_models_to_evaluate -= 1
                        curr_task_results.append(result)
                        if result.evaluated_model_id == 0:
                            elite_evaluated = True

                        # Update ob stats
                        # if policy.needs_ob_stat and result.ob_count > 0:
                        #     ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        #     ob_count_this_batch += result.ob_count

                    else:
                        num_results_skipped += 1

            # Compute skip fraction
            frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
            if num_results_skipped > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    num_results_skipped, 100. * frac_results_skipped))

            # Assemble results + elite
            # noise_inds_n = list(population[:num_elites])
            # returns_n2 = list(population_score[:num_elites])
            #
            # for r in curr_task_results:
            #     noise_inds_n.extend(r.noise_inds_n)
            #     returns_n2.extend(r.returns_n2)
            #
            # noise_inds_n = np.array(noise_inds_n)
            # returns_n2 = np.array(returns_n2)
            # lengths_n2 = np.array([r.lengths_n2 for r in curr_task_results])

            # scores = np.array(
            #     sorted([result.fitness.item() for result in curr_task_results], reverse=True)
            # )
            scored_models = [(result.evaluated_model_id, result.evaluated_model, result.fitness.item())
                             for result in curr_task_results]
            elite_scored_models = [t for t in scored_models if t[0] == 0]
            other_scored_models = [t for t in scored_models if t[0] != 0]

            best_elite_score = (0, elite_scored_models[0][1], max([score for _, _, score in elite_scored_models]))

            scored_models = [best_elite_score] + other_scored_models

            scored_models.sort(key=lambda x: x[2], reverse=True)
            scores = np.array([fitness for (_, _, fitness) in scored_models])
            # pick parents for next generation, give new index
            parents = [(i, model) for (i, (_, model, _)) in enumerate(scored_models[:truncation])]

            print([(i, round(f, 2)) for (i, _, f) in scored_models[:5]])
            # input('PRESS ENTER')

            score_stats[0].append(scores.min())
            score_stats[1].append(scores.mean())
            score_stats[2].append(scores.max())

            step_tend = time.time()
            time_stats.append(step_tend - step_tstart)
            print('TIME STATSSSSSS ', time_stats)

            # Process returns
            # todo what does argpartition do
            # idx = np.argpartition(returns_n2, (-population_size, -1))[-1:-population_size - 1:-1]
            # population = noise_inds_n[idx]
            # population_score = returns_n2[idx]

            # todo assertions
            # assert len(population) == population_size
            # assert np.max(returns_n2) == population_score[0]

            print('Elite: {} score: {}'.format(scored_models[0], scores[0]))

            #     def get(self, i, dim):
            #         return self.noise[i:i + dim]
            # population[0] is the elite, elite[0] is ...? so this just sets policy to the elite

            # EVOLVE
            # Elitism
            # next_generation = [(0, parents[0])]
            # for idx in range(population_size):
            #     next_generation.append((idx + 1, copy.deepcopy(rs.choice(parents))))
            #     # self.models contains compressed models
            #     next_generation[-1][1].evolve(config.noise_stdev)
            #
            # population = copy.deepcopy(next_generation)
            # population_dict = {model_id: (model_id, model) for (model_id, model) in population}

            # todo adjust to pytorch
            # this evolves the elite of last iteration --> to be able to evaluate it

            # policy.set_trainable_flat(noise.get(population[0][0], policy.num_params))
            # sets gradients to zero again?
            # policy.reinitialize()
            # v = policy.get_trainable_flat()

            # for seed in population[0][1:]:
            #    v += config.noise_stdev * noise.get(seed, policy.num_params)

            # policy.set_trainable_flat(v)

            # Update number of steps to take
            # if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            #     old_tslimit = tslimit
            #     tslimit = int(tslimit_incr_ratio * tslimit)
            #     logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

            tlogger.record_tabular("RewMax", scores.max())
            tlogger.record_tabular("RewMean", scores.mean())
            tlogger.record_tabular("RewMin", scores.min())
            tlogger.record_tabular("RewStd", scores.std())
            # tlogger.record_tabular("EpLenMean", lengths_n2.mean())

            # tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
            # tlogger.record_tabular("EvalEpRewMedian", np.nan if not eval_rets else np.median(eval_rets))
            # tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
            # # tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
            # tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            #         np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))
            # tlogger.record_tabular("EvalEpCount", len(eval_rets))

            # todo norm
            # tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))

            # tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
            # tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
            # tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
            # tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

            num_unique_workers = len(set(worker_ids))
            tlogger.record_tabular("UniqueWorkers", num_unique_workers)
            tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
            tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
            # tlogger.record_tabular("ObCount", ob_count_this_batch)

            tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
            tlogger.record_tabular("TimeElapsed", step_tend - tstart)
            tlogger.dump_tabular()

            # if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
            if config.snapshot_freq != 0:
                import os.path as osp
                # todo filename --> .h5? or .p like torch files
                # todo to log_dir
                filename = 'snapshot_iter{:05d}_rew{}.h5'.format(
                    curr_task_id,
                    np.nan if not eval_rets else int(np.mean(eval_rets))
                )
                assert not osp.exists(filename)
                policy.save(filename)
                tlogger.log('Saved snapshot {}'.format(filename))

            if plot:
                import matplotlib.pyplot as plt

                plt.figure()
                x = np.arange(len(score_stats[1]))
                plt.fill_between(x=x, y1=score_stats[0], y2=score_stats[2], facecolor='blue', alpha=0.3)
                plt.plot(x=x, y=score_stats[1], label='Training loss', color='blue')
                plt.savefig(log_dir + '/loss_plot_{i}.png'.format(i=i))

                plt.figure()
                plt.plot(x=x, y=time_stats, label='Time per generation', color='black')
                plt.savefig(log_dir + '/time_plot_{i}.png'.format(i=i))

    # except KeyboardInterrupt:
    #     pass

    # def plot_acc_loss(mins: list, means: list, maxes: list) -> None:
    #     # plt.xlabel('epochs')
    #     loss_plot, = plt.errorbar(x=np.arange(len(means)), y=means, yerr=(mins, maxes), label='Training loss')
    #     # t_acc_plot, = plt.plot(1 - train_accs, label='Train err')
    #     # v_acc_plot, = plt.plot(1 - val_accs, label='Val err')
    #     # plt.legend(handles=[loss_plot, t_acc_plot, v_acc_plot])
    #     plt.savefig(log_dir + '/loss_plot.png')
    #     # show_plots and plt.draw()

    # plt.errorbar(x=np.arange(len(score_stats[1])), y=score_stats[1], yerr=(score_stats[0], score_stats[2]),
    #              label='Training loss')
    # plt.savefig(log_dir + '/loss_plot.png')


def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))

    torch.set_grad_enabled(False)

    # assert isinstance(noise, SharedNoiseTable)

    # redis client
    worker = WorkerClient(master_redis_cfg, relay_redis_cfg)

    exp = worker.get_experiment()
    # config, env, sess, policy = setup(exp, single_threaded=True)
    config, policy = setup(exp)

    rs = np.random.RandomState()  # todo randomstate vs seed? blogpost
    worker_id = rs.randint(2 ** 31)
    # todo worker_id random int???? what if two get the same?

    # assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, GATask)

        # if policy.needs_ob_stat:
        #     policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        # todo eval_prob
        if False and rs.rand() < config.eval_prob:
            pass
            # Evaluation: noiseless weights and noiseless actions
            # policy.set_trainable_flat(task_data.params)

            # too many values to unpack (expected 2)
            # eval_rews, eval_length, _ = policy.rollout(env)  # eval rollouts don't obey task_data.timestep_limit
            # todo .rollout(img_id, sentence) or (img_id, class_prediction) or sth
            # todo so this is what takes so long? see if we can find python profiler or something to verify this
            # eval_rews, eval_length, _ = policy.rollout()
            # fitness = policy.rollout(task_data.batch_data)

            # eval_return = eval_rews.sum()
            # logger.info('Eval result: task={} fitness={:.3f}'.format(task_id, fitness))

            # this pushes the result of the worker back to the master via redis
            # worker.push_result(task_id, Result(
            #     worker_id=worker_id,
            #     eval_return=fitness,
            # ))
        else:
            # todo what is this? check paper
            # --> I think this is what it's about, eval is just to show progress
            # Rollouts with noise
            # noise_inds, returns, signreturns, lengths = [], [], [], []
            # task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            # while not noise_inds or time.time() - task_tstart < min_task_runtime:
            # while True:
            # if len(task_data.population) > 0:
                # seeds[0] is [a random member of the population (ie a SNT index)]
                # seeds = list(task_data.population[rs.randint(len(task_data.population))]) \
                #         + [noise.sample_index(rs, policy.num_params)]
            # else:
                # this happens only in the first iteration
                # noise.sample_index(rs, np) --> rs.randint(0, len(self.noise) - np + 1)
                # seeds = [noise.sample_index(rs, policy.num_params)]

            # parent_id, compressed_parent = task_data.parents[rs.randint(len(task_data.parents))]
            index = rs.randint(len(task_data.parents))
            # print(index)
            # parent_id, compressed_parent = rs.choice(task_data.parents)
            parent_id, compressed_parent = task_data.parents[index]

            if compressed_parent is None:
                # 0'th iteration: first models still have to be generated
                model = CompressedModel()
            else:
                model = copy.deepcopy(compressed_parent)
                # elite doesn't have to be evolved
                if index != 0:
                    # print(model)
                    model.evolve(config.noise_stdev)
                    assert isinstance(model, CompressedModel)
                    # print(model)

            policy.set_model(model)

            # get(i, dim) --> self.noise[i:i + dim]
            # v = noise.get(seeds[0], policy.num_params)

            # todo torchify
            # policy.set_trainable_flat(v)
            # this normalizes the weights (div by std) and sets the biases to zero
            # policy.reinitialize()
            # v = policy.get_trainable_flat()

            # for seed in seeds[1:]:
            #    v += config.noise_stdev * noise.get(seed, policy.num_params)
            # policy.set_trainable_flat(v)

            # rews_pos, len_pos = rollout_and_update_ob_stat(
            #     policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)
            # rollout_rews, rollout_len, rollout_nov = policy.rollout(env,
            #                               timestep_limit=timestep_limit, random_stream=rs)

            fitness = policy.rollout(task_data.batch_data)

            # noise_inds.append(seeds)
            # returns.append(fitness)

            # noise_inds = seeds.copy()

            # signreturns.append(np.sign(rews_pos).sum())
            # lengths.append(len_pos)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                # noise_inds_n=noise_inds,
                evaluated_model_id=parent_id,
                evaluated_model=model,
                fitness=np.array([fitness], dtype=np.float32)
                # returns_n2=np.array(returns, dtype=np.float32),
                # signreturns_n2=np.array(signreturns, dtype=np.float32),
                # lengths_n2=np.array(lengths, dtype=np.int32),
                # eval_return=None,
                # eval_length=None,
                # ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                # ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                # ob_count=task_ob_stat.count
            ))
