import logging
import os
import pickle
import time

# todo necessary?
# import h5py
import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.layers as layers

# from . import tf_util as U
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from es_distributed.main import mkdir_p

logger = logging.getLogger(__name__)


class TfPolicy:
    # todo args, kwargs
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        # self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)

        # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        # self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        # self._setfromflat = U.SetFromFlat(self.trainable_variables)
        # self._getflat = U.GetFlat(self.trainable_variables)

        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        # for v in self.trainable_variables:
        #     shp = v.get_shape().as_list()
        #     logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        # for v in self.all_variables:
        #     shp = v.get_shape().as_list()
        #     logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        # placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        # self.set_all_vars = U.function(
        #     inputs=placeholders,
        #     outputs=[],
        #     updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        # )
        raise NotImplementedError

    # todo
    def _make_net(self, o, is_ref):
        raise NotImplementedError

    def reinitialize(self):
        # for v in self.trainable_variables:
        #     v.reinitialize.eval()
        raise NotImplementedError

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        # todo used to save model & params? see ga.py
        # assert filename.endswith('.h5')
        # with h5py.File(filename, 'w', libver='latest') as f:
        #     for v in self.all_variables:
        #         f[v.name] = v.eval()
        #     # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
        #     # (like Gym spaces or numpy arrays)
        #     f.attrs['name'] = type(self).__name__
        #     f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))
        raise NotImplementedError

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        # todo used to load existing model
        # with h5py.File(filename, 'r') as f:
        #     args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
        #     if extra_kwargs:
        #         kwargs.update(extra_kwargs)
        #     policy = cls(*args, **kwargs)
        #     policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        # return policy
        raise NotImplementedError

    # === Rollouts/training ===

    # def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
    def rollout(self, env=None, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        # todo returns reward & time used?
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        # env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        # timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        # rews = []
        # t = 0
        # if save_obs:
        #     obs = []
        # ob = env.reset()
        # for _ in range(timestep_limit):
        #     ac = self.act(ob[None], random_stream=random_stream)[0]
        #     if save_obs:
        #         obs.append(ob)
        #     ob, rew, done, _ = env.step(ac)
        #     rews.append(rew)
        #     t += 1
        #     if render:
        #         env.render()
        #     if done:
        #         break
        # rews = np.array(rews, dtype=np.float32)
        # if save_obs:
        #     return rews, t, np.array(obs)
        # return rews, t
        raise NotImplementedError

    # def act(self, ob, random_stream=None):
    #     raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    # @property
    # def needs_ob_stat(self):
    #     raise NotImplementedError
    #
    # def set_ob_stat(self, ob_mean, ob_std):
    #     raise NotImplementedError


class Policy:

    def __init__(self):
        # todo adjust to pytorch
        # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        # self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self.policy_net = None

    def rollout(self, data):
        assert self.policy_net is not None, 'set model first!'

        inputs, labels = data
        outputs = self.policy_net(inputs)

        # todo for now use cross entropy loss as fitness
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        return -loss.item()

    def accuracy_on(self, data):
        inputs, labels = data
        outputs = self.policy_net(inputs)

        prediction = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        accuracy = float(correct) / labels.size()[0]
        return accuracy

    def save(self, path, filename):
        # todo check self-critical --> also save iteration,... not only params?
        mkdir_p(path)
        assert not os.path.exists(os.path.join(path, filename))
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))

    def load(self, filename):
        pass

    def nb_learnable_params(self):
        pass

    def set_model(self, compressed_model):
        pass

    def reinitialize_params(self):
        # todo rescale weights manually (can we use weight norm?)
        pass

    def parameter_vector(self):
        assert self.policy_net is not None, 'set model first!'
        return nn.utils.parameters_to_vector(self.policy_net.parameters())


class Cifar10Policy(Policy):
    def __init__(self, *args):
        super(Cifar10Policy, self).__init__()
        # self.model = Cifar10Classifier(random_state())
        # self.model = None

    def set_model(self, compressed_model):
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=Cifar10Net)
        assert isinstance(uncompressed_model, PolicyNet)
        self.policy_net = uncompressed_model

    def nb_learnable_params(self):
        torch.set_grad_enabled(True)
        model = Cifar10Net(random_state())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.set_grad_enabled(False)
        return count


class MnistPolicy(Policy):
    def __init__(self, *args):
        super(MnistPolicy, self).__init__()
        # self.model = Cifar10Classifier(random_state())
        # self.model = None

    def set_model(self, compressed_model):
        assert isinstance(compressed_model, CompressedModel)
        # model: compressed model
        uncompressed_model = compressed_model.uncompress(to_class_name=MnistNet)
        assert isinstance(uncompressed_model, PolicyNet)
        self.policy_net = uncompressed_model

    def nb_learnable_params(self):
        torch.set_grad_enabled(True)
        model = MnistNet(random_state())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.set_grad_enabled(False)
        return count


class PolicyNet(nn.Module):
    def __init__(self, rng_state):
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        torch.manual_seed(rng_state)
        self.evolve_states = []
        self.add_tensors = {}

    def _initialize_params(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                # todo kaiming normal or:
                # We use Xavier initialization (Glorot & Bengio, 2010) as our policy initialization
                # function φ where all bias weights are set to zero, and connection weights are drawn
                # from a standard normal distribution with variance 1/Nin, where Nin is the number of
                # incoming connections to a neuron
                # nn.init.kaiming_normal_(tensor)
                nn.init.xavier_normal_(tensor)
            else:
                tensor.data.zero_()

    def evolve(self, sigma, rng_state):
        # Evolve params 1 step
        # rng_state = int
        torch.manual_seed(rng_state)
        self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            # fill to_add elements sampled from normal distr
            to_add.normal_(mean=0.0, std=sigma)
            tensor.data.add_(to_add)

    def compress(self):
        return CompressedModel(self.rng_state, self.evolve_states)

    def forward(self, x):
        pass


class Cifar10Net(PolicyNet):
    def __init__(self, rng_state):
        super(Cifar10Net, self).__init__(rng_state)

        # 60K params
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 291466 params
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #
        # self.fc2 = nn.Linear(128, 32)
        # self.fc3 = nn.Linear(32, 10)
        # self.fc4 = nn.Linear(8, 2)

        self._initialize_params()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # def _conv_conv_pool(conv1, conv2, pool, x):
        #     x = F.relu(conv1(x))
        #     x = F.relu(conv2(x))
        #     return pool(x)
        #
        # x = _conv_conv_pool(self.conv1, self.conv2, self.pool, x)
        # x = _conv_conv_pool(self.conv3, self.conv4, self.pool, x)
        # x = _conv_conv_pool(self.conv5, self.conv6, self.avg_pool, x)
        #
        # x = x.view(-1, 128)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # return x


class BlockSlidesNet32(nn.Module):

    def __init__(self):
        super(BlockSlidesNet32, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.bn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # self.avg_pool = nn.AvgPool2d(8, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 2)

        # self.bn_fc2 = nn.BatchNorm1d(32)
        # self.bn_fc3 = nn.BatchNorm1d(8)

    def forward(self, x):
        def _conv_conv_pool(conv1, conv2, pool, x):
            x = F.relu(conv1(x))
            x = F.relu(conv2(x))
            return pool(x)

        x = _conv_conv_pool(self.conv1, self.conv2, self.pool, x)
        x = _conv_conv_pool(self.conv3, self.conv4, self.pool, x)
        x = _conv_conv_pool(self.conv5, self.conv6, self.avg_pool, x)

        x = x.view(-1, 128)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class MnistNet(PolicyNet):
    def __init__(self, rng_state):
        super(MnistNet, self).__init__(rng_state)

        # todo compare fitness incr rate with and without weight norm + time per generation
        # self.conv1 = nn.utils.weight_norm(nn.Conv2d(1, 10, 5, 1))
        # self.conv2 = nn.utils.weight_norm(nn.Conv2d(10, 20, 5, 1))
        # self.fc1 = nn.utils.weight_norm(nn.Linear(4*4*20, 100))
        # self.fc2 = nn.utils.weight_norm(nn.Linear(100, 10))

        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(4*4*20, 100)
        self.fc2 = nn.Linear(100, 10)

        self._initialize_params()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CompressedModel:
    def __init__(self, start_rng: float = None, other_rng: list = None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []

    def evolve(self, sigma, rng_state=None):
        # evolve params 1 step
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))

    def uncompress(self, to_class_name=MnistNet):
        # evolve through all steps
        start_rng, other_rng = self.start_rng, self.other_rng
        m = to_class_name(start_rng)
        for sigma, rng in other_rng:
            m.evolve(sigma, rng)
        return m

    def __str__(self):
        result = '[( {start_rng} )'.format(start_rng=self.start_rng)
        for _, rng in self.other_rng:
            result += '( {rng} )'.format(rng=rng)
        return result + ']'


def random_state():
    rs = np.random.RandomState()
    return rs.randint(0, 2 ** 31 - 1)


# AS EXAMPLE
class GAAtariPolicy(TfPolicy):
    def _initialize(self, ob_space, ac_space, nonlin_type, ac_init_std=0.1):
        self.ob_space_shape = ob_space.shape
        self.ac_space = ac_space
        self.ac_init_std = ac_init_std
        self.num_actions = self.ac_space.n
        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))

            a = self._make_net(o)
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        x = o
        x = self.nonlin(U.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4, std=1.0))
        x = self.nonlin(U.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2, std=1.0))

        x = U.flattenallbut0(x)
        x = self.nonlin(U.dense(x, 256, 'fc', U.normc_initializer(1.0), std=1.0))

        a = U.dense(x, self.num_actions, 'out', U.normc_initializer(self.ac_init_std), std=self.ac_init_std)

        return tf.argmax(a, 1)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    # Dont add random noise since action space is discrete
    def act(self, train_vars, random_stream=None):
        return self._act(train_vars)

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, worker_stats=None,
                policy_seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = [];
        novelty_vector = []
        rollout_details = {}
        t = 0

        if save_obs:
            obs = []

        if policy_seed:
            env.seed(policy_seed)
            np.random.seed(policy_seed)
            if random_stream:
                random_stream.seed(policy_seed)

        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act(ob[None], random_stream=random_stream)[0]

            if save_obs:
                obs.append(ob)
            ob, rew, done, info = env.step(ac)
            rews.append(rew)

            t += 1
            if render:
                env.render()
            if done:
                break

        # Copy over final positions to the max timesteps
        rews = np.array(rews, dtype=np.float32)
        novelty_vector = env.unwrapped._get_ram()  # extracts RAM state information
        if save_obs:
            return rews, t, np.array(obs), np.array(novelty_vector)
        return rews, t, np.array(novelty_vector)


# class MujocoPolicy(Policy):
#
#     def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type):
#         self.ac_space = ac_space
#         self.ac_bins = ac_bins
#         self.ac_noise_std = ac_noise_std
#         self.hidden_dims = hidden_dims
#         self.connection_type = connection_type
#
#         assert len(ob_space.shape) == len(self.ac_space.shape) == 1
#         assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
#             'Action bounds required'
#
#         self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]
#
#         with tf.variable_scope(type(self).__name__) as scope:
#             # Observation normalization
#             ob_mean = tf.get_variable(
#                 'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
#             ob_std = tf.get_variable(
#                 'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
#             in_mean = tf.placeholder(tf.float32, ob_space.shape)
#             in_std = tf.placeholder(tf.float32, ob_space.shape)
#             self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
#                 tf.assign(ob_mean, in_mean),
#                 tf.assign(ob_std, in_std),
#             ])
#
#             # Policy network
#             o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
#             a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
#             self._act = U.function([o], a)
#         return scope
#
#     def _make_net(self, o):
#         # Process observation
#         if self.connection_type == 'ff':
#             x = o
#             for ilayer, hd in enumerate(self.hidden_dims):
#                 x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
#         else:
#             raise NotImplementedError(self.connection_type)
#
#         # Map to action
#         adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
#         assert isinstance(self.ac_bins, str)
#         ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')
#
#         if ac_bin_mode == 'uniform':
#             # Uniformly spaced bins, from ac_space.low to ac_space.high
#             num_ac_bins = int(ac_bin_arg)
#             aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
#             ac_range_1a = (ahigh - alow)[None, :]
#             a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
#
#         elif ac_bin_mode == 'custom':
#             # Custom bins specified as a list of values from -1 to 1
#             # The bins are rescaled to ac_space.low to ac_space.high
#             acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
#             logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
#             assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
#             acvals_ak = (
#                     (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
#                     + alow[:, None]
#             )
#
#             aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
#             a = tf.gather_nd(
#                 acvals_ak,
#                 tf.concat(2, [
#                     tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
#                     tf.expand_dims(aidx_na, -1)
#                 ])  # (n,a,2)
#             )  # (n,a)
#         elif ac_bin_mode == 'continuous':
#             a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
#         else:
#             raise NotImplementedError(ac_bin_mode)
#
#         return a
#
#     def act(self, ob, random_stream=None):
#         a = self._act(ob)
#         if random_stream is not None and self.ac_noise_std != 0:
#             a += random_stream.randn(*a.shape) * self.ac_noise_std
#         return a
#
#     @property
#     def needs_ob_stat(self):
#         return True
#
#     @property
#     def needs_ref_batch(self):
#         return False
#
#     def set_ob_stat(self, ob_mean, ob_std):
#         self._set_ob_mean_std(ob_mean, ob_std)
#
#     def initialize_from(self, filename, ob_stat=None):
#         """
#         Initializes weights from another policy, which must have the same architecture (variable names),
#         but the weight arrays can be smaller than the current policy.
#         """
#         with h5py.File(filename, 'r') as f:
#             f_var_names = []
#             f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
#             assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'
#
#             init_vals = []
#             for v in self.all_variables:
#                 shp = v.get_shape().as_list()
#                 f_shp = f[v.name].shape
#                 assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
#                     'This policy must have more weights than the policy to load'
#                 init_val = v.eval()
#                 # ob_mean and ob_std are initialized with nan, so set them manually
#                 if 'ob_mean' in v.name:
#                     init_val[:] = 0
#                     init_mean = init_val
#                 elif 'ob_std' in v.name:
#                     init_val[:] = 0.001
#                     init_std = init_val
#                 # Fill in subarray from the loaded policy
#                 init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
#                 init_vals.append(init_val)
#             self.set_all_vars(*init_vals)
#
#         if ob_stat is not None:
#             ob_stat.set_from_init(init_mean, init_std, init_count=1e5)
#
#     def _get_pos(self, model):
#         mass = model.body_mass
#         xpos = model.data.xipos
#         center = (np.sum(mass * xpos, 0) / np.sum(mass))
#         return center[0], center[1], center[2]
#
#     def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, policy_seed=None,
#                 bc_choice=None):
#         """
#         If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
#         Otherwise, no action noise will be added.
#         """
#         env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
#         timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
#         rews = []
#         x_traj, y_traj = np.zeros(timestep_limit), np.zeros(timestep_limit)
#         t = 0
#         if save_obs:
#             obs = []
#
#         if policy_seed:
#             env.seed(policy_seed)
#             np.random.seed(policy_seed)
#             if random_stream:
#                 random_stream.seed(policy_seed)
#
#         ob = env.reset()
#         for _ in range(timestep_limit):
#             ac = self.act(ob[None], random_stream=random_stream)[0]
#             if save_obs:
#                 obs.append(ob)
#             ob, rew, done, _ = env.step(ac)
#             x_traj[t], y_traj[t], _ = self._get_pos(env.unwrapped.model)
#             rews.append(rew)
#             t += 1
#             if render:
#                 env.render()
#             if done:
#                 break
#
#         x_pos, y_pos, _ = self._get_pos(env.unwrapped.model)
#         rews = np.array(rews, dtype=np.float32)
#         x_traj[t:] = x_traj[t - 1]
#         y_traj[t:] = y_traj[t - 1]
#         if bc_choice and bc_choice == "traj":
#             novelty_vector = np.concatenate((x_traj, y_traj), axis=0)
#         else:
#             novelty_vector = np.array([x_pos, y_pos])
#         if save_obs:
#             return rews, t, np.array(obs), novelty_vector
#         return rews, t, novelty_vector
#
# def bins(x, dim, num_bins, name):
#     scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
#     scores_nab = tf.reshape(scores, [-1, dim, num_bins])
#     return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1
#
#
# class ESAtariPolicy(Policy):
#     def _initialize(self, ob_space, ac_space):
#         self.ob_space_shape = ob_space.shape
#         self.ac_space = ac_space
#         self.num_actions = ac_space.n
#
#         with tf.variable_scope(type(self).__name__) as scope:
#             o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))
#             is_ref_ph = tf.placeholder(tf.bool, shape=[])
#
#             a = self._make_net(o, is_ref_ph)
#             self._act = U.function([o, is_ref_ph], a)
#         return scope
#
#     def _make_net(self, o, is_ref):
#         x = o
#         x = layers.convolution2d(x, num_outputs=16, kernel_size=8, stride=4, activation_fn=None, scope='conv1')
#         x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
#                               activation_fn=tf.nn.relu, epsilon=1e-3)
#         x = layers.convolution2d(x, num_outputs=32, kernel_size=4, stride=2, activation_fn=None, scope='conv2')
#         x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
#                               activation_fn=tf.nn.relu, epsilon=1e-3)
#
#         x = layers.flatten(x)
#         x = layers.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc')
#         x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
#                               activation_fn=tf.nn.relu, epsilon=1e-3)
#         a = layers.fully_connected(x, num_outputs=self.num_actions, activation_fn=None, scope='out')
#         return tf.argmax(a, 1)
#
#     def set_ref_batch(self, ref_batch):
#         self.ref_list = []
#         self.ref_list.append(ref_batch)
#         self.ref_list.append(True)
#
#     @property
#     def needs_ob_stat(self):
#         return False
#
#     @property
#     def needs_ref_batch(self):
#         return True
#
#     def initialize_from(self, filename):
#         """
#         Initializes weights from another policy, which must have the same architecture (variable names),
#         but the weight arrays can be smaller than the current policy.
#         """
#         with h5py.File(filename, 'r') as f:
#             f_var_names = []
#             f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
#             assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'
#
#             init_vals = []
#             for v in self.all_variables:
#                 shp = v.get_shape().as_list()
#                 f_shp = f[v.name].shape
#                 assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
#                     'This policy must have more weights than the policy to load'
#                 init_val = v.eval()
#                 # ob_mean and ob_std are initialized with nan, so set them manually
#                 if 'ob_mean' in v.name:
#                     init_val[:] = 0
#                     init_mean = init_val
#                 elif 'ob_std' in v.name:
#                     init_val[:] = 0.001
#                     init_std = init_val
#                 # Fill in subarray from the loaded policy
#                 init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
#                 init_vals.append(init_val)
#             self.set_all_vars(*init_vals)
#
#     def act(self, train_vars, random_stream=None):
#         return self._act(*train_vars)
#
#     def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, worker_stats=None,
#                 policy_seed=None):
#         """
#         If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
#         Otherwise, no action noise will be added.
#         """
#         env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
#
#         timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
#         rews = [];
#         novelty_vector = []
#         t = 0
#
#         if save_obs:
#             obs = []
#
#         if policy_seed:
#             env.seed(policy_seed)
#             np.random.seed(policy_seed)
#             if random_stream:
#                 random_stream.seed(policy_seed)
#
#         ob = env.reset()
#         self.act(self.ref_list, random_stream=random_stream)  # passing ref batch through network
#
#         for _ in range(timestep_limit):
#             start_time = time.time()
#             ac = self.act([ob[None], False], random_stream=random_stream)[0]
#
#             if worker_stats:
#                 worker_stats.time_comp_act += time.time() - start_time
#
#             start_time = time.time()
#             ob, rew, done, info = env.step(ac)
#             ram = env.unwrapped._get_ram()  # extracts RAM state information
#
#             if save_obs:
#                 obs.append(ob)
#             if worker_stats:
#                 worker_stats.time_comp_step += time.time() - start_time
#
#             rews.append(rew)
#             novelty_vector.append(ram)
#
#             t += 1
#             if render:
#                 env.render()
#             if done:
#                 break
#
#         rews = np.array(rews, dtype=np.float32)
#         if save_obs:
#             return rews, t, np.array(obs), np.array(novelty_vector)
#         return rews, t, np.array(novelty_vector)
#
#