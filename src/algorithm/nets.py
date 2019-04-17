"""
contains code from https://github.com/uber-research/safemutations
and https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
"""
import json
import logging
import os
import time

import torch
from abc import abstractmethod, ABC

from torch import nn

from algorithm.tools.utils import random_state, find_file_with_pattern

logger = logging.getLogger(__name__)


class SerializableModel:

    @abstractmethod
    def serialize(self, path=''):
        raise NotImplementedError

    @abstractmethod
    def from_serialized(self, serialized):
        raise NotImplementedError


class PolicyNet(nn.Module, SerializableModel, ABC):
    def __init__(self, rng_state=None, from_param_file=None, grad=False):
        # todo from param file not really needed anymore since now serialized == from param file?
        super(PolicyNet, self).__init__()

        self.rng_state = rng_state
        self.from_param_file = from_param_file

        if rng_state:
            torch.manual_seed(rng_state)

        self.evolve_states = []
        self.add_tensors = {}

        self.nb_learnable_params = 0
        self.nb_params = 0
        self.sensitivity = None
        self.it = -1
        self.orig_batch_size = 0

        self.grad = grad

    def _initialize_params(self):
        if self.from_param_file:
            self.load_state_dict(torch.load(self.from_param_file, map_location='cpu'))

        for name, tensor in self.named_parameters():
            # todo this is completely unnecessary
            if tensor.size() not in self.add_tensors:
                # print('CAUTION: new tensor size: ', tensor.size())
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())

            if not self.from_param_file:
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

        self.nb_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params = self._count_parameters()

        # freeze all params
        if not self.grad:
            for param in self.parameters():
                param.requires_grad = False

    def evolve(self, sigma, rng_state=None):
        # Evolve params 1 step
        # rng_state = int
        rng = rng_state if rng_state is not None else random_state()

        torch.manual_seed(rng)
        # todo this is needed for when in seeds mode we want to be able to evolve uncompressed networks
        # as well (not strictly necessary)
        # self.evolve_states.append((sigma, rng))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            # fill to_add elements sampled from normal distr
            to_add.normal_(mean=0.0, std=sigma)
            tensor.data.add_(to_add.to(tensor.device))
            tensor.requires_grad = False

        for param in self.parameters():
            param.requires_grad = False

    def evolve_safely(self, sigma):
        # sensitivity = self._get_sensitivity()

        torch.manual_seed(random_state())

        param_vector = nn.utils.parameters_to_vector(self.parameters())
        noise = torch.empty_like(param_vector, requires_grad=False).normal_(mean=0.0, std=sigma)

        # noise_spread = noise.max() - noise.min()

        # logger.info('Old Params: {}'.format((param_vector.min(), param_vector.mean(), param_vector.max())))
        # logger.info('Noise: {}'.format((noise.min(), noise.mean(), noise.max())))
        noise /= self.sensitivity

        # new_noise_spread = noise.max() - noise.min()

        # logger.info('Noise / sens: {}'.format((noise.min(), noise.mean(), noise.max())))
        # noise[noise > sigma] = sigma
        # noise = noise * (noise_spread / new_noise_spread)
        # logger.info('Noise norm: {}'.format((noise.min(), noise.mean(), noise.max())))

        new_param_vector = param_vector + noise
        # logger.info('New Params: {}'.format((param_vector.min(), param_vector.mean(), param_vector.max())))

        nn.utils.vector_to_parameters(new_param_vector, self.parameters())

        for param in self.parameters():
            param.requires_grad = False

    def set_sensitivity(self, task_id, parent_id, experiences, directory, underflow):
        sensitivity_filename = 'sens_t{t}_p{p}.txt'.format(t=task_id, p=parent_id)
        if find_file_with_pattern(sensitivity_filename, directory):
            # logger.info('Loaded sensitivity for known parent')
            self.sensitivity = torch.load(os.path.join(directory, sensitivity_filename))
        else:
            if self.orig_batch_size == 0:
                self.orig_batch_size = experiences.size(0)

            start_time = time.time()
            torch.set_grad_enabled(True)
            for param in self.parameters():
                param.requires_grad = True

            # self.it = it

            # todo experiences requires grad?
            experiences = torch.Tensor(experiences)  # .requires_grad_()
            old_output = self.forward(experiences)
            num_outputs = old_output.size(1)

            jacobian = torch.zeros(num_outputs, self.nb_params)
            grad_output = torch.zeros(*old_output.size())

            for k in range(num_outputs):
                self.zero_grad()
                grad_output.zero_()
                grad_output[:, k] = 1.0

                # torch.autograd.backward(tensors=old_output, grad_tensors=grad_output, retain_graph=True)
                # torch.autograd.grad(outputs=old_output, inputs=input_data, grad_outputs=...)
                old_output.backward(gradient=grad_output, retain_graph=True)
                jacobian[k] = self._extract_grad()

            # sum over the outputs, keeping params separate
            proportion = float(self.orig_batch_size) / experiences.size(0)
            sensitivity = torch.sqrt((jacobian ** 2).sum(0)) * proportion
            # sensitivity[sensitivity < 1e-6] = 1.0
            sensitivity[sensitivity < underflow] = underflow

            torch.set_grad_enabled(False)
            for param in self.parameters():
                param.requires_grad = False
            for param in sensitivity:
                param.requires_grad = False

            time_elapsed = time.time() - start_time
            logger.info('Safe mutation sensitivity computed in {0:.2f}s'.format(time_elapsed))
            torch.save(sensitivity.clone().detach().requires_grad_(False),
                       os.path.join(directory, sensitivity_filename))
            self.sensitivity = sensitivity.clone().detach().requires_grad_(False)
            logger.info('Sensitivity parent {}: {}'
                        .format(parent_id, (sensitivity.min(), sensitivity.mean(), sensitivity.max())))
            del old_output, jacobian, grad_output, experiences, num_outputs, sensitivity

    def _extract_grad(self):
        tot_size = self.nb_params
        pvec = torch.zeros(tot_size, dtype=torch.float32)
        count = 0
        for param in self.parameters():
            sz = param.grad.data.flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.flatten()
            count += sz
        return pvec.clone().detach()

    def _count_parameters(self):
        count = nn.utils.parameters_to_vector(self.parameters()).size(0)
        return count

    def get_nb_learnable_params(self):
        return self.nb_learnable_params

    # see comment in evolve()
    # def compress(self):
    #     return CompressedModel(self.rng_state, self.evolve_states, self.from_param_file)

    def serialize(self, path=''):
        # return copy.deepcopy(self.state_dict())
        # filename = 'params_i{i}.pth'.format(i=i)
        # path_to_offspring = os.path.join(log_dir, filename)

        torch.save(self.state_dict(), path)
        return path

    def from_serialized(self, serialized):
        # assert isinstance(serialized, dict)
        # self.load_state_dict(serialized)
        # todo map_location not cpu if on gpu!
        state_dict = torch.load(serialized, map_location='cpu')
        self.from_param_file = serialized
        self.load_state_dict(state_dict)

    # def forward(self, x):
    #     pass


class CompressedModel(SerializableModel):
    def __init__(self, start_rng: int = None, other_rng: list = None, from_param_file: str = None):
        if start_rng is None and from_param_file is None:
            self.start_rng, self.from_param_file = random_state(), None
        elif start_rng is None and from_param_file is not None:
            self.start_rng, self.from_param_file = None, from_param_file
        elif start_rng is not None and from_param_file is None:
            self.start_rng, self.from_param_file = start_rng, None
        else:
            raise ValueError('start_rng and from_param_file cannot be both set')

        self.other_rng = other_rng if other_rng is not None else []

    def evolve(self, sigma, rng_state=None):
        # evolve params 1 step
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))

    def uncompress(self, to_class_name):
        # evolve through all steps
        m = to_class_name(self.start_rng, self.from_param_file)
        for sigma, rng in self.other_rng:
            m.evolve(sigma, rng)
        return m

    def serialize(self, path=''):
        return self.__dict__

    def from_serialized(self, serialized):
        assert isinstance(serialized, CompressedModel)
        self.start_rng = serialized.start_rng
        self.other_rng = serialized.other_rng
        self.from_param_file = serialized.from_param_file

    def __str__(self):
        start = self.start_rng if self.start_rng else self.from_param_file

        result = '[( {start} )'.format(start=start)
        for _, rng in self.other_rng:
            result += '( {rng} )'.format(rng=rng)
        return result + ']'
