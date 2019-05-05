"""
contains code from https://github.com/uber-research/safemutations
and https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
"""
import logging

import torch
from abc import abstractmethod, ABC

from torch import nn

from algorithm.ga.safe_mutations import Sensitivity
from algorithm.tools.utils import random_state

logger = logging.getLogger(__name__)


class SerializableModel:

    @abstractmethod
    def serialize(self, path=''):
        raise NotImplementedError

    @abstractmethod
    def from_serialized(self, serialized):
        raise NotImplementedError


class PolicyNet(nn.Module, SerializableModel, ABC):
    def __init__(self, rng_state=None, from_param_file=None, grad=False, options=None, vbn=False):
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
        self.grad = grad
        self.vbn = vbn

        self.eval()

        self.sensitivity_wrapper = Sensitivity(self)

    def _initialize_params(self):
        if self.from_param_file:
            self.load_state_dict(torch.load(self.from_param_file, map_location='cpu'))

        for name, param in self.named_parameters():
            # todo this is completely unnecessary
            # if tensor.size() not in self.add_tensors:
                # print('CAUTION: new tensor size: ', tensor.size())
                # self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())

            logging.info('Params: %s, %s, %s', name, param.size(), param.data.abs().mean())

            if not self.from_param_file and 'bn' not in name and 'ln' not in name:
                # exclude batch norm layers from xavier initialization
                if 'weight' in name:
                    # todo kaiming normal or:
                    # We use Xavier initialization (Glorot & Bengio, 2010) as our policy initialization
                    # function φ where all bias weights are set to zero, and connection weights are drawn
                    # from a standard normal distribution with variance 1/Nin, where Nin is the number of
                    # incoming connections to a neuron
                    # nn.init.kaiming_normal_(tensor)
                    nn.init.xavier_normal_(param)
                else:
                    param.data.zero_()

        self.nb_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.nb_params = self._count_parameters()

        logging.warning('LEARNABLE: {}, TOTAL: {}'.format(self.nb_learnable_params, self.nb_params))

        # freeze all params
        if not self.grad:
            for param in self.parameters():
                param.requires_grad = False

    def evolve(self, sigma, rng_state=None, safe=False):
        # Evolve params 1 step
        # rng_state = int
        rng = rng_state if rng_state is not None else random_state()

        torch.manual_seed(rng)
        # todo this is needed for when in seeds mode we want to be able to evolve uncompressed networks
        # as well (not strictly necessary)
        # self.evolve_states.append((sigma, rng))

        # for name, tensor in sorted(self.named_parameters()):
        #     to_add = self.add_tensors[tensor.size()]
        #     # fill to_add elements sampled from normal distr
        #     to_add.normal_(mean=0.0, std=sigma)
        #     tensor.data.add_(to_add.to(tensor.device))
        #     tensor.requires_grad = False

        param_vector = nn.utils.parameters_to_vector(self.parameters())
        noise = torch.empty_like(param_vector, requires_grad=False).normal_(mean=0.0, std=sigma)

        # logging.info('noise %s', noise)
        if safe:
            noise /= self.sensitivity_wrapper.get_sensitivity()
            # logging.info('sens %s', self.sensitivity_wrapper.get_sensitivity())
        # logging.info('noise %s', noise)

        new_param_vector = param_vector + noise
        self.set_from_vector(new_param_vector)

        for param in self.parameters():
            param.requires_grad = False

        return torch.empty_like(noise).copy_(noise).numpy()

    def calc_sensitivity(self, task_id, parent_id, experiences, batch_size, directory, underflow, method):
        self.sensitivity_wrapper.calc_sensitivity(task_id, parent_id, experiences, batch_size, directory,
                                                  underflow, method)

    def set_sensitivity_vector(self, vector):
        self.sensitivity_wrapper.set_sensitivity(vector)

    def get_sensitivity_vector(self):
        return self.sensitivity_wrapper.get_sensitivity()

    def extract_grad(self):
        tot_size = self.nb_params
        pvec = torch.zeros(tot_size, dtype=torch.float)
        count = 0
        # params = torch.zeros(tot_size, dtype=torch.float32)
        for param in self.parameters():
            sz = param.grad.data.flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.flatten()
            # params[count:count + sz] = param.data.flatten()
            count += sz
        return pvec.clone().detach()  # , params

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

    def set_from_vector(self, vector):
        assert len(vector) == self.nb_learnable_params
        nn.utils.vector_to_parameters(vector, self.parameters())

    def parameter_vector(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def _contained_forward(self, x, orig_bs=0, i=-1):
        raise NotImplementedError

    # def subject_to_sensitivity_params(self):
    #     raise NotImplementedError


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
