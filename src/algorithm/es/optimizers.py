"""
    Code based on https://github.com/openai/evolution-strategies-starter/
"""
import numpy as np
import torch


class Optimizer(object):
    def __init__(self, theta):
        self.theta = theta
        self.dim = len(self.theta)
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.theta
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        new_theta = self.theta + step
        self.theta = new_theta
        return ratio, new_theta

    def set_theta(self, theta):
        self.theta = theta
        self.dim = len(theta)

    def _compute_step(self, globalg):
        raise NotImplementedError

    def save_to_file(self, path):
        raise NotImplementedError

    def load_from_file(self, path):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.9):
        super().__init__(theta)
        self.v = np.zeros(self.dim, dtype=np.float)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, grad):
        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        step = -self.stepsize * self.v
        return step

    def save_to_file(self, path):
        pass

    def load_from_file(self, path):
        pass


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float)
        self.v = np.zeros(self.dim, dtype=np.float)

    def _compute_step(self, grad):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def save_to_file(self, path):
        state = {
            # 'theta': self.theta,
            'dim': self.dim,
            't': self.t,
            'stepsize': self.stepsize,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'm': self.m,
            'v': self.v,
        }
        torch.save(state, path)

    def load_from_file(self, path):
        state = torch.load(path, map_location='cpu')
        # self.theta = state['theta']
        self.dim = state['dim']
        self.t = state['t']
        self.stepsize = state['stepsize']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.epsilon = state['epsilon']
        self.m = state['m']
        self.v = state['v']
