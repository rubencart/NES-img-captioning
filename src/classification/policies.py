import torch
from abc import ABC

from torch import nn

from algorithm.nets import PolicyNet
from algorithm.policies import Policy, NetsPolicy, SeedsPolicy


class ClfPolicy(Policy, ABC):
    def rollout(self, data):
        # CAUTION: memory: https://pytorch.org/docs/stable/notes/faq.html
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        inputs, labels = data
        outputs = self.policy_net(inputs)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        result = -float(loss.item())

        del inputs, labels, outputs, loss, criterion
        return result

    def accuracy_on(self, data):
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        inputs, labels = data
        outputs = self.policy_net(inputs)

        prediction = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        accuracy = float(correct) / labels.size()[0]

        del inputs, labels, outputs, prediction, correct
        return accuracy


class NetsClfPolicy(ClfPolicy, NetsPolicy):
    pass


class SeedsClfPolicy(ClfPolicy, SeedsPolicy):
    pass
