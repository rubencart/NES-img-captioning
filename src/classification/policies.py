import logging

import numpy as np
import torch
from abc import ABC

from torch import nn

from algorithm.nets import PolicyNet
from algorithm.policies import Policy, NetsPolicy, SeedsPolicy

logger = logging.getLogger(__name__)


class ClfPolicy(Policy, ABC):
    def rollout(self, data, config):
        # CAUTION: memory: https://pytorch.org/docs/stable/notes/faq.html
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')
        # logger.info('***** DEVICE : {} *****'.format(device))

        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        self.policy_net.to(device)

        outputs = self.policy_net(inputs)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        result = -float(loss.item())

        del inputs, labels, outputs, loss, criterion
        return result

    def accuracy_on(self, dataloader, config, directory) -> float:
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        accuracies = []
        end = config.num_val_batches if config.num_val_batches else len(dataloader)
        for i, data in enumerate(dataloader):
            if i >= end:
                break

            torch.set_grad_enabled(False)
            self.policy_net.eval()

            device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')
            # logger.info('***** DEVICE : {} *****'.format(device))

            inputs, labels = data
            inputs.to(device)
            labels.to(device)
            self.policy_net.to(device)

            # logging.info('doing FW')
            outputs = self.policy_net(inputs)
            # logging.info('FW done: {}'.format(outputs))

            prediction = outputs.cpu().detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct = prediction.eq(labels.view_as(prediction)).sum().item()
            accuracies.append(float(correct) / labels.size()[0])
            del inputs, labels, outputs, prediction, correct

        # todo accuracy calculation not correct, proportions
        accuracy = np.mean(accuracies).item()

        del accuracies
        return accuracy


class NetsClfPolicy(ClfPolicy, NetsPolicy):
    pass


class SeedsClfPolicy(ClfPolicy, SeedsPolicy):
    pass
