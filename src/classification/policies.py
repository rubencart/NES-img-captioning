import logging

import numpy as np
import torch

from torch import nn

from algorithm.nets import PolicyNet
from algorithm.policies import Policy

logger = logging.getLogger(__name__)


class ClfPolicy(Policy):

    def rollout(self, placeholder, data, config):
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)

        device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')

        inputs, labels = data
        placeholder.data.resize_(inputs.shape).copy_(inputs)
        placeholder.to(device)
        labels.to(device)
        self.policy_net.to(device)

        # virtual batch norm
        if self.vbn:
            self.policy_net.train()
            self.policy_net(torch.empty_like(self.ref_batch).copy_(self.ref_batch))

        self.policy_net.eval()

        outputs = self.policy_net(placeholder)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # print(loss) --> tensor(2.877)
        result = -float(loss.detach().item())

        del inputs, labels, outputs, loss, criterion
        return result

    def accuracy_on(self, dataloader, config, directory) -> float:
        assert self.policy_net is not None, 'Set model first!'
        assert isinstance(self.policy_net, PolicyNet), '{}'.format(type(self.policy_net))

        torch.set_grad_enabled(False)
        self.policy_net.eval()

        accuracies = []
        end = config.num_val_batches if config.num_val_batches else len(dataloader)
        for i, data in enumerate(dataloader):
            if i >= end:
                break

            device = torch.device('cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu')

            inputs, labels = data
            inputs.to(device)
            labels.to(device)
            self.policy_net.to(device)
            outputs = self.policy_net(inputs)

            # get the index of the max log-probability
            prediction = outputs.cpu().detach().argmax(dim=1, keepdim=True)

            correct = prediction.eq(labels.view_as(prediction)).sum().item()
            accuracies.append(float(correct) / labels.size()[0])
            del inputs, labels, outputs, prediction, correct

        # todo accuracy calculation not 100% correct, last batch might be smaller so should count less
        accuracy = np.mean(accuracies).item()

        del accuracies
        return accuracy
