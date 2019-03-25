import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

from policies import MnistNet, random_state, Cifar10Net

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
from utils import mkdir_p

parser = argparse.ArgumentParser()
parser.add_argument('--save_to_path', type=str, default='snapshots/sgd_{pid}/', help='')
parser.add_argument('--save_to_file', type=str, default='params_acc{acc}.pth', help='')
parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'],
                    type=str, default='mnist', help='')
args = parser.parse_args()

if args.dataset == 'mnist':
    net = MnistNet(random_state())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST has 60k training images
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

else:
    net = Cifar10Net(random_state())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)

# 60k / 128 = 470 batches
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=True, num_workers=4)

try:
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print('Training...: {}'.format(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            if i % 200 == 199:  # print every 2000 mini-batches
                # print('[%d, %5d] running loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 2000))
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss))
                running_loss = 0.0

                with torch.no_grad():
                    # outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()

                    print('Accuracy: %d %%' % (100 * correct / total))
except KeyboardInterrupt:
    pass

print('Finished Training')
print('Testing')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

path = args.save_to_path.format(pid=os.getpid())
file = args.save_to_file.format(acc=round(100 * correct / total))
mkdir_p(path)
print('Saving params to file: {}'.format(os.path.join(path + file)))

assert not os.path.exists(os.path.join(path + file))
torch.save(net.state_dict(), os.path.join(path + file))
