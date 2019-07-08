import torchvision
from torchvision.transforms import transforms

from algorithm.tools.experiment import Experiment
from algorithm.tools.utils import Config


class MnistExperiment(Experiment):
    """
        Subclass for MNIST experiment
    """

    def __init__(self, exp, config: Config, master=True):
        super().__init__(exp, config, master=master)

    def init_loaders(self, config=None, batch_size=None, workers=None, _=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.trainloader, self.valloader, self.testloader = \
            self._init_torchvision_loaders(torchvision.datasets.MNIST, transform, config, batch_size, workers)
        self._orig_trainloader_lth = len(self.trainloader)