import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10


class CIFARDataset:
    def __init__(self, args):
        self.data = args.dataset

        if self.data == 'CIFAR10':
            self.nClass = 10
            self.nTrain = 50000
            self.nTest = 10000

        elif self.data == 'CIFAR100':
            self.nClass = 100
            self.nTrain = 50000
            self.nTest = 10000

        self.dataset = {}
        self._getData()


    def _getData(self):
        self._data_transform()
        if self.data == 'cifar10':
            self.dataset['train'] = CIFAR10('./dataset/cifar10', train=True, download=True, transform=self.train_transform)
            self.dataset['unlabeled'] = CIFAR10('./dataset/cifar10', train=True, download=True, transform=self.train_transform)
            self.dataset['test'] = CIFAR10('./dataset/cifar10', train=False, download=True, transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets

        elif self.data == 'cifar100':
            self.dataset['train'] = CIFAR100('./dataset/cifar100', train=True, download=True, transform=self.train_transform)
            self.dataset['unlabeled'] = CIFAR100('./dataset/cifar100', train=True, download=True, transform=self.train_transform)
            self.dataset['test'] = CIFAR100('./dataset/cifar100', train=False, download=True, transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets
        return self.dataset

    def _data_transform(self):
        if self.data == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            add_transform = [T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]

        elif self.data == 'cifar100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            add_transform = [T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]

        base_transform = [T.ToTensor(), T.Normalize(mean, std)]
        self.test_transform = T.Compose(base_transform)
        self.train_transform = T.Compose(base_transform + add_transform)