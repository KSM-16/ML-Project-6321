import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10


class CIFARDataset:
    """Wrapper class for loading CIFAR-10 or CIFAR-100 with transforms."""

    def __init__(self, args):
        self.data = args.dataset  # dataset name (cifar10 or cifar100)

        # Store dataset-specific metadata
        if self.data == 'CIFAR10':
            self.nClass = 10
            self.nTrain = 50000
            self.nTest = 10000

        elif self.data == 'CIFAR100':
            self.nClass = 100
            self.nTrain = 50000
            self.nTest = 10000

        self.dataset = {}
        self._getData()  # Load dataset and apply transforms


    def _getData(self):
        """Load train, unlabeled, and test splits for CIFAR-10/100."""
        self._data_transform()  # Prepare transforms

        # Load CIFAR-10 datasets
        if self.data == 'cifar10':
            self.dataset['train'] = CIFAR10('./dataset/cifar10', train=True, download=True,
                                            transform=self.train_transform)
            self.dataset['unlabeled'] = CIFAR10('./dataset/cifar10', train=True, download=True,
                                                transform=self.train_transform)
            self.dataset['test'] = CIFAR10('./dataset/cifar10', train=False, download=True,
                                           transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets  # Ground truth labels

        # Load CIFAR-100 datasets
        elif self.data == 'cifar100':
            self.dataset['train'] = CIFAR100('./dataset/cifar100', train=True, download=True,
                                             transform=self.train_transform)
            self.dataset['unlabeled'] = CIFAR100('./dataset/cifar100', train=True, download=True,
                                                 transform=self.train_transform)
            self.dataset['test'] = CIFAR100('./dataset/cifar100', train=False, download=True,
                                            transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets

        return self.dataset


    def _data_transform(self):
        """Define normalization and augmentation transforms for CIFAR datasets."""

        # CIFAR-10 statistics + augmentation
        if self.data == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            add_transform = [T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]

        # CIFAR-100 statistics + augmentation
        elif self.data == 'cifar100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            add_transform = [T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]

        # Base transforms applied to both datasets
        base_transform = [T.ToTensor(), T.Normalize(mean, std)]

        # Assign transforms
        self.test_transform = T.Compose(base_transform)
        self.train_transform = T.Compose(base_transform + add_transform)
