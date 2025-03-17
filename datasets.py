import multiprocessing
import os

import torch
from loguru import logger
from torch.utils import data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]  # this is mean of stds per images
cifar10_std_fixed = (0.2471, 0.2435, 0.2616) # true stds

"""
MNIST and CIFAR10 datasets with `index` also returned in `__getitem__`
"""


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download)
        self.use_index = use_index


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download)
        self.use_index = use_index

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.use_index:
            return img, target, index
        else:
            return img, target


def load_data(args, data, batch_size, test_batch_size, use_index=False, aug=True, unnormalized_eval=False):
    root = args.datadir
    if data == 'MNIST':
        """Fix 403 Forbidden error in downloading MNIST
        See https://github.com/pytorch/vision/issues/1938."""
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        dummy_input = torch.randn(2, 1, 28, 28)
        mean, std = torch.tensor([0.0]), torch.tensor([1.0])
        train_data = datasets.MNIST(f"{root}", train=True, download=True, transform=transforms.ToTensor())
        if args.valid_share is not None:
            test_data = datasets.MNIST(
                f"{root}", train=True, download=True, transform=transforms.ToTensor())
        else:
            test_data = datasets.MNIST(
                f"{root}", train=False, download=True, transform=transforms.ToTensor())
    elif data == 'CIFAR':
        mean = torch.tensor(cifar10_mean)
        std = torch.tensor(cifar10_std)
        if args.true_std:
            std = torch.tensor(cifar10_std_fixed)
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean=mean, std=std)
        if aug:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 2, padding_mode='edge'), # NOTE: padding from expressive losses paper.
                # NOTE: The above padding is significantly more effective (hence the mismatch with expressive losses results on CNN7).
                transforms.RandomCrop(32, padding=4), # NOTE: padding from N-FGSM paper
                transforms.ToTensor(),
                normalize])
        else:
            # No random cropping
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        if unnormalized_eval:
            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = CIFAR10(f"{root}", train=True, download=True,
                             transform=transform, use_index=use_index)
        if args.valid_share is not None:
            test_data = CIFAR10(f"{root}", train=True, download=True,
                                transform=transform_test, use_index=use_index)
        else:
            test_data = CIFAR10(f"{root}", train=False, download=True,
                                transform=transform_test, use_index=use_index)

    elif data == 'CIFAR100':
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std = torch.tensor([0.2673, 0.2564, 0.2762])
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean=mean, std=std)

        if aug:
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        if unnormalized_eval:
            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = CIFAR100(f"{root}", train=True, download=True,
                             transform=transform)
        if args.valid_share is not None:
            test_data = CIFAR100(f"{root}", train=True, download=True,
                                transform=transform_test)
        else:
            test_data = CIFAR100(f"{root}", train=False, download=True,
                                transform=transform_test)


    elif data == "SVHN":
        mean = torch.tensor([0.4380, 0.4440, 0.4730])
        std = torch.tensor([0.1751, 0.1771, 0.1744])
        mean = torch.tensor([0, 0, 0]) if args.no_norm else mean
        std = torch.tensor([1, 1, 1]) if args.no_norm else std

        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean=mean, std=std)
        train_data = datasets.SVHN(f"{root}", split='train', download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
        if unnormalized_eval:
            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        if args.valid_share is not None:
            test_data = datasets.SVHN(f"{root}", split='train', download=True, transform=transform_test)
        else:
            test_data = datasets.SVHN(f"{root}", split='test', download=True, transform=transform_test)
    elif data == "tinyimagenet":
        mean = torch.tensor([0.4802, 0.4481, 0.3975])
        std = torch.tensor([0.2302, 0.2265, 0.2262])
        dummy_input = torch.randn(2, 3, 64, 64)
        normalize = transforms.Normalize(mean=mean, std=std)
        data_dir = f"{root}/tiny-imagenet-200"
        train_data = datasets.ImageFolder(data_dir + '/train',
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(64, 4, padding_mode='edge'),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
        if unnormalized_eval:
            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        if args.valid_share is not None:
            test_data = datasets.ImageFolder(data_dir + '/train', transform=transform_test)
        else:
            test_data = datasets.ImageFolder(data_dir + '/val', transform=transform_test)
    elif data == "imagenet64":
        # Code adapted from auto_lirpa's repository (removing the 56 cropping)
        mean = torch.tensor([0.4815, 0.4578, 0.4082])
        std = torch.tensor([0.2153, 0.2111, 0.2121])
        dummy_input = torch.randn(2, 3, 64, 64)
        normalize = transforms.Normalize(mean=mean, std=std)
        data_dir = f"{root}/imagenet64"
        train_data = datasets.ImageFolder(data_dir + '/train',
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(64, 4, padding_mode='edge'),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
        if unnormalized_eval:
            transform_test = transforms.Compose([transforms.ToTensor()])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        if args.valid_share is not None:
            test_data = datasets.ImageFolder(data_dir + '/train', transform=transform_test)
        else:
            test_data = datasets.ImageFolder(data_dir + '/test', transform=transform_test)

    else:
        raise ValueError(f"Unsupported value for dataset name: {data}")

    if args.valid_share is not None:
        train_size = int(args.valid_share * len(train_data))
        test_size = len(train_data) - train_size
        if not args.valid_shuffle:
            # the test data already points to the training set, but without the train-time transforms
            test_data = torch.utils.data.Subset(test_data, range(train_size, train_size + test_size))
            train_data = torch.utils.data.Subset(train_data, range(train_size))
        else:
            # randomized validation selection
            random_indices = torch.randperm(len(train_data))
            test_data = torch.utils.data.Subset(test_data, random_indices[train_size:])
            train_data = torch.utils.data.Subset(train_data, random_indices[:train_size])

    if args.subset < 1:
        train_subset_size = int(len(train_data) * args.subset)
        test_subset_size = int(len(test_data) * args.subset)
        train_data = torch.utils.data.Subset(train_data, range(train_subset_size))
        test_data = torch.utils.data.Subset(test_data, range(test_subset_size))

    train_data = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.data_loader_workers,
    )
    test_data = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, pin_memory=True, num_workers=args.data_loader_workers,
    )

    # NOTE: in the unnormalized case, the original std and mean are passed
    train_data.mean = test_data.mean = mean
    train_data.std = test_data.std = std
    for loader in [train_data, test_data]:
        loader.mean, loader.std = mean, std
        loader.data_max = data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        loader.data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    dummy_input = dummy_input.to(args.device)

    return dummy_input, train_data, test_data

