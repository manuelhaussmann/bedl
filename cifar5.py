from copy import copy

import torch as th
from torchvision import datasets
from torchvision import transforms as trafo


def create_cifar5():
    train_data_orig = datasets.CIFAR10("../data/cifar10", train=True, download=True,
                                       transform=trafo.Compose([trafo.ToTensor()]))
    test_data_orig = datasets.CIFAR10("../data/cifar10", train=False, download=True,
                                 transform=trafo.Compose([trafo.ToTensor()]))

    # Normalization computed on the reduced training set
    tr = trafo.Compose([trafo.ToTensor(), trafo.Normalize((0.4905, 0.4854, 0.4514), (0.2454, 0.2415, 0.2620))])


    # Prepare training and out of distribution data
    mask = th.zeros(len(train_data_orig))
    labels = th.LongTensor(train_data_orig.targets)


    mask[labels.eq(0)] = 1
    mask[labels.eq(1)] = 1
    mask[labels.eq(2)] = 1
    mask[labels.eq(3)] = 1
    mask[labels.eq(4)] = 1


    train_data_reduced = copy(train_data_orig)
    ood_data_reduced = copy(train_data_orig)

    train_data_reduced.data = train_data_orig.data[mask.numpy() == 1]
    train_data_reduced.targets = labels[mask.eq(1)]
    train_data_reduced.transform = tr

    ood_data_reduced.data = train_data_orig.data[mask.numpy() == 0]
    ood_data_reduced.targets = labels[mask.eq(0)] - 5
    ood_data_reduced.transform = tr


    # Prepare reduced test data
    mask = th.zeros(len(test_data_orig))
    labels = th.LongTensor(test_data_orig.targets)


    mask[labels.eq(0)] = 1
    mask[labels.eq(1)] = 1
    mask[labels.eq(2)] = 1
    mask[labels.eq(3)] = 1
    mask[labels.eq(4)] = 1


    test_data_reduced = copy(test_data_orig)
    test_data_reduced.data = test_data_orig.data[mask.numpy() == 1]
    test_data_reduced.targets = labels[mask.eq(1)]
    test_data_reduced.transform = tr

    return train_data_reduced, test_data_reduced, ood_data_reduced


def create_cifar10():
    train_data = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trafo.Compose([
            trafo.ToTensor(),
            trafo.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)),
        ]))
    test_data = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trafo.Compose([
            trafo.ToTensor(),
            trafo.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)),
        ]))

    print("WARNING: There is no OOD Loader here")
    return train_data, test_data, 0
