import os

from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def create_transform(size):
    return transforms.Compose([
        transforms.RandomCrop(size, padding=size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])


def split_dataset(dataset, train_ratio=0.9, num_classes=10):
    index_label = [[] for _ in range(num_classes)]
    for index, data in enumerate(dataset):
        _, label = data
        # if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
        #     y = y.item()
        index_label[label].append(index)
    index_train, index_test = [], []
    for label in range(num_classes):
        train_count = int(len(index_label[label]) * train_ratio)
        index_train += index_label[label][:train_count]
        index_test += index_label[label][train_count:]
    return Subset(dataset, index_train), Subset(dataset, index_test)


def create_dataset(config):
    if config.name == 'cifar-10':
        transform = create_transform(config.image_size)
        dataset_train = CIFAR10(config.path, train=True, transform=transform, download=True)
        dataset_test = CIFAR10(config.path, train=False, transform=transform, download=True)
    elif config.name == 'cifar-100':
        transform = create_transform(config.image_size)
        dataset_train = CIFAR100(config.path, train=True, transform=transform, download=True)
        dataset_test = CIFAR100(config.path, train=False, transform=transform, download=True)
    elif config.name == 'cifar-10-dvs':
        os.makedirs(config.path, exist_ok=True)
        dataset = CIFAR10DVS(config.path, data_type='frame', frames_number=config.time_steps, split_by='number')
        dataset_train, dataset_test = split_dataset(dataset, num_classes=config.num_classes)
    elif config.name == 'dvs-128-gesture':
        os.makedirs(config.path, exist_ok=True)
        dataset_train = DVS128Gesture(config.path, train=True, data_type='frame', frames_number=config.time_steps, split_by='number')
        dataset_test = DVS128Gesture(config.path, train=False, data_type='frame', frames_number=config.time_steps, split_by='number')
    return dataset_train, dataset_test
