from torchvision import datasets, transforms


general_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


def create_dataset(name, path):
    if name == 'cifar10':
        dataset_train = datasets.CIFAR10(path, train=True, transform=general_transform, download=True)
        dataset_test = datasets.CIFAR10(path, train=False, transform=general_transform, download=True)
    elif name == 'cifar100':
        dataset_train = datasets.CIFAR100(path, train=True, transform=general_transform, download=True)
        dataset_test = datasets.CIFAR100(path, train=False, transform=general_transform, download=True)
    return dataset_train, dataset_test
