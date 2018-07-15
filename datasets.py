import os

from torchvision import datasets, transforms
import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length / 2, 0, h).astype('i')
            y2 = np.clip(y + self.length / 2, 0, h).astype('i')
            x1 = np.clip(x - self.length / 2, 0, w).astype('i')
            x2 = np.clip(x + self.length / 2, 0, w).astype('i')
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class FashionMNIST(datasets.MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

def get_dataset(dataset_root, dataset, batch_size, is_cuda=True, aug='+'):
    if dataset == 'mnist':
        train, train_loader, test, test_loader = get_mnist(dataset_root, batch_size, is_cuda, aug)
    elif dataset == 'fashion-mnist':
        train, train_loader, test, test_loader = get_fashion_mnist(dataset_root, batch_size, is_cuda, aug)
    elif dataset == 'cifar10':
        train, train_loader, test, test_loader = get_cifar10(dataset_root, batch_size, is_cuda, aug)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader

def get_mnist(dataset_root, batch_size, is_cuda=True, aug='+'):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]))
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader

def get_fashion_mnist(dataset_root, batch_size, is_cuda=True, aug='+'):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = FashionMNIST(os.path.join(dataset_root, 'fashion_mnist'), train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]))
    test = FashionMNIST(os.path.join(dataset_root, 'fashion_mnist'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=False, **kwargs)
    
    return train, train_loader, test, test_loader


def get_cifar10(dataset_root, batch_size, is_cuda=True, aug='+'):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    if aug == '-':
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ] 
    elif aug == '+':
        transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    elif aug == '++':
        transform = [
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]

    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True, download=True,
                        transform=transforms.Compose(transform))
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                        ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=False, **kwargs)
 
    return train, train_loader, test, test_loader