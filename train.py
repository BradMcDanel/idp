from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import pickle
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import data_parallel
cudnn.benchmark = True 

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

import datasets
import net

def train(model, train_loader, optimizer):
    model.train()
    for data, target in tqdm(train_loader, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        prediction = data_parallel(model, data)
        loss = F.cross_entropy(prediction, target)
        loss.backward()
        optimizer.step()

def test(model, test_loader):
    model.eval()
    num_correct, model_loss = 0, 0
    for data, target in tqdm(test_loader, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        prediction = data_parallel(model, data)
        loss = F.cross_entropy(prediction, target, size_average=False).item()
        pred = prediction.data.max(1, keepdim=True)[1]
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct
        model_loss += loss

    N = len(test_loader.dataset)
    print('Test  Loss:: {}'.format(model_loss / N))
    print('Test  Acc.:: {}'.format(100. * (num_correct / N)))

    return model_loss / N


def train_model(model, model_path, train_loader, test_loader, lr, epochs):
    figure_path = model_path.split('/')
    figure_path[0] = 'figures'
    figure_path = '/'.join(figure_path) + '.png'
    optimizer = optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for epoch in range(epochs):
        print('[Epoch {}]'.format(epoch+1))
        train(model, train_loader, optimizer)
        test(model, test_loader)
        model.idp_percent(0.5)
        test(model, test_loader)
        model.idp_percent(1.0)
        torch.save(model, model_path)
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDP Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--aug', default='-')
    parser.add_argument('--output', default='models/model.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda, args.aug)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().next()
    B, C, W, H = x.shape
    model = net.MLP(C*W*H, 10, coeff_type='linear')

    if args.cuda:
        model = model.cuda()

    ps = filter(lambda x: x.requires_grad, model.parameters())
    train_model(model, args.output, train_loader, test_loader,
                args.lr, args.epochs)