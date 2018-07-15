import argparse
import os
from tqdm import tqdm

import torch
from torch.nn.parallel import data_parallel
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import datasets


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

    return  100.* (num_correct / N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDP Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--models', default='models/', help='model directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()


    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().next()
    B, C, W, H = x.shape

    coeff_types = ['all-one', 'linear', 'harmonic']
    idp_pcts = np.linspace(0.1, 1, 16).tolist()
    acc_dict = {}
    for coeff_type in coeff_types:
        model = torch.load(os.path.join(args.models, coeff_type + '.pth'))
        model.cuda()
        acc_dict[coeff_type] = []
        for idp_pct in idp_pcts:
            model.idp_percent(idp_pct)
            acc = test(model, test_loader)
            acc_dict[coeff_type].append(acc)

    for coeff_type in coeff_types:
        plt.plot(idp_pcts, acc_dict[coeff_type], '-o', label=coeff_type)
    plt.grid()
    plt.legend(loc=0)
    plt.ylim((95, 100))
    plt.xlabel('IDP (%)')
    plt.ylabel('Classification Accuracy (%)')
    plt.savefig('figures/mlp.png', dpi=300, bbox_inches='tight')
    plt.clf()