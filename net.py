from __future__ import print_function
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def all_one_coeffs(n):
    return torch.Tensor([1 for i in range(n)])

def linear_coeffs(n, k=1):
    return torch.Tensor([1. - (i * 1. * k / n) for i in range(n)])

def harmonic_coeffs(n, k=1):
    return torch.Tensor([1. * k / i for i in range(1, n + 1)])

class IDPLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, coeffs):
        ctx.save_for_backward(input, weight, bias, coeffs)
        output = input.mm((coeffs*weight).t())
        output += (torch.dot(coeffs.view(-1),bias)).unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, coeffs = ctx.saved_variables
        grad_input = grad_weight = grad_bias = grad_coeffs = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(coeffs*weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_coeffs


linear = IDPLinearFunc.apply


class IDPLinear(nn.Module):
    def __init__(self, in_size, out_size, coeff_type='all-one'):
        super(IDPLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size).fill_(0.01))
        self.out_size = out_size
        self.active = out_size
        nn.init.xavier_uniform_(self.weight)
        
        if coeff_type == 'all-one':
            self.register_buffer('coeffs', all_one_coeffs(out_size).view(-1, 1))
        elif coeff_type == 'linear':
            self.register_buffer('coeffs', linear_coeffs(out_size).view(-1, 1))
        elif coeff_type == 'harmonic':
            self.register_buffer('coeffs', harmonic_coeffs(out_size).view(-1, 1))
    
    def forward(self, x):
        coeffs = self.coeffs.clone()
        if self.active < self.out_size:
            coeffs[self.active:,0] = 0
        return linear(x, self.weight, self.bias, coeffs)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, coeff_type='all-one'):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 100), 
            nn.ReLU(inplace=True),
            IDPLinear(100, 100, coeff_type=coeff_type), 
            nn.ReLU(inplace=True),
            nn.Linear(100, out_size), 
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = self.model(x)
        return h
    
    def idp_percent(self, percent):
        for layer in self.model:
            if type(layer) == IDPLinear:
                layer.active = int(percent*layer.out_size)