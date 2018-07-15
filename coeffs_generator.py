import numpy as np

def uniform(n):
    return [1 for i in range(n)]

def harmonic(n, k=1):
    return [1. * k / i for i in range(1, n + 1)]

def linear(n, k=1):
    return [1. - (i * 1. * k / n) for i in range(n)]