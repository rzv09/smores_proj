import numpy as np

def MARE(prediction, truth):
    eps = 1e-8
    return np.sum((np.abs(prediction - truth)) / np.abs(truth + eps)) / len(truth)
