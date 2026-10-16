import numpy as np

def softmax(Z):
    # Z is a vector of length 10 containing z_c for all 10 digits
    # we subtract np.max(Z, ...) from Z to avoid floating values becoming too large

    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy(P, y):
    N = len(y)
    correct_probabilities = P[np.arange(N), y]
    return -np.mean(np.log(correct_probabilities))