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

def train(X, y, learning_rate=0.3, epochs=500, lam=0):
    N, num_features = X.shape
    num_classes = 10

    W = np.zeros((num_features, num_classes))
    b = np.zeros(num_classes)

    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1

    for epoch in range(epochs):
        Z = X @ W + b
        P = softmax(Z)

        error = P - Y
        dW = X.T @ error / N + lam * W
        db = np.mean(error, axis=0)

        W -= learning_rate * dW
        b -= learning_rate * db

    return W, b

def predict(X, W, b):
    Z = X @ W + b
    return np.argmax(Z, axis=1)