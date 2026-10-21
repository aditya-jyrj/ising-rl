import numpy as np
from numba import njit


# LOGISTIC REGRESSION FUNCTIONS

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

def image_to_magnetic_field(image, alpha_over_J):
    return alpha_over_J * (2 * image - 1)



# ISING METROPOLIS FUNCTIONS

def generate_lattice(L, ratio=0.5):
    lattice = np.ones((L, L), dtype=np.int8)
    lattice[np.random.random((L, L)) > ratio] = -1

    return lattice

@njit 
def metropolis(lattice, sweeps, T, H):
    L = lattice.shape[0]
    lattice = lattice.copy()

    for _ in range(sweeps * L * L):
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)

        spin = lattice[x,y]

        neighbour_sum = (
              lattice[(x - 1) % L, y]
            + lattice[(x + 1) % L, y]
            + lattice[x, (y - 1) % L]
            + lattice[x, (y + 1) % L]
        )
        dE = 2 * spin * (neighbour_sum + H[x, y])

        # 3 / 4. change state with designated probabilities
        if dE <= 0 or np.random.random() < np.exp(-dE / T):
            lattice[x, y] *= -1
            
    return lattice

def structure_factor(lattice):
    N = lattice.size
    spin_k = np.fft.fft2(lattice)
    S_k = np.abs(spin_k)**2 / N

    S_k = np.fft.fftshift(S_k)
    S_k[S_k.shape[0] // 2, S_k.shape[1] // 2] = 0

    return S_k