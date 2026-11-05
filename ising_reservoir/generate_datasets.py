import os
import time
import numpy as np
from sklearn.datasets import fetch_openml

import utils as ut

# Physical parameters
BJ_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
Balpha = 1.0

# Monte Carlo parameters
sweeps = 20000
n_samples = 100
interval = 50

# Load MNIST
X, y = fetch_openml(
    "mnist_784", version=1,
    return_X_y=True, as_frame=False
)

X = X.astype(float) / 255
y = y.astype(int)

n_images = len(X)

os.makedirs("datasets", exist_ok=True)

for BJ in BJ_values:

    filename = f"datasets/BJ{BJ}_Balpha{Balpha}_{n_images}_{sweeps}_{n_samples}_{interval}.npz"

    # Avoid regenerating completed datasets
    if os.path.exists(filename):
        print(f"Skipping BJ = {BJ}: dataset already exists")
        continue

    print(f"\nGenerating dataset for BJ = {BJ}")

    start = time.perf_counter()

    X_ising = np.zeros((n_images, 784), dtype=np.float32)

    for i in range(n_images):
        X_ising[i] = ut.ising_features(
            X[i], BJ, Balpha,
            sweeps=sweeps,
            n_samples=n_samples,
            interval=interval
        )

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start
            remaining = elapsed / (i + 1) * (n_images - i - 1)

            print(
                f"BJ={BJ}: {i+1}/{n_images} images | "
                f"Elapsed: {elapsed/3600:.2f} h | "
                f"Remaining: {remaining/3600:.2f} h",
                flush=True
            )

    elapsed = time.perf_counter() - start

    np.savez_compressed(filename, X=X_ising, y=y[:n_images], BJ=BJ, Balpha=Balpha, sweeps=sweeps, n_samples=n_samples, interval=interval, elapsed=elapsed)

    print(f"Completed BJ = {BJ}")
    print(f"Time taken: {elapsed / 3600:.2f} hours")

print("\nAll datasets generated.")