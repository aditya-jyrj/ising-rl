import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import imageio.v2 as imageio

def generate_lattice(L, ratio=0.5):
    lattice = np.ones((L, L), dtype=np.int8)
    lattice[np.random.random((L, L)) > ratio] = -1

    return lattice

def get_energy(lattice, H=0.0):
    # nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False

    interaction = -lattice * convolve(lattice, kern, mode='wrap') 
    field = -H * lattice
    
    return interaction.sum() / 2 + field.sum()


# numba speeds up the iterations to almost C speed
@njit 
def metropolis(lattice, steps, BJ, H, energy, save_every=None):
    L = lattice.shape[0]
    if L != lattice.shape[1]:
        raise ValueError("metropolis is configured for only square lattices")

    if save_every is not None:
        n_frames = steps // save_every + 1
        frames = np.empty((n_frames, L, L), dtype=np.int8)
        frames[0] = lattice
        frame_idx = 1
    else:
        frames = None

    net_spins = np.zeros(steps)
    net_energies = np.zeros(steps)

    # 1. call current state mu
    lattice = lattice.copy()
    net_spin = lattice.sum()

    for t in range(0, steps):

        # 2. pick random spins and flip it
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)

        spin = lattice[x,y]

        # compute change in energy
        neighbour_sum = (
              lattice[(x - 1) % L, y]
            + lattice[(x + 1) % L, y]
            + lattice[x, (y - 1) % L]
            + lattice[x, (y + 1) % L]
        )
        dE = 2 * spin * (neighbour_sum + H)

        # 3 / 4. change state with designated probabilities
        if dE <= 0 or np.random.random() < np.exp(-BJ * dE):
            lattice[x, y] *= -1
            energy += dE
            net_spin -= 2 * spin
            
        net_spins[t] = net_spin
        net_energies[t] = energy

        if save_every is not None and (t+1) % save_every == 0:
            frames[frame_idx] = lattice
            frame_idx += 1
            
    return lattice, net_spins, net_energies, frames


def make_gif(frames, filename="ising_evolution.gif", fps=16):
    images = []

    for frame in frames:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(frame, cmap="binary", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        # draw canvas and extract image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(image)

        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps)