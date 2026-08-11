import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

import imageio.v2 as imageio
from IPython.display import Image

def get_energy(lattice, H=0.0):
    # nearest neighbours summation
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False

    interaction = -lattice * convolve(lattice, kern, mode='constant', cval=0) 
    field = -H * lattice
    
    return interaction.sum() / 2 + field.sum()

# numba speeds up the iterations to almost C speed
@numba.njit 
def metropolis(lattice, steps, BJ, H, energy, save_every):
    N = lattice.shape[0]
    if N != lattice.shape[1]:
        raise ValueError("metropolis is configured for only square lattices")

    # 1. call current state mu
    lattice = lattice.copy()
    net_spins = np.zeros(steps-1)
    net_energies = np.zeros(steps-1)

    n_frames = steps // save_every + 1
    frames = np.empty((n_frames, N, N), dtype=np.int8)

    frames[0] = lattice
    frame_idx = 1

    for t in range(0, steps-1):

        # 2. pick random spins and flip it
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)

        spin_i = lattice[x,y]
        spin_f = spin_i * -1 # proposed sign flip

        # compute change in energy
        E_i = -H * spin_i
        E_f = -H * spin_f
        if x>0:
            E_i += -spin_i*lattice[x-1,y]
            E_f += -spin_f*lattice[x-1,y]
        if x<N-1:
            E_i += -spin_i*lattice[x+1,y]
            E_f += -spin_f*lattice[x+1,y]
        if y>0:
            E_i += -spin_i*lattice[x,y-1]
            E_f += -spin_f*lattice[x,y-1]
        if y<N-1:
            E_i += -spin_i*lattice[x,y+1]
            E_f += -spin_f*lattice[x,y+1]

        # 3 / 4. change state with designated probabilities
        dE = E_f-E_i
        if dE <= 0:
            lattice[x, y] = spin_f
            energy += dE
        elif np.random.random() < np.exp(-BJ * dE):
            lattice[x, y] = spin_f
            energy += dE
            
        net_spins[t] = lattice.sum()
        net_energies[t] = energy

        if (t+1) % save_every == 0:
            frames[frame_idx] = lattice
            frame_idx += 1
            
    return lattice, net_spins, net_energies, frames

def make_gif(frames, filename="ising_evolution.gif", fps=10):
    images = []

    for frame in frames:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(frame, cmap="binary", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        # draw canvas and extract image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(image)

        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps)