import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import imageio.v2 as imageio

def generate_lattice(L):
    return np.random.uniform(0, 2*np.pi, (L, L))


def get_energy(lattice, H, phi):
    right = np.roll(lattice, -1, axis=1) # right: 2D array of angles of the spin immediately to the right 
    down = np.roll(lattice, -1, axis=0)  # down:  2D array of angles of the spin immediately below 

    horizontal_energy = -np.cos(lattice - right).sum()
    vertical_energy = -np.cos(lattice - down).sum()

    field_energy = -H * np.cos(lattice - phi).sum()

    return horizontal_energy + vertical_energy + field_energy


@njit
def metropolis(lattice, steps, BJ, H, phi, energy, delta, save_every=None):
    L = lattice.shape[0]
    if L != lattice.shape[1]:
        raise ValueError("metropolis is configured for only square lattices")

    if save_every is not None:
        n_frames = steps // save_every + 1
        frames = np.empty((n_frames, L, L))
        frames[0] = lattice
        frame_idx = 1
    else:
        frames = None

    total_spins_x = np.zeros(steps+1)
    total_spins_y = np.zeros(steps+1)
    total_energies = np.zeros(steps+1)

    lattice = lattice.copy()
    total_spin_x = np.cos(lattice).sum()
    total_spin_y = np.sin(lattice).sum()

    total_spins_x[0] = total_spin_x
    total_spins_y[0] = total_spin_y
    total_energies[0] = energy

    for t in range(1, steps+1):

        # pick random spin and rotate it by some random angle within [-delta, delta]
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)

        old_angle = lattice[x, y]
        new_angle = (old_angle + np.random.uniform(-delta, delta)) % (2*np.pi)

        neighbours = (
            lattice[(x - 1) % L, y],
            lattice[(x + 1) % L, y],
            lattice[x, (y - 1) % L],
            lattice[x, (y + 1) % L]
        )

        old_energy = -H * np.cos(old_angle - phi)
        new_energy = -H * np.cos(new_angle - phi)

        for neighbour in neighbours:
            old_energy -= np.cos(old_angle - neighbour)
            new_energy -= np.cos(new_angle - neighbour)

        dE = new_energy - old_energy

        if dE <= 0 or np.random.random() < np.exp(-BJ * dE):
            lattice[x, y] = new_angle
            energy += dE

            # update total spin
            total_spin_x += np.cos(new_angle) - np.cos(old_angle)
            total_spin_y += np.sin(new_angle) - np.sin(old_angle)

        total_spins_x[t] = total_spin_x
        total_spins_y[t] = total_spin_y
        total_energies[t] = energy

        if save_every is not None and t % save_every == 0:
            frames[frame_idx] = lattice
            frame_idx += 1
            
    return lattice, total_spins_x, total_spins_y, total_energies, frames


def make_gif(frames, filename="xy_evolution.gif", fps=16, mode="colours"):
    images = []

    L = frames.shape[1]
    x, y = np.meshgrid(np.arange(L), np.arange(L))

    for frame in frames:
        fig, ax = plt.subplots(figsize=(6, 6))

        if mode == "arrows":
            spin_x = np.cos(frame)
            spin_y = np.sin(frame)

            ax.quiver(x, y, spin_x, spin_y, pivot="mid")

        elif mode == "colours":
            ax.imshow(frame, cmap="twilight", vmin=0, vmax=2*np.pi)

        else:
            raise ValueError("mode must be 'arrows' or 'colours'")

        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(L - 0.5, -0.5)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(image.copy())

        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps)