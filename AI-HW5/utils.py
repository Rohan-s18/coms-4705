import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from gridworld_hmm import Gridworld_HMM


def loc_error(beliefs, trajectory):
    errors = []
    for i in range(len(trajectory)):
        belief = beliefs[i]
        belief[trajectory[i]] -= 1
        errors.append(np.sum(np.abs(belief)))
    return errors


def inference(shape, walls, epsilons, T, N, m, p):
    filtering_error = np.zeros((len(epsilons), T))

    for e in range(len(epsilons)):
        env = Gridworld_HMM(shape, epsilons[e], walls, p)
        cells = np.nonzero(env.grid == 0)
        indices = cells[0] * env.grid.shape[1] + cells[1]

        for n in range(N):
            trajectory = []
            observations = []
            curr = np.random.choice(indices)
            for t in range(T):
                trajectory.append(np.random.choice(env.trans.shape[0], p=env.trans[curr]))
                curr = trajectory[-1]
                observations.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))
            if m == 0:
                filtering_error[e] += loc_error(env.forward(observations), trajectory)
            elif m == 1:
                all_counts = env.particle_filter(observations)
                dist = (all_counts.T / np.sum(all_counts, axis=1)).T
                filtering_error[e] += loc_error(dist, trajectory)

    return filtering_error / N


def visualize_one_run(shape, walls, epsilon, T, m, p):
    env = Gridworld_HMM(shape, epsilon, walls, p)
    cells = np.nonzero(env.grid == 1)
    indices = cells[0] * env.grid.shape[1] + cells[1]

    trajectory = []
    observations = []
    curr = np.random.choice(indices)
    for t in range(T):
        trajectory.append(np.random.choice(env.trans.shape[0], p=env.trans[curr]))
        curr = trajectory[-1]
        observations.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))

    if m == 2:
        beliefs = env.forward(observations)
    elif m == 3:
        counts = env.particle_filter(observations)
        beliefs = (counts.T / np.sum(counts, axis=1)).T
    for i, j in walls:
        beliefs[:, i*shape[1]+j] = -1

    fig, ax = plt.subplots(1, 1)
    cmap = "summer"
    ax.imshow(np.ones(shape), cmap=cmap)
    ax.set_title(f"Estimated distribution with epsilon={epsilon}")
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()

    def update(frame):
        ax.clear()
        
        ax.set_title(f"Forward algorithm with epsilon={epsilon}")
        curr_belief = beliefs[frame].reshape(-1, shape[1])
        ax.imshow(curr_belief, cmap=cmap)
        ax.plot(trajectory[frame] % shape[1], trajectory[frame] // shape[1], 'ro')

        if m == 3:
            count = counts[frame].reshape(-1, shape[1])
            nonzeros = np.where(curr_belief > 0)
            for i in range(len(nonzeros[0])):
                text = ax.text(nonzeros[1][i], nonzeros[0][i], int(count[nonzeros[0][i]][nonzeros[1][i]]),
                               ha='center', va='center', color='black')
        
        ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    num_frames = T
    _ = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)

    plt.show()
