"""
author: rohan singh (rs4607)
coding assignment submission (problem 4) for hw 5 coms 4701
"""

import numpy as np
import numpy.typing as npt

class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = [], num_particles: int = 30):
        if walls:
            self.grid = np.ones(size)
            for cell in walls:
                self.grid[cell] = 0
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = (self.grid / np.sum(self.grid)).flatten()

        self.epsilon = epsilon
        self.particles = np.random.choice(len(self.init), size=num_particles, p=self.init)
        self.weights = np.ones(num_particles)

        self.trans = self.initT()
        self.obs = self.initO(self.epsilon)

    def neighbors(self, cell):
        i, j = cell
        m, n = self.grid.shape
        adjacent = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = [(i, j)]
        for a1, a2 in adjacent:
            if 0 <= a1 < m and 0 <= a2 < n and self.grid[a1, a2] == 1:
                neighbors.append((a1, a2))
        return neighbors


    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        N = self.grid.size
        T = np.zeros((N, N))

        # Fill transition matrix
        for i in range(N):
            neighbors = self.neighbors((i // self.grid.shape[1], i % self.grid.shape[1]))
            num_neighbors = len(neighbors)
            for ni in neighbors:
                ni_idx = ni[0] * self.grid.shape[1] + ni[1]
                T[i, ni_idx] = 1 / num_neighbors

        # Ensure rows sum to 1
        row_sums = T.sum(axis=1)
        T = T / row_sums[:, np.newaxis]

        return T

    def initO(self, epsilon):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        N = self.grid.size
        O = np.zeros((16, N))

        for i in range(N):
            correct_obs = self.get_correct_observation(i)
            for e in range(16):
                d = self.calculate_discrepancy(correct_obs, e)
                O[e, i] = (1 - epsilon) ** (4 - d) * epsilon ** d

        return O

    def get_correct_observation(self, i):
        """
        Generate the correct observation for a given state `i`.
        """
        x, y = divmod(i, self.grid.shape[1])
        adjacent = self.neighbors((x, y))
        obs_value = 0
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # NESW order
        for idx, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if (nx, ny) in adjacent:
                obs_value |= (1 << (3 - idx))  # Set the bit
        return obs_value

    def calculate_discrepancy(self, correct_obs, observed):
        """
        Calculate the number of bits that differ between the correct observation and observed.
        """
        return bin(correct_obs ^ observed).count('1')


    """
    4.3. Forward algorithm
    """

    def forward(self, observations: list[int]):
        """Perform forward algorithm over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        T = len(observations)
        belief_state = np.zeros((T, self.grid.size))

        belief_state[0, :] = self.init

        for t in range(1, T):
            for state in range(self.grid.size):
                # Compute belief at state `state` using transition and observation probabilities
                belief_state[t, state] = np.sum(belief_state[t - 1, :] * self.trans[:, state] * self.obs[observations[t], :])

            # Normalize the belief state
            belief_state[t, :] /= np.sum(belief_state[t, :])

        return belief_state


    """
    4.4. Particle filter
    """

    def transition(self):
        """
        Sample the transition matrix for all particles.
        Update self.particles in place.
        """
        N = len(self.particles)
        for i in range(N):
            curr_state = self.particles[i]
            next_state = np.random.choice(self.grid.size, p=self.trans[curr_state, :])
            self.particles[i] = next_state

    def observe(self, observation):
        """
        Compute the weights for all particles.
        Update self.weights in place.
        Args:
          obs (int): Integer observation value.
        """
        for i in range(len(self.particles)):
            state = self.particles[i]
            self.weights[i] *= self.obs[observation, state]

    def resample(self):
        """
        Resample all particles.
        Update self.particles and self.weights in place.
        """
        N = len(self.particles)
        self.weights /= np.sum(self.weights)  # Normalize the weights
        indices = np.random.choice(N, size=N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(N)

    def particle_filter(self, observations: list[int]):
        """Apply particle filter over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Counts of particles in each state at each timestep.
        """
        T = len(observations)
        particle_counts = np.zeros((T, self.grid.size))

        for t in range(T):
            self.transition()
            self.observe(observations[t])
            self.resample()

            # Count particles in each state
            for i in range(self.grid.size):
                particle_counts[t, i] = np.sum(self.particles == i)

        return particle_counts
