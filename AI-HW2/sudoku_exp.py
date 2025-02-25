import random
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate(N, num_clues):
    n = int(np.sqrt(N))
    ran = range(n)
    rows = [g * n + r for g in sample(ran, n) for r in sample(ran, n)]
    cols = [g * n + c for g in sample(ran, n) for c in sample(ran, n)]
    nums = sample(range(1, N + 1), N)
    S = np.array([[nums[(n * (r % n) + r // n + c) % N] for c in cols] for r in rows])
    indices = sample(range(N ** 2), num_clues)
    values = S.flatten()[indices]
    empty_board = np.zeros(N ** 2, dtype=int)
    empty_board[indices] = values
    board = np.reshape(empty_board, (N, N))
    clues = [(i // N, i % N) for i in indices]
    return board, clues

def initialize(board):
    N = board.shape[0]
    n = int(N ** 0.5)
    for i in range(N):
        subgrid = board[(i // n) * n:(i // n) * n + n, (i % n) * n:(i % n) * n + n]
        for j in range(1, N + 1):
            if j not in subgrid:
                idx = np.argwhere(subgrid == 0)[0]
                subgrid[idx[0], idx[1]] = j
        board[(i // n) * n:(i // n) * n + n, (i % n) * n:(i % n) * n + n] = subgrid
    return board

def successors(board, clues):
    N = board.shape[0]
    n = int(N ** 0.5)
    successors = []
    for i in range(N):
        subgrid = [((i // n) * n + j, (i % n) * n + k) for j in range(n) for k in range(n)]
        for j in range(N):
            if subgrid[j] not in clues:
                for k in range(j+1, N):
                    if subgrid[k] not in clues:
                        succ = np.copy(board)
                        succ[subgrid[j]], succ[subgrid[k]] = succ[subgrid[k]], succ[subgrid[j]]
                        successors.append(succ)
    return successors

def num_errors(board):
    N = board.shape[0]
    digits = range(1, N+1)
    errors = sum(N - np.sum(np.in1d(digits, board[i])) for i in range(N))
    errors += sum(N - np.sum(np.in1d(digits, board[:, i])) for i in range(N))
    return errors

def simulated_annealing(board, clues, startT, decay, tol=1e-4):
    current_error = np.inf
    T = startT / decay
    error_history = []
    while current_error > 0 and T >= tol:
        T *= decay
        all_succ = successors(board, clues)
        if not all_succ: break
        succ = random.choice(all_succ)
        succ_error = num_errors(succ)
        if succ_error < current_error or np.random.random() < np.exp((current_error - succ_error) / T):
            board = succ
            current_error = succ_error
        error_history.append(current_error)
    return board, error_history

def plot_error_history(error_history, title):
    plt.figure()
    plt.plot(error_history, label="Error")
    plt.xlabel("Iteration")
    plt.ylabel("Errors")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_error_histogram(errors, title):
    plt.figure()
    plt.hist(errors, bins=range(max(errors)+2), align='left', rwidth=0.8)
    plt.xlabel("Final Errors")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def run_experiments():
    # params = [(9, 40, 100, 0.5, 30), (9, 40, 10, 0.5, 30), (9, 40, 1, 0.5, 30)]
    params = [(9, 40, 100, 1.5, 30)]
    for n, c, s, d, b in params:
        final_errors = []
        for _ in range(b):
            board, clues = generate(n, c)
            sol, _ = simulated_annealing(initialize(board), clues, s, d)
            final_errors.append(num_errors(sol))
        plot_error_histogram(final_errors, f"Final Error Distribution (s={s}, d={d}, b={b})")

if __name__ == "__main__":
    run_experiments()
