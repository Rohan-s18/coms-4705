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

def simulated_annealing(board, clues, startT, decay, tol=1e-4, track_errors=False):
    T = startT / decay
    current_error = num_errors(board)
    error_history = [current_error]

    while current_error > 0 and T >= tol:
        T *= decay
        all_succ = successors(board, clues)
        if not all_succ: break

        succ = random.choice(all_succ)
        succ_error = num_errors(succ)
        if succ_error < current_error or np.random.random() < np.exp((current_error - succ_error) / T):
            board, current_error = succ, succ_error

        if track_errors:
            error_history.append(current_error)

    return board, error_history

def plot_error_history(error_history, title="Error History"):
    plt.figure(figsize=(8, 5))
    plt.plot(error_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Iterations")
    plt.ylabel("Number of Errors")
    plt.title(title)
    plt.grid()
    plt.show()

def plot_histogram(error_list, title="Final Errors Histogram"):
    plt.figure(figsize=(8, 5))
    plt.hist(error_list, bins=range(min(error_list), max(error_list)+2), edgecolor='black', alpha=0.75)
    plt.xlabel("Final Number of Errors")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid()
    plt.show()

#def main_2():
#    errs = [2, 2, 0, 0, 2, 0, 6, 2, 2, 2, 4, 0, 0, 2, 2, 4, 2, 0, 0, 2, 2, 0, 2, 2, 4, 2, 0, 2, 2, 0]
#    plot_histogram

def main():
    parser = argparse.ArgumentParser(description="Sudoku Solver with Simulated Annealing")
    parser.add_argument("-n", required=True, type=int, help="Grid size (e.g., 9 for 9x9)")
    parser.add_argument("-c", required=True, type=int, help="Number of clues")
    parser.add_argument("-s", default=100, type=float, help="Starting temperature (default 100)")
    parser.add_argument("-d", default=0.5, type=float, help="Decay rate (default 0.5)")
    parser.add_argument("-b", type=int, help="Batch size (number of searches)")
    args = parser.parse_args()

    if args.b is None:
        board, clues = generate(args.n, args.c)
        print("Sudoku puzzle:\n", board, "\n")
        sol, error_history = simulated_annealing(initialize(board), clues, args.s, args.d, track_errors=True)
        print(sol)
        print("Final errors:", num_errors(sol))
        plot_error_history(error_history, f"Error History (Decay={args.d}, StartT={args.s})")

    else:
        final_errors = []
        for _ in range(args.b):
            board, clues = generate(args.n, args.c)
            sol, _ = simulated_annealing(initialize(board), clues, args.s, args.d)
            final_errors.append(num_errors(sol))

        print("Final errors after batch run:", final_errors)
        plot_histogram(final_errors, f"Final Errors Histogram (Decay={args.d}, StartT={args.s}, Batch={args.b})")

if __name__ == "__main__":
    main()
