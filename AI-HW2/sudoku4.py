import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate(N, num_clues):
    n = int(np.sqrt(N))
    ran = range(n)

    rows = [g * n + r for g in random.sample(ran, n) for r in random.sample(ran, n)]
    cols = [g * n + c for g in random.sample(ran, n) for c in random.sample(ran, n)]
    nums = random.sample(range(1, N + 1), N)

    S = np.array([[nums[(n * (r % n) + r // n + c) % N] for c in cols] for r in rows])
    indices = random.sample(range(N ** 2), num_clues)
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
                for k in range(j + 1, N):
                    if subgrid[k] not in clues:
                        succ = np.copy(board)
                        succ[subgrid[j]], succ[subgrid[k]] = succ[subgrid[k]], succ[subgrid[j]]
                        successors.append(succ)
    return successors

def num_errors(board):
    N = board.shape[0]
    digits = range(1, N + 1)
    errors = sum(N - np.sum(np.in1d(digits, board[i])) for i in range(N))
    errors += sum(N - np.sum(np.in1d(digits, board[:, i])) for i in range(N))
    return errors

def simulated_annealing(board, clues, startT, decay, tol=1e-4):
    current_error = np.inf
    T = startT / decay
    while current_error > 0 and T >= tol:
        T *= decay
        all_succ = successors(board, clues)
        if not all_succ:
            break
        succ = random.choice(all_succ)
        succ_error = num_errors(succ)
        if succ_error < current_error or np.random.random() < np.exp((current_error - succ_error) / T):
            board = succ
            current_error = succ_error
    return board

def run_experiment(startT_values, n=9, c=40, decay=0.99, batch_size=30):
    results = {}

    for startT in startT_values:
        final_errors = []
        for _ in range(batch_size):
            board, clues = generate(n, c)
            sol = simulated_annealing(initialize(board), clues, startT, decay)
            final_errors.append(num_errors(sol))

        results[startT] = final_errors

        # Generate and save histogram
        plt.figure()
        plt.hist(final_errors, bins=range(max(final_errors) + 2), edgecolor="black", align="left")
        plt.xlabel("Final Errors")
        plt.ylabel("Frequency")
        plt.title(f"Final Error Distribution (startT={startT})")
        plt.savefig(f"plt{6 + startT_values.index(startT)}.png")
        plt.close()

    return results

def main():
    parser = argparse.ArgumentParser(description="Sudoku Simulated Annealing Experiment")
    parser.add_argument("-n", type=int, default=9, help="Grid size (default 9x9)")
    parser.add_argument("-c", type=int, default=40, help="Number of clues (default 40)")
    parser.add_argument("-b", type=int, default=30, help="Batch size (default 30)")
    args = parser.parse_args()

    startT_values = [1, 10, 100]
    run_experiment(startT_values, args.n, args.c, decay=0.99, batch_size=args.b)

if __name__ == "__main__":
    main()
