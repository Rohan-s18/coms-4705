import numpy as np
import time
import sys

def dp(N, A, B):
    R = 0.0
    for j in range(N):
        R += A[j] * B[j]
    return R

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dp4.py <N> <repetitions>")
        sys.exit(1)

    N = int(sys.argv[1])
    reps = int(sys.argv[2])

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    times = []
    for r in range(reps):
        start = time.perf_counter()
        _ = dp(N, A, B)
        end = time.perf_counter()
        times.append(end - start)

    avg = np.mean(times[reps//2:])
    bytes_ = 2.0 * N * A.itemsize
    bandwidth = bytes_ / avg / 1e9
    flops = (2.0 * N) / avg / 1e9

    print(f"N: {N} <T>: {avg:.6f} sec B: {bandwidth:.3f} GB/sec F: {flops:.3f} GFLOP/sec")
