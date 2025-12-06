#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

float bdp(long N, float *pA, float *pB) {
    return cblas_sdot(N, pA, 1, pB, 1);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <N> <repetitions>\n", argv[0]);
        return -1;
    }

    long N = atol(argv[1]);
    int reps = atoi(argv[2]);

    float *A = (float*)mkl_malloc(sizeof(float) * N, 64);
    float *B = (float*)mkl_malloc(sizeof(float) * N, 64);
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    struct timespec start, end;
    double total = 0.0;

    for (int r = 0; r < reps; r++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        volatile float res = bdp(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double t = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1e9;
        if (r >= reps / 2) total += t;
    }

    double avg = total / (reps / 2);
    double bytes = 2.0 * N * sizeof(float);
    double bandwidth = bytes / avg / 1e9;
    double flops = (2.0 * N) / avg / 1e9;

    printf("N: %ld <T>: %lf sec B: %lf GB/sec F: %lf GFLOP/sec\n",
           N, avg, bandwidth, flops);

    mkl_free(A); mkl_free(B);
    return 0;
}
