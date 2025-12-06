#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    for (long j = 0; j < N; j += 4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] +
             pA[j+2]*pB[j+2] + pA[j+3]*pB[j+3];
    return R;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <N> <repetitions>\n", argv[0]);
        return -1;
    }

    long N = atol(argv[1]);
    int reps = atoi(argv[2]);

    float *A = (float*)malloc(sizeof(float) * N);
    float *B = (float*)malloc(sizeof(float) * N);
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    struct timespec start, end;
    double total = 0.0;

    for (int r = 0; r < reps; r++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        volatile float res = dpunroll(N, A, B);
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

    free(A); free(B);
    return 0;
}
