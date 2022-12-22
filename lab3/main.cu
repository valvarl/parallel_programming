#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

const double G = 6.6743e-10;
const int N = 100;
const int NUM_THREADS = 1;
// const int NUM_THREADS = N < 1024 ? N : 1024;
const int BLOCK = (N + NUM_THREADS - 1) / NUM_THREADS;
const double T = 10;
const double FRAMES = 400;
const double delta_t = T / FRAMES;

double randn(double mu, double sigma) {
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;
 
    if (call == 1) {
        call = !call;
        return (mu + sigma * (double) X2);
    }
 
    do {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    } while (W >= 1 || W == 0);
 
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}

double randfrom(double min, double max) {
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

__device__
void Fn(const double *r, const double *m, int n, double *out) {
    out[n] = 0;

    for (int k = n % 2; k < 2 * N; k += 2) {
        if (k != n) {
            out[n] += m[k / 2] * (r[n] - r[k]) / pow(abs(r[n] - r[k]), 3);
        }
    }

    out[n] *= -G;
}

__global__
void step_multiple_thread(const double *r, const double *v, const double *m, double *r1, double *v1) {
    for (int i = threadIdx.x * BLOCK; i < min(N, (threadIdx.x + 1) * BLOCK); ++i) {
        int n = blockIdx.x == 0 ? 2 * i : 2 * i + 1;
        Fn(r, m, n, v1);
        v1[n] += delta_t * v[n];
        r1[n] = r[n] + delta_t * v1[n];
    }
}

int main() {
    FILE *ouf = fopen("single_tread.csv", "w");
    fprintf(ouf, "step,\"time, s\"");
    for (int i = 0; i < N; ++i) {
        fprintf(ouf, ",x%d,y%d", i, i);
    }
    fprintf(ouf, "\n");
    
    double *r, *v, *m, *d_r, *d_v, *d_m, *d_r1, *d_v1;
    r = (double*)malloc(2 * N * sizeof(double));
    v = (double*)malloc(2 * N * sizeof(double));
    m = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < 2 * N; ++i) {
        r[i] = randn(0, 1);
        v[i] = 0;
        m[i / 2] = randfrom(0, 100000);
    }

    FILE *weights = fopen("single_weights.csv", "w");
    for (int i = 0; i < N; ++i) {
        fprintf(weights, "%f,", m[i]);
    }

    cudaMalloc(&d_r, 2 * N * sizeof(double)); 
    cudaMalloc(&d_v, 2 * N * sizeof(double));
    cudaMalloc(&d_r1, 2 * N * sizeof(double)); 
    cudaMalloc(&d_v1, 2 * N * sizeof(double));
    cudaMalloc(&d_m, N * sizeof(double));

    cudaMemcpy(d_r, r, 2 * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, 2 * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, N * sizeof(double), cudaMemcpyHostToDevice);

    for (int st = 0; st <= FRAMES; ++st) {
        clock_t start = clock();
        step_multiple_thread<<<2, NUM_THREADS>>>(d_r, d_v, d_m, d_r1, d_v1);
        cudaMemcpy(d_r, d_r1, 2 * N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v, d_v1, 2 * N * sizeof(double), cudaMemcpyDeviceToDevice);

        clock_t end = clock();
        double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
        cudaMemcpy(r, d_r1, 2 * N * sizeof(double), cudaMemcpyDeviceToHost);

        fprintf(ouf, "%d,%f", st, time_taken);
        for (int j = 0; j < 2 * N; ++j) {
            fprintf(ouf, ",%f", r[j]);
        }
        fprintf(ouf, "\n");
        fflush(ouf);
    }

    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_r1);
    cudaFree(d_v1);
    cudaFree(d_m);
    free(r);
    free(v);
    free(m);
}
    