#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 5;
const double delta_r = 0.005, delta_t = 0.01;
const int lim_iter = 100000;


__device__
void _x1(double X1[5], double X0[5], double Ax) {
    X1[0] = X0[0] - delta_r * (X0[0] + X0[2] * cos(3 * M_PI / 2 - X0[3]) - Ax);
}

__device__
void _x2(double X1[5], double X0[5], double Bx) {
    X1[1] = X0[1] - delta_r *  (X0[1] + X0[2] * cos(3 * M_PI / 2 + X0[4]) - Bx);
}

__device__
void _y(double *X1, double *X0, double Ay) {
    X1[2] = X0[2] - delta_r * (X0[2] + X0[2] * sin(3 * M_PI / 2 - X0[3]) - Ay);
}

__device__
void _f1(double *X1, double *X0, double C) {
    X1[3] = X0[3] - delta_r * ((X0[3] + X0[4]) * X0[2] + (X0[1] - X0[0]) - C);
}

__device__
void _f2(double X1[5], double X0[5], double By) {
    X1[4] = X0[4] - delta_r * (X0[2] + X0[2] * sin(3 * M_PI / 2 + X0[4]) - By);
}

__global__
void step_multiple_thread(double *X0, double *X1, double Ax, double Ay, double Bx, double By, double C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    switch (i) {
    case 0:
        _x1(X1, X0, Ax);
        break;
    case 1:
        _x2(X1, X0, Bx);
        break;
    case 2:
        _y(X1, X0, Ay);
        break;
    case 3:
        _f1(X1, X0, C);
        break;
    case 4:
        _f2(X1, X0, By);
        break;    
    }
}

__global__
void step_single_thread(double *X0, double *X1, double Ax, double Ay, double Bx, double By, double C) {
    _x1(X1, X0, Ax);
    _x2(X1, X0, Bx);
    _y(X1, X0, Ay);
    _f1(X1, X0, C);
    _f2(X1, X0, By);  
}

double p = 2000, m = 100, g = 10, v = 0;
double Ax = -0.353, Ay = 0.3, Bx = 0.353, By = 0.3, C = 3 * M_PI / 8;

int main() {
    FILE *ouf = fopen("single_tread.csv", "w");
    fprintf(ouf, "step, x1, x2, y, f1, f2, Ax, Ay, Bx, By, C, 'time, s'\n");

    double *x0, *x1, *d_x0, *d_x1;
    x0 = (double*)malloc(N * sizeof(double));
    x1 = (double*)malloc(N * sizeof(double));

    cudaMalloc(&d_x0, N * sizeof(double)); 
    cudaMalloc(&d_x1, N * sizeof(double));
    
    double x0_init[] = { -0.1, 0.1, 0.0, 2.0, 2.0 };
    memcpy(x0, x0_init, N * sizeof(double));

    int st = 0;
    for (double t = 0; t <= 2.5; t += delta_t, st += 1) {
        clock_t start = clock();
        cudaMemcpy(d_x0, x0, N * sizeof(double), cudaMemcpyHostToDevice);

        for (int step = 0; step < lim_iter; ++step) {
            // step_multiple_thread<<<1, 5>>>(d_x0, d_x1, Ax, Ay, Bx, By, C);
            step_single_thread<<<1, 1>>>(d_x0, d_x1, Ax, Ay, Bx, By, C);
            cudaMemcpy(d_x0, d_x1, N * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(x1, d_x1, N * sizeof(double), cudaMemcpyDeviceToHost);
        clock_t end = clock();
        double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;

        Ay += v * delta_t;
        By = Ay;
        v += (p * (x1[1] - x1[0]) - m * g) / m * delta_t;

        fprintf(ouf, "%d, ", st);
        for (int i = 0; i < 5; ++i) {
            fprintf(ouf, "%f, ", x1[i]);
        }
        fprintf(ouf, "%f, %f, %f, %f, %f, %f\n", Ax, Ay, Bx, By, C, time_taken);
        fflush(ouf);
    }

    cudaFree(d_x0);
    cudaFree(d_x1);
    free(x0);
    free(x1);
}
