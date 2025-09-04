// main.cu
// Host-side workflow for matrix multiplication (memory, launch, validation, timing)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>  // for CPU timing

// API declarations for kernel launchers
extern "C" void matmul_naive_cuda(const float *d_A, const float *d_B, float *d_C, int N);
extern "C" void matmul_tiled_cuda(const float *d_A, const float *d_B, float *d_C, int N);

int N = 1024; // Matrix dimension

void check_result(const float *ref, const float *out, int N, const char *msg) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(ref[i] - out[i]) > 1e-3) {
            printf("%s mismatch at %d: ref=%f out=%f\n", msg, i, ref[i], out[i]);
            return;
        }
    }
    printf("%s PASSED\n", msg);
}

int main() {
    size_t size = N * N * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // CPU timing start
    clock_t cpu_start = clock();

    // Reference CPU multiplication
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float val = 0.0f;
            for (int k = 0; k < N; ++k)
                val += h_A[row * N + k] * h_B[k * N + col];
            h_ref[row * N + col] = val;
        }
    }

    // CPU timing end and duration in ms
    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Matrix Multiply time: %.3f ms\n", cpu_time_ms);

    // Device memory allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time_ms;

    // Launch naive kernel and time it
    cudaEventRecord(start, 0);
    matmul_naive_cuda(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // Synchronize and copy result
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU Naive Matrix Multiply time: %.3f ms\n", gpu_time_ms);
    check_result(h_ref, h_C, N, "Naive Matrix Multiply");

    // Launch tiled kernel and time it
    cudaEventRecord(start, 0);
    matmul_tiled_cuda(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU Tiled Matrix Multiply time: %.3f ms\n", gpu_time_ms);
    check_result(h_ref, h_C, N, "Tiled Matrix Multiply");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
