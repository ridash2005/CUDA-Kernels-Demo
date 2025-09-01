#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include "vector_ops.h"

#define N 1000000         // Vector size
#define TOLERANCE 1e-5f   // Verification tolerance

#define checkCudaExit(call) do {                     \
    cudaError_t err = call;                          \
    if (err != cudaSuccess) {                        \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE);                          \
    }                                               \
} while(0)

// Enhanced verification: prints up to 5 mismatches
void verifyResultDbg(const float* ref, const float* res, int n, const char* msg, float tol) {
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(ref[i] - res[i]) > tol) {
            if (failures < 5) {
                printf("[FAILED] %s at idx %d: expected %f, got %f\n", msg, i, ref[i], res[i]);
            }
            failures++;
        }
    }
    if (failures == 0) {
        printf("[PASSED] %s\n", msg);
    } else {
        printf("[FAILED] %s with %d mismatches\n", msg, failures);
    }
}

int main() {
    int n = N;
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c || !h_ref) {
        fprintf(stderr, "Host memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    float *d_a, *d_b, *d_c;
    float *d_dot_float;
    checkCudaExit(cudaMalloc(&d_a, bytes));
    checkCudaExit(cudaMalloc(&d_b, bytes));
    checkCudaExit(cudaMalloc(&d_c, bytes));
    checkCudaExit(cudaMalloc(&d_dot_float, sizeof(float)));

    checkCudaExit(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    checkCudaExit(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    checkCudaExit(cudaMemset(d_dot_float, 0, sizeof(float)));

    cudaEvent_t start, stop;
    checkCudaExit(cudaEventCreate(&start));
    checkCudaExit(cudaEventCreate(&stop));

    // CPU timings + reference calculations
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(h_a, h_b, h_ref, n);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuAddTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    cpuStart = std::chrono::high_resolution_clock::now();
    cpuVectorSub(h_a, h_b, h_ref, n);
    cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuSubTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    cpuStart = std::chrono::high_resolution_clock::now();
    cpuVectorMul(h_a, h_b, h_ref, n);
    cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuMulTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    cpuStart = std::chrono::high_resolution_clock::now();
    double cpuDotDouble = cpuDotProductDouble(h_a, h_b, n);
    cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuDotTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    int threadsPerBlock = getOptimalBlockSize();
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float msAdd = 0.f, msSub = 0.f, msMul = 0.f, msDot = 0.f;

    // GPU Vector Add
    checkCudaExit(cudaEventRecord(start));
    launchGpuVectorAdd(d_a, d_b, d_c, n, threadsPerBlock);
    checkCudaExit(cudaGetLastError());
    checkCudaExit(cudaDeviceSynchronize());
    checkCudaExit(cudaEventRecord(stop));
    checkCudaExit(cudaEventSynchronize(stop));
    checkCudaExit(cudaEventElapsedTime(&msAdd, start, stop));
    checkCudaExit(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cpuVectorAdd(h_a, h_b, h_ref, n);
    verifyResultDbg(h_ref, h_c, n, "GPU Vector Add", TOLERANCE);

    // GPU Vector Sub
    checkCudaExit(cudaEventRecord(start));
    launchGpuVectorSub(d_a, d_b, d_c, n, threadsPerBlock);
    checkCudaExit(cudaGetLastError());
    checkCudaExit(cudaDeviceSynchronize());
    checkCudaExit(cudaEventRecord(stop));
    checkCudaExit(cudaEventSynchronize(stop));
    checkCudaExit(cudaEventElapsedTime(&msSub, start, stop));
    checkCudaExit(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cpuVectorSub(h_a, h_b, h_ref, n);
    verifyResultDbg(h_ref, h_c, n, "GPU Vector Sub", TOLERANCE);

    // GPU Vector Mul
    checkCudaExit(cudaEventRecord(start));
    launchGpuVectorMul(d_a, d_b, d_c, n, threadsPerBlock);
    checkCudaExit(cudaGetLastError());
    checkCudaExit(cudaDeviceSynchronize());
    checkCudaExit(cudaEventRecord(stop));
    checkCudaExit(cudaEventSynchronize(stop));
    checkCudaExit(cudaEventElapsedTime(&msMul, start, stop));
    checkCudaExit(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cpuVectorMul(h_a, h_b, h_ref, n);
    verifyResultDbg(h_ref, h_c, n, "GPU Vector Mul", TOLERANCE);

    // GPU Dot Product
    checkCudaExit(cudaMemset(d_dot_float, 0, sizeof(float)));
    checkCudaExit(cudaEventRecord(start));
    kDotProductFloatAtomic<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_a, d_b, d_dot_float, n);
    checkCudaExit(cudaGetLastError());
    checkCudaExit(cudaDeviceSynchronize());
    checkCudaExit(cudaEventRecord(stop));
    checkCudaExit(cudaEventSynchronize(stop));
    checkCudaExit(cudaEventElapsedTime(&msDot, start, stop));

    float gpuDotFloat = 0.f;
    checkCudaExit(cudaMemcpy(&gpuDotFloat, d_dot_float, sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nResults:\n");
    printf("GPU Dot Product (float atomicAdd): %.6f\n", gpuDotFloat);
    printf("CPU Dot Product (double precision): %.6f\n", cpuDotDouble);

    double diff = fabs(static_cast<double>(gpuDotFloat) - cpuDotDouble);
    double tolerance = 1e-2 * fabs(cpuDotDouble); // 1% relative tolerance

    if (diff <= tolerance) {
        printf("[PASSED] Dot Product Comparison (within tolerance)\n");
    } else {
        printf("[FAILED] Dot Product Comparison (difference too large)\n");
        printf("Difference: %.6f exceeds tolerance %.6f\n", diff, tolerance);
    }

    printf("\nTiming Results (milliseconds):\n");
    printf("CPU Vector Add: %.3f ms\n", cpuAddTime);
    printf("GPU Vector Add: %.3f ms\n", msAdd);
    printf("CPU Vector Sub: %.3f ms\n", cpuSubTime);
    printf("GPU Vector Sub: %.3f ms\n", msSub);
    printf("CPU Vector Mul: %.3f ms\n", cpuMulTime);
    printf("GPU Vector Mul: %.3f ms\n", msMul);
    printf("CPU Dot Product: %.3f ms\n", cpuDotTime);
    printf("GPU Dot Product: %.3f ms\n", msDot);

    checkCudaExit(cudaFree(d_a));
    checkCudaExit(cudaFree(d_b));
    checkCudaExit(cudaFree(d_c));
    checkCudaExit(cudaFree(d_dot_float));

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    checkCudaExit(cudaEventDestroy(start));
    checkCudaExit(cudaEventDestroy(stop));

    return 0;
}
