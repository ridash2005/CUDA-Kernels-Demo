#include "vector_ops.h"
#include <stdio.h>

int getOptimalBlockSize() {
    int device;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) != cudaSuccess) return DEFAULT_BLOCK_SIZE;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return DEFAULT_BLOCK_SIZE;
    int maxThreads = prop.maxThreadsPerBlock;
    if (maxThreads <= 0 || maxThreads > 1024) maxThreads = DEFAULT_BLOCK_SIZE;
    return maxThreads;
}

__global__ void kVectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void kVectorSub(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void kVectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void kDotProductFloatAtomic(const float* a, const float* b, float* result, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

void launchGpuVectorAdd(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock) {
    if (threadsPerBlock <= 0) threadsPerBlock = getOptimalBlockSize();
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    kVectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
}

void launchGpuVectorSub(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock) {
    if (threadsPerBlock <= 0) threadsPerBlock = getOptimalBlockSize();
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    kVectorSub<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
}

void launchGpuVectorMul(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock) {
    if (threadsPerBlock <= 0) threadsPerBlock = getOptimalBlockSize();
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    kVectorMul<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
}

void cpuVectorAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

void cpuVectorSub(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

void cpuVectorMul(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

double cpuDotProductDouble(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)a[i] * (double)b[i];
    return sum;
}

int verifyResult(const float* ref, const float* res, int n, const char* msg, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabs(ref[i] - res[i]) > tol) {
            printf("[FAILED] %s at idx %d: expected %f, got %f\n", msg, i, ref[i], res[i]);
            return 0;
        }
    }
    printf("[PASSED] %s\n", msg);
    return 1;
}
