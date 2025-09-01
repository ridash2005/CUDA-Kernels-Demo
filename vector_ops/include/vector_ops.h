#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_BLOCK_SIZE 256

int getOptimalBlockSize();

__global__ void kVectorAdd(const float* a, const float* b, float* c, int n);
__global__ void kVectorSub(const float* a, const float* b, float* c, int n);
__global__ void kVectorMul(const float* a, const float* b, float* c, int n);

__global__ void kDotProductFloatAtomic(const float* a, const float* b, float* result, int n);

void launchGpuVectorAdd(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock);
void launchGpuVectorSub(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock);
void launchGpuVectorMul(const float* d_a, const float* d_b, float* d_c, int n, int threadsPerBlock);

void cpuVectorAdd(const float* a, const float* b, float* c, int n);
void cpuVectorSub(const float* a, const float* b, float* c, int n);
void cpuVectorMul(const float* a, const float* b, float* c, int n);

double cpuDotProductDouble(const float* a, const float* b, int n);

int verifyResult(const float* ref, const float* res, int n, const char* msg, float tol);

#ifdef __cplusplus
}
#endif

#endif // VECTOR_OPS_H
