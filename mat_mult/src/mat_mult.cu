// src/mat_mult.cu
// Device kernels and launch wrappers for matrix multiplication
#include "mat_mult.h"
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulNaiveKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++)
            value += A[row * N + k] * B[k * N + col];
        C[row * N + col] = value;
    }
}

__global__ void matMulTiledKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        tileA[threadIdx.y][threadIdx.x] = (row < N && m * TILE_WIDTH + threadIdx.x < N)
            ? A[row * N + m * TILE_WIDTH + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (col < N && m * TILE_WIDTH + threadIdx.y < N)
            ? B[(m * TILE_WIDTH + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = value;
}

extern "C" void matmul_naive_cuda(const float *d_A, const float *d_B, float *d_C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matMulNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

extern "C" void matmul_tiled_cuda(const float *d_A, const float *d_B, float *d_C, int N) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matMulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}
