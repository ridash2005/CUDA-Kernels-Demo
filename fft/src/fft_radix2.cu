#include "fft_radix2.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Bit-reversal permutation on host
__host__ int bitReverse(int x, int log2n) {
    int n = 0;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}


// Radix-2 FFT kernel
__global__ void fftRadix2Kernel(float2* data, int n, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = step * 2;
    if (tid < n / 2) {
        int k = tid % step;
        int j = (tid / step) * m;

        float angle = -2.0f * M_PI * k / m;
        float2 w = make_float2(cosf(angle), sinf(angle));

        float2 t = data[j + k + step];
        float2 u = data[j + k];

        // t = w * t
        float2 tw;
        tw.x = w.x * t.x - w.y * t.y;
        tw.y = w.x * t.y + w.y * t.x;

        data[j + k] = make_float2(u.x + tw.x, u.y + tw.y);
        data[j + k + step] = make_float2(u.x - tw.x, u.y - tw.y);
    }
}


// Host runner
void runRadix2FFT(std::vector<float2>& input) {
    int N = input.size();
    int log2n = static_cast<int>(log2f(N));

    // Bit reversal reorder on host
    std::vector<float2> temp(N);
    for (int i = 0; i < N; i++) {
        temp[bitReverse(i, log2n)] = input[i];
    }
    input = temp;

    float2* d_data;
    cudaMalloc(&d_data, N * sizeof(float2));
    cudaMemcpy(d_data, input.data(), N * sizeof(float2), cudaMemcpyHostToDevice);

    for (int step = 1; step < N; step *= 2) {
        int threads = N / 2;
        int blockSize = 256;
        int gridSize = (threads + blockSize - 1) / blockSize;
        fftRadix2Kernel<<<gridSize, blockSize>>>(d_data, N, step);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(input.data(), d_data, N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Normalize
    for (auto &v : input) {
        v.x /= N;
        v.y /= N;
    }
}
