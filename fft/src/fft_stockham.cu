#include "fft_stockham.h"
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Host bit-reversal helper
static int bitReverseInt(int x, int log2n) {
    int y = 0;
    for (int i = 0; i < log2n; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}


// Stockham kernel: one stage
__global__ void fftStockhamKernel(const float2* __restrict__ in,
                                 float2* __restrict__ out,
                                 int n, int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int group = tid / (2 * m);
    int k = tid % (2 * m);

    int i0 = group * (2 * m) + (k % m);
    int i1 = i0 + m;

    int k_mod = k % m;
    float angle = -M_PI * k_mod / m;
    float2 w = make_float2(cosf(angle), sinf(angle));

    float2 u = in[i0];
    float2 v = in[i1];
    
    float2 vw;
    vw.x = w.x * v.x - w.y * v.y;
    vw.y = w.x * v.y + w.y * v.x;

    out[i0] = make_float2(u.x + vw.x, u.y + vw.y);
    out[i1] = make_float2(u.x - vw.x, u.y - vw.y);
}


// Host runner
void runStockhamFFT(std::vector<float2>& input) {
    const int N = static_cast<int>(input.size());
    if (N <= 1) return;

    int log2n = 0;
    while ((1 << log2n) < N) ++log2n;

    // Bit-reverse input on host
    std::vector<float2> reordered(N);
    for (int i = 0; i < N; ++i) {
        int r = bitReverseInt(i, log2n);
        reordered[r] = input[i];
    }

    float2 *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float2));
    cudaMalloc(&d_out, N * sizeof(float2));
    cudaMemcpy(d_in, reordered.data(), N * sizeof(float2), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    for (int m = 1; m < N; m <<= 1) {
        int threads = N;
        int gridSize = (threads + blockSize - 1) / blockSize;
        fftStockhamKernel<<<gridSize, blockSize>>>(d_in, d_out, N, m);
        cudaDeviceSynchronize();

        std::swap(d_in, d_out);
    }

    cudaMemcpy(input.data(), d_in, N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Normalize by N
    const float invN = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        input[i].x *= invN;
        input[i].y *= invN;
    }
}
