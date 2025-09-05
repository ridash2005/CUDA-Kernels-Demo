#include <cstdio>
#include "mv_ops.h"
#include <cuda_runtime.h>
#include <cmath>

#define TILE_WIDTH 16

// CUDA Kernels

__global__ void matVecMulKernel(const float *A, const float *x, float *y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float val = 0.0f;
        for (int j = 0; j < N; ++j)
            val += A[row * N + j] * x[j];
        y[row] = val;
    }
}

__global__ void bandMatVecMulKernel(const float *A_band, const float *x, float *y, int N, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float val = 0.0f;
        int lower = max(0, row - k);
        int upper = min(N - 1, row + k);
        for (int j=lower; j <= upper; ++j) {
            int band_col = k + j - row;
            val += A_band[row * (2*k + 1) + band_col] * x[j];
        }
        y[row] = val;
    }
}

__global__ void symMatVecMulKernel(const float *A, const float *x, float *y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N) {
        float val = 0.0f;
        for(int col=0; col<N; ++col)
            val += A[row*N+col] * x[col];
        y[row] = val;
    }
}

__global__ void triMatVecMulKernel(const float *A, const float *x, float *y, int N, bool lower) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N){
        float val = 0.0f;
        if (lower) {
            for(int col=0; col <= row; ++col)
                val += A[row*N + col] * x[col];
        } else {
            for(int col=row; col < N; ++col)
                val += A[row*N + col] * x[col];
        }
        y[row] = val;
    }
}

// Triangular solve placeholder
void tri_solve(const float *A, const float *b, float *x, int N, bool lower) {
    printf("[tri_solve] GPU implementation not provided; please implement CPU or iterative GPU version.\n");
}

__global__ void rank1UpdateKernel(float *A, const float *x, const float *y, float alpha, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N)
        A[row*N + col] += alpha * x[row] * y[col];
}

__global__ void rank2UpdateKernel(float *A, const float *x, const float *y, float alpha, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N)
        A[row*N + col] += alpha * (x[row] * y[col] + y[row] * x[col]);
}

// Timing helpers

template<typename F>
void launch_and_time(const char *name, F&& launch_kernel) {
    printf("Running %s...\n", name);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_kernel();
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("%s finished in %.3f ms\n", name, elapsed_ms);
}

// Host wrappers

void matvec_mul(const float *A, const float *x, float *y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    launch_and_time("matvec_mul", [&](){
        matVecMulKernel<<<blocks, threads>>>(A, x, y, N);
    });
}

void band_matvec_mul(const float *A_band, const float *x, float *y, int N, int k) {
    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    launch_and_time("band_matvec_mul", [&](){
        bandMatVecMulKernel<<<blocks, threads>>>(A_band, x, y, N, k);
    });
}

void sym_matvec_mul(const float *A, const float *x, float *y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    launch_and_time("sym_matvec_mul", [&](){
        symMatVecMulKernel<<<blocks, threads>>>(A, x, y, N);
    });
}

void tri_matvec_mul(const float *A, const float *x, float *y, int N, bool lower) {
    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    launch_and_time(lower ? "tri_matvec_mul_lower" : "tri_matvec_mul_upper", [&](){
        triMatVecMulKernel<<<blocks, threads>>>(A, x, y, N, lower);
    });
}

void rank1_update(float *A, const float *x, const float *y, float alpha, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    launch_and_time("rank1_update", [&](){
        rank1UpdateKernel<<<blocks, threads>>>(A, x, y, alpha, N);
    });
}

void rank2_update(float *A, const float *x, const float *y, float alpha, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    launch_and_time("rank2_update", [&](){
        rank2UpdateKernel<<<blocks, threads>>>(A, x, y, alpha, N);
    });
}

// Validation function

void validate_results(const float *ref, const float *out, int N, const char *msg) {
    printf("Validating results for %s...\n", msg);
    for(int i=0; i<N; i++) {
        if(fabs(ref[i] - out[i]) > 1e-3f) {
            printf("%s mismatch at %d: ref=%f vs out=%f\n", msg, i, ref[i], out[i]);
            return;
        }
    }
    printf("%s PASSED\n", msg);
}

// Wrapper launchers with fixed parameters needed for passing to function pointers

void band_matvec_mul_launcher(const float *A_band, const float *x, float *y, int N) {
    const int bandwidth = 5;
    band_matvec_mul(A_band, x, y, N, bandwidth);
}

void tri_matvec_lower_launcher(const float *A, const float *x, float *y, int N) {
    tri_matvec_mul(A, x, y, N, true);
}

void tri_matvec_upper_launcher(const float *A, const float *x, float *y, int N) {
    tri_matvec_mul(A, x, y, N, false);
}

// GPU timing helpers definitions

double time_gpu_func(void(*gpu_func)(const float*, const float*, float*, int),
                     const float* d_in1, const float* d_in2,
                     float* d_out, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_func(d_in1, d_in2, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms;
}

double time_gpu_func_alpha(void(*gpu_func)(float*, const float*, const float*, float, int),
                           float* d_A, const float* d_x, const float* d_y,
                           float alpha, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_func(d_A, d_x, d_y, alpha, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms;
}

// CPU reference functions - full implementations

void cpu_matvec_reference(const float *A, const float *x, float *y, int N) {
    for (int i=0; i<N; ++i) {
        float val=0.0f;
        for (int j=0; j<N; ++j)
            val += A[i*N+j]*x[j];
        y[i] = val;
    }
}

void cpu_band_matvec_reference(const float *A_band, const float *x, float *y, int N, int k) {
    for (int i=0; i<N; ++i) {
        float val=0.0f;
        int lower = (i-k < 0) ? 0 : i-k;
        int upper = (i+k >= N) ? N-1 : i+k;
        for (int j=lower; j<=upper; ++j) {
            int band_col = k + j - i;
            val += A_band[i*(2*k+1) + band_col]*x[j];
        }
        y[i] = val;
    }
}

void cpu_sym_matvec_reference(const float *A, const float *x, float *y, int N) {
    for (int i=0; i<N; ++i) {
        float val=0.0f;
        for (int j=0; j<N; ++j)
            val += A[i*N+j]*x[j];
        y[i] = val;
    }
}

void cpu_tri_matvec_reference(const float *A, const float *x, float *y, int N, bool lower) {
    for (int i=0; i<N; ++i) {
        float val=0.0f;
        if (lower) {
            for (int j=0; j<=i; ++j)
                val += A[i*N+j]*x[j];
        } else {
            for (int j=i; j<N; ++j)
                val += A[i*N+j]*x[j];
        }
        y[i] = val;
    }
}

void cpu_rank1_update_reference(float *A, const float *x, const float *y, float alpha, int N) {
    for (int i=0; i<N; ++i)
        for (int j=0; j<N; ++j)
            A[i*N+j] += alpha * x[i] * y[j];
}

void cpu_rank2_update_reference(float *A, const float *x, const float *y, float alpha, int N) {
    for (int i=0; i<N; ++i)
        for (int j=0; j<N; ++j)
            A[i*N+j] += alpha * (x[i]*y[j] + y[i]*x[j]);
}
