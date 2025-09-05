#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include "mv_ops.h"

using namespace std::chrono;

// Declare extern CPU references and timing helpers defined in mv_ops.cu
extern void cpu_matvec_reference(const float*, const float*, float*, int);
extern void cpu_band_matvec_reference(const float*, const float*, float*, int, int);
extern void cpu_sym_matvec_reference(const float*, const float*, float*, int);
extern void cpu_tri_matvec_reference(const float*, const float*, float*, int, bool);
extern void cpu_rank1_update_reference(float*, const float*, const float*, float, int);
extern void cpu_rank2_update_reference(float*, const float*, const float*, float, int);

extern double time_gpu_func(void(*)(const float*, const float*, float*, int),
                           const float*, const float*, float*, int);

extern double time_gpu_func_alpha(void(*)(float*, const float*, const float*, float, int),
                                 float*, const float*, const float*, float, int);

extern void validate_results(const float*, const float*, int, const char*);

extern void band_matvec_mul_launcher(const float*, const float*, float*, int);
extern void tri_matvec_lower_launcher(const float*, const float*, float*, int);
extern void tri_matvec_upper_launcher(const float*, const float*, float*, int);

// CPU timing helper
template <typename Func, typename... Args>
double time_cpu_func(Func f, Args&&... args) {
    auto start = high_resolution_clock::now();
    f(std::forward<Args>(args)...);
    auto stop = high_resolution_clock::now();
    return duration<double, std::milli>(stop - start).count();
}

int main(int argc, char** argv) {
    bool run_cpu_reference = true;
    if (argc > 1 && strcmp(argv[1], "nocpu") == 0) {
        run_cpu_reference = false;
        printf("Running with CPU validation DISABLED\n");
    } else {
        printf("Running with CPU validation ENABLED\n");
    }

    const int N = 1024;
    const int mat_size = N * N * sizeof(float);
    const int vec_size = N * sizeof(float);
    const int bandwidth = 5;

    float *h_A = (float*)malloc(mat_size);
    float* h_A_rank1 = (float*)malloc(mat_size);
    float* h_A_rank2 = (float*)malloc(mat_size);
    memcpy(h_A_rank1, h_A, mat_size);
    memcpy(h_A_rank2, h_A, mat_size);

    float *h_B = (float*)malloc(mat_size);
    float *h_x = (float*)malloc(vec_size);
    float *h_y = (float*)malloc(vec_size);
    float *h_ref = (float*)malloc(vec_size);

    if (!h_A || !h_B || !h_x || !h_y || !h_ref) {
        printf("Host memory allocation failed\n");
        return -1;
    }

    srand(time(NULL));
    for (int i=0; i < N*N; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i=0; i < N*N; ++i) h_B[i] = (float)rand() / RAND_MAX;
    for (int i=0; i < N; ++i) h_x[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_x, *d_y;
    cudaMalloc(&d_A, mat_size);
    cudaMalloc(&d_B, mat_size);
    cudaMalloc(&d_x, vec_size);
    cudaMalloc(&d_y, vec_size);

    cudaMemcpy(d_A, h_A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, vec_size, cudaMemcpyHostToDevice);

    float *d_A_rank1, *d_A_rank2;
    cudaMalloc(&d_A_rank1, mat_size);
    cudaMalloc(&d_A_rank2, mat_size);
    cudaMemcpy(d_A_rank1, h_A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rank2, h_A, mat_size, cudaMemcpyHostToDevice);


    double cpu_time = 0.0, gpu_time = 0.0;

    // matvec_mul
    if (run_cpu_reference)
        cpu_time = time_cpu_func(cpu_matvec_reference, h_A, h_x, h_ref, N);
    gpu_time = time_gpu_func(matvec_mul, d_A, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU matvec_mul time: %.3f ms\n", cpu_time);
    printf("GPU matvec_mul time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_ref, h_y, N, "matvec_mul");

    // band_matvec_mul
    if (run_cpu_reference)
        cpu_time = time_cpu_func(cpu_band_matvec_reference, h_B, h_x, h_ref, N, bandwidth);
    gpu_time = time_gpu_func(band_matvec_mul_launcher, d_B, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU band_matvec_mul time: %.3f ms\n", cpu_time);
    printf("GPU band_matvec_mul time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_ref, h_y, N, "band_matvec_mul");

    // sym_matvec_mul
    if (run_cpu_reference)
        cpu_time = time_cpu_func(cpu_sym_matvec_reference, h_A, h_x, h_ref, N);
    gpu_time = time_gpu_func(sym_matvec_mul, d_A, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU sym_matvec_mul time: %.3f ms\n", cpu_time);
    printf("GPU sym_matvec_mul time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_ref, h_y, N, "sym_matvec_mul");

    // tri_matvec_mul lower
    if (run_cpu_reference)
        cpu_time = time_cpu_func(cpu_tri_matvec_reference, h_A, h_x, h_ref, N, true);
    gpu_time = time_gpu_func(tri_matvec_lower_launcher, d_A, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU tri_matvec_mul_lower time: %.3f ms\n", cpu_time);
    printf("GPU tri_matvec_mul_lower time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_ref, h_y, N, "tri_matvec_mul_lower");

    // tri_matvec_mul upper
    if (run_cpu_reference)
        cpu_time = time_cpu_func(cpu_tri_matvec_reference, h_A, h_x, h_ref, N, false);
    gpu_time = time_gpu_func(tri_matvec_upper_launcher, d_A, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU tri_matvec_mul_upper time: %.3f ms\n", cpu_time);
    printf("GPU tri_matvec_mul_upper time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_ref, h_y, N, "tri_matvec_mul_upper");

    // rank1_update
    float alpha = 1.5f;
    if(run_cpu_reference)
        cpu_rank1_update_reference(h_A_rank1, h_x, h_y, alpha, N);
    gpu_time = time_gpu_func_alpha(rank1_update, d_A_rank1, d_x, d_y, alpha, N);
    //cudaMemcpy(h_y_gpu, d_A_rank1, mat_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU rank1_update time: %.3f ms\n", cpu_time);
    printf("GPU rank1_update time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_A, h_y, N * N, "rank1_update");

    // rank2_update
    if(run_cpu_reference)
        cpu_rank2_update_reference(h_A_rank2, h_x, h_y, alpha, N);
    gpu_time = time_gpu_func_alpha(rank2_update, d_A_rank2, d_x, d_y, alpha, N);
    //cudaMemcpy(h_y_gpu, d_A_rank2, mat_size, cudaMemcpyDeviceToHost);
    if (run_cpu_reference) printf("CPU rank2_update time: %.3f ms\n", cpu_time);
    printf("GPU rank2_update time: %.3f ms\n", gpu_time);
    if (run_cpu_reference) validate_results(h_A, h_y, N * N, "rank2_update");

    // cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_A);
    free(h_B);
    free(h_x);
    free(h_y);
    free(h_ref);

    printf("All operations completed successfully.\n");
    return 0;
}
