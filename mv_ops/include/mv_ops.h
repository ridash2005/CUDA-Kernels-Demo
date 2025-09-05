#pragma once

// Matrix-vector operations
void matvec_mul(const float *A, const float *x, float *y, int N);
void band_matvec_mul(const float *A_band, const float *x, float *y, int N, int k);
void sym_matvec_mul(const float *A, const float *x, float *y, int N);
void tri_matvec_mul(const float *A, const float *x, float *y, int N, bool lower);
void tri_solve(const float *A, const float *b, float *x, int N, bool lower);
void rank1_update(float *A, const float *x, const float *y, float alpha, int N);
void rank2_update(float *A, const float *x, const float *y, float alpha, int N);

// Validation helper
void validate_results(const float *ref, const float *out, int N, const char *msg);

// Wrappers for kernels requiring fixed parameters
void band_matvec_mul_launcher(const float *A_band, const float *x, float *y, int N);
void tri_matvec_lower_launcher(const float *A, const float *x, float *y, int N);
void tri_matvec_upper_launcher(const float *A, const float *x, float *y, int N);

// Timing helpers for GPU
double time_gpu_func(void(*gpu_func)(const float*, const float*, float*, int),
                    const float*, const float*, float*, int);

double time_gpu_func_alpha(void(*gpu_func)(float*, const float*, const float*, float, int),
                          float*, const float*, const float*, float, int);

// CPU reference implementations
void cpu_matvec_reference(const float*, const float*, float*, int);
void cpu_band_matvec_reference(const float*, const float*, float*, int, int);
void cpu_sym_matvec_reference(const float*, const float*, float*, int);
void cpu_tri_matvec_reference(const float*, const float*, float*, int, bool);
void cpu_rank1_update_reference(float*, const float*, const float*, float, int);
void cpu_rank2_update_reference(float*, const float*, const float*, float, int);
