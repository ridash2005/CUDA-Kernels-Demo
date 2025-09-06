#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#define M_PI 3.14159265358979323846
// CPU DFT for validation
std::vector<float2> cpuDFT(const std::vector<float2>& input) {
    int N = input.size();
    std::vector<float2> output(N);

    for (int k = 0; k < N; k++) {
        float sumReal = 0.0f, sumImag = 0.0f;
        for (int n = 0; n < N; n++) {
            float angle = -2.0f * M_PI * k * n / N;
            float c = cosf(angle), s = sinf(angle);
            sumReal += input[n].x * c - input[n].y * s;
            sumImag += input[n].x * s + input[n].y * c;
        }
        output[k] = make_float2(sumReal / N, sumImag / N);
    }
    return output;
}

// RMS error calculator
float rmsError(const std::vector<float2>& a, const std::vector<float2>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float dx = a[i].x - b[i].x;
        float dy = a[i].y - b[i].y;
        err += dx * dx + dy * dy;
    }
    return sqrtf(err / a.size());
}
