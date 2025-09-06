#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "fft_radix2.h"
#include "fft_stockham.h"
#include "fft_validation.cpp"

// Declare extern functions
std::vector<float2> cpuDFT(const std::vector<float2>& input);
float rmsError(const std::vector<float2>& a, const std::vector<float2>& b);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


int main() {
    const int N = 8;
    std::vector<float2> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = make_float2(sinf(2.0f * M_PI * i / N), 0.0f);
    }

    // Compute CPU DFT reference
    auto ref = cpuDFT(input);

    // Radix-2 FFT
    auto dataRadix = input;
    runRadix2FFT(dataRadix);
    std::cout << "Radix-2 FFT (normalized):\n";
    for (int i = 0; i < N; i++) {
        std::cout << i << ": " << dataRadix[i].x << " + " << dataRadix[i].y << "j\n";
    }

    // Stockham FFT
    auto dataStockham = input;
    runStockhamFFT(dataStockham);
    std::cout << "Stockham FFT (normalized):\n";
    for (int i = 0; i < N; i++) {
        std::cout << i << ": " << dataStockham[i].x << " + " << dataStockham[i].y << "j\n";
    }

    // RMS error compared to CPU DFT
    std::cout << "\nRMS Error vs CPU DFT\n";
    std::cout << "Radix-2 : " << rmsError(dataRadix, ref) << "\n";
    std::cout << "Stockham: " << rmsError(dataStockham, ref) << "\n";

    return 0;
}
