#ifndef FFT_STOCKHAM_H
#define FFT_STOCKHAM_H

#include <vector>
#include <cuda_runtime.h>

void runStockhamFFT(std::vector<float2>& input);

#endif // FFT_STOCKHAM_H
