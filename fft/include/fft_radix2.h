#ifndef FFT_RADIX2_H
#define FFT_RADIX2_H

#include <vector>
#include <cuda_runtime.h>

void runRadix2FFT(std::vector<float2>& input);

#endif // FFT_RADIX2_H
