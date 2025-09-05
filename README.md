# CUDA-Kernels-Demo

This repository contains a collection of CUDA programs demonstrating fundamental and advanced GPU computing techniques. Each module is designed to illustrate core CUDA programming concepts and optimization strategies, making this repository a practical resource for learning, experimentation, and showcasing GPU acceleration capabilities.

---

## Repository Structure

- **vector_ops/**  
  High-performance vector addition, subtraction, multiplication, and dot product using both CPU and GPU implementations, with dynamic kernel configuration and extensive validation.  
  *(See detailed section below for vector_ops subproject)*

- **mv_ops/**
CUDA implementations of core matrix-vector operations with comprehensive CPU reference implementations and GPU accelerations.
Includes dense, banded, symmetric, and triangular matrix-vector multiply kernels, along with rank-1 and rank-2 matrix update operations.
  *(See detailed section below for mat_mult subproject)*

---

## Getting Started

### Prerequisites

- NVIDIA CUDA Toolkit (12.6 or compatible)
- CUDA-capable GPU (Pascal/Volta/Turing/Ampere or newer; CC 6.0+ recommended)
- Supported C++ compiler compatible with nvcc
- Windows/WSL/Linux/Mac (see platform notes for specifics)
- Git Bash/WSL recommended for Windows users

### Cloning the Repository
To start, clone the repository using this command:

`git clone https://github.com/your-username/CUDA-Kernels-Demo.git`
`cd CUDA-Kernels-Demo`

### Setting GPU Architecture

- Common architectures:  
Pascal: `sm_61` | Volta: `sm_70` | Turing: `sm_75` | Ampere: `sm_80`  
See [CUDA GPU list](https://developer.nvidia.com/cuda-gpus) for your model.
- Adjust `NVCC_ARCH` in Makefile as needed.


### Usage

1. **Set the CUDA architecture flag in the Makefile:**  
`NVCC_ARCH := -arch=sm_61`
Edit this to your GPU’s compute capability (see table).

---

## Platform Notes

- On **Windows**, adapt your Makefile `clean` target to use `del` instead of `rm` or use a Unix-like shell (Git Bash/WSL).
- Ensure your architecture (`sm_xy`) option matches your GPU—see instructions in vector_ops README.

---

## vector_ops Subproject

### Overview

`vector_ops/` provides reference implementations and GPU-accelerated versions of:

- Vector Addition, Subtraction, Multiplication
- Dot Product (atomic, float-based reduction)
- Dynamic block sizing based on detected GPU capability
- Verification of GPU results against CPU versions
- Detailed timing for CPU and GPU performance benchmarking

### Folder Structure

-vector_ops/
- ├── include/
- │   └── vector_ops.h          
- ├── src/
- │   ├── vector_ops.cu          
- ├── Makefile               
- └── README.md

**Build and run:**
`make clean` # cleans previous builds
`make` # builds the application
`./vector_ops_app.exe`

### Adjusting Vector Size N

- Edit `#define N` in `main.cu` to control vector size.
- The vector size must fit GPU memory (`N * sizeof(float) * number_of_vectors + overhead`).
- Use smaller values (e.g., `N=1000`) for debugging; larger for benchmarking.

### Output

- **Verification:** Each operation prints [PASSED] or [FAILED] for correctness.
- **Timing:** Millisecond execution times for CPU and GPU operations.
- **Diagnostics:** Up to five failed indices printed if a verification fails.
- **Dot Product:** Scalar comparison with 1% relative error tolerance (float vs double).

### Troubleshooting

| Problem                                     | Solution                                                         |
|---------------------------------------------|------------------------------------------------------------------|
| `DEFAULT_BLOCK_SIZE` undefined              | Ensure all source files include `vector_ops.h`                   |
| Makefile `rm` not found on Windows          | Use Git Bash/WSL or switch to `del` command in Makefile          |
| CUDA arch mismatch (`sm_xy`)                | Update Makefile flag for your GPU architecture                   |
| Out of memory errors                        | Lower `N` or check available GPU memory with `nvidia-smi`        |
| Kernel result mismatch                      | Confirm device-to-host memory copies and use proper verification |
| Include path/SDK errors in IDE              | Add CUDA/include and Windows Kits folder to IDE include paths    |
| Slow verification for large `N`             | Reduce `N` or parallelize CPU code                               |

### Extending vector_ops

- Add new vector operations using the kernel/template pattern.
- Implement double precision kernels and verify hardware support.
- Adapt kernel launches to multi-streams or batched vector processing.
- Profile GPU execution with Nvidia Nsight or Visual Profiler.
- Explore multi-GPU support for very large data sets.

## mv_ops Subproject - README

### Overview

The mv_ops subproject provides UTF-8 CUDA implementations of high-performance matrix-vector operations. It demonstrates GPU parallelization techniques applied to:

- Dense matrix-vector multiplication and specialized variants (banded, symmetric, triangular)
- Rank-1 and rank-2 matrix updates
- Comprehensive CPU reference implementations for correctness validation
- High-resolution CPU and GPU timing for benchmarking



### Folder Structure

- mv_ops/
- ├── include/
- │   └── mv_ops.h          
- ├── src/
- │   ├── mv_ops.cu          
- ├── Makefile               
- └── README.md    

**Build and run:**
`make clean` # cleans previous builds
`make` # builds the application
`./mv_app.exe`


### User-Configurable Parameters

These parameters can be tuned directly in the source code (`src/mv_ops.cu` and `src/main.cu`) to adapt computation to various GPU architectures and application needs:

| Parameter              | Description                                    | Approximate Location        | Default Value              |
|------------------------|------------------------------------------------|----------------------------|-----------------------------|
| `N`                    | Matrix/vector dimension (square matrices)      | `src/main.cu`               | 1024                       |
| `bandwidth`            | Bandwidth parameter for banded matrix kernels  | `src/mv_ops.cu`             | 5                          |
| `THREADS_PER_BLOCK_1D` | Threads per block in 1D kernels                | `src/mv_ops.cu`             | 256                        |
| `THREADS_X_2D`         | Threads per block along X in 2D kernels        | `src/mv_ops.cu`             | 16                         |
| `THREADS_Y_2D`         | Threads per block along Y in 2D kernels        | `src/mv_ops.cu`             | 16                         |
| `VALIDATION_TOLERANCE` | Tolerance for floating-point result validation | `src/mv_ops.cu`             | 1e-3                       |
| CPU validation toggle  | Enable/disable CPU reference validation        | `src/main.cu`               | Enabled by default         |


### Troubleshooting Guide

| Issue                         | Recommended Actions                                       |
|------------------------------ |---------------------------------------------------------- |
| Kernel launch failure         | Verify launch dimensions against matrix/vector sizes      |
| Out of GPU memory             | Decrease matrix/vector size or release other allocations  |
| Result validation mismatch    | Verify data transfers, kernel logic, and validation args  |
| Unexpectedly long GPU time    | Ensure proper CUDA event synchronization and error checks |
| Slow kernels                  | Tune thread block sizes and memory access patterns        |


### Extending mv_ops

- Support for double-precision (FP64) and mixed precision arithmetic
- Batched matrix-vector/matrix operations for simultaneous multi-problem execution
- Use of asynchronous CUDA streams and pinned host memory for overlapping compute and data transfers
- Further kernel optimizations including warp shuffles and loop unrolling
- Full GPU implementation of triangular solve (TRSM) and matrix inversion kernels
- Profiling integration using Nsight Systems and Nsight Compute for bottleneck analysis


---

## Contributing

- Fork the repo, make improvements, and submit pull requests!
- For issues, describe your platform (GPU model, OS, compiler, CUDA version) and error output for easier replication and support.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Future Modules

Planned subprojects include:

- Matrix operations
- GPU sorting & searching
- Parallel reduction and scan
- Convolution and filtering kernels
- Deep learning primitives

*Feel free to contribute your own CUDA learning modules!*

---

## Contact

For questions or contributions, please open an issue or submit a pull request.
gmail: rickaryadas@gmail.com



