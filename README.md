# CUDA-Kernels-Demo

This repository contains a collection of CUDA programs demonstrating fundamental and advanced GPU computing techniques. Each module is designed to illustrate core CUDA programming concepts and optimization strategies, making this repository a practical resource for learning, experimentation, and showcasing GPU acceleration capabilities.

---

## Repository Structure

- **vector_ops/**  
  High-performance vector addition, subtraction, multiplication, and dot product using both CPU and GPU implementations, with dynamic kernel configuration and extensive validation.  
  *(See detailed section below for vector_ops subproject)*

- *(Future subprojects: matrix_ops, reductions, convolution, etc.)*

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

### Folder Content

- `include/vector_ops.h` — Function prototypes, macros, typedefs
- `src/vector_ops.cu` — CUDA kernels and CPU implementations
- `src/main.cu` — Driver code: setup, execution, timing, and validation
- `Makefile` — Build instructions/configuration flags

### Usage

1. **Set the CUDA architecture flag in the Makefile:**  
`NVCC_ARCH := -arch=sm_61`
Edit this to your GPU’s compute capability (see table).

2. **Build and run:**
`make clean` # cleans previous builds
`make` # builds the application
`./vector_ops_app`


### Setting GPU Architecture

- Common architectures:  
Pascal: `sm_61` | Volta: `sm_70` | Turing: `sm_75` | Ampere: `sm_80`  
See [CUDA GPU list](https://developer.nvidia.com/cuda-gpus) for your model.
- Adjust `NVCC_ARCH` in Makefile as needed.

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

For bug reports, support, or feedback, open issues or contact the maintainer directly.  
Happy parallel programming and GPU acceleration!


