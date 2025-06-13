# Mantiuk Algorithm Implementation Comparison

This repository compares different implementations of the Mantiuk algorithm for HDR (High Dynamic Range) image processing, focusing on performance optimization between CPU and CUDA-accelerated versions.

## Example Results

Here's a comparison between the input HDR image and the processed result:

| Input HDR Image | Processed Result |
|----------------|------------------|
| ![Input HDR](input_preview/001.png) | ![Processed Result](outputs/001.png) |

The processed image shows how the algorithm preserves local contrast while compressing the dynamic range to be displayable on standard monitors.

## Algorithm Overview

The Mantiuk algorithm is a tone mapping operator that preserves local contrast while compressing the dynamic range of HDR images. The implementation uses the following key steps:

1. **Gradient Domain Processing**: The algorithm operates in the gradient domain to preserve local contrast.
2. **Poisson Solver**: Uses Jacobi iteration method to solve the Poisson equation, which reconstructs the final image from the processed gradients.
3. **Parameter Control**:
   - Alpha (α): Controls the strength of contrast enhancement
   - Offset: Prevents division by zero in gradient computation
   - Gamma: Controls the overall brightness of the output

The main computational bottleneck is the Jacobi iteration solver, which is why we implement different versions:
- CPU version: Sequential processing
- Naive GPU version: Basic CUDA implementation
- Shared memory GPU version: Optimized CUDA implementation using shared memory

## System Requirements

- Linux Operating System
- GCC Compiler
- OpenCV Library
- CUDA Toolkit (for GPU implementations)

## Compilation

1. Install required dependencies:
```bash
sudo apt-get update
sudo apt-get install build-essential libopencv-dev
```

2. Compile all implementations:
```bash
make all
```

## Running the Code

Use the `runtime_cmp.sh` script to compare execution times of different implementations:

```bash
./runtime_cmp.sh <input_image> [iterations]
```

### Parameters

- `<input_image>`: Path to the input image (required)
- `[iterations]`: Number of Jacobi solver iterations, defaults to 500 (optional)
  - Higher iteration count provides more accurate results but increases computation time
  - The Jacobi solver converges gradually, so more iterations generally lead to better quality
  - Typical values range from 100 to 1000 depending on desired quality vs. speed trade-off

### Example

```bash
./runtime_cmp.sh input.png 1000
```

### Output Description

The script compares execution times of three implementations:
- mantiuk_cpu: CPU version
- mantiuk_naive: Basic GPU version
- mantiuk_shared: GPU version with shared memory

Each implementation generates a corresponding output image with filename format `out_<implementation_name>.png`.

## Notes

1. Ensure all executables have proper permissions:
```bash
chmod +x runtime_cmp.sh
chmod +x mantiuk_*
```

2. If you encounter "binary missing" errors, verify that all implementations have been compiled correctly.

3. It is recommended to use smaller images for testing to save execution time.
