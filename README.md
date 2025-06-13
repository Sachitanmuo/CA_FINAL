# Mantiuk Algorithm Implementation Comparison

This project contains multiple implementations of the Mantiuk algorithm and provides performance comparison tools.

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
- `[iterations]`: Number of algorithm iterations, defaults to 500 (optional)

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
