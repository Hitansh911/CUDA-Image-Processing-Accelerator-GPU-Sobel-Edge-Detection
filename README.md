# CUDA Image Processing Accelerator

A GPU-accelerated image-processing capstone project using NVIDIA CUDA.

## Project goal

This project compares CPU and CUDA GPU implementations of a 3x3 Sobel edge-detection filter.

Pipeline:

Input PPM image -> Grayscale -> Sobel Edge Detection -> Output PGM image

The same input is processed on the CPU and GPU. Execution times are reported so the GPU acceleration can be demonstrated during the final presentation.

## Why this project

Sobel edge detection is a simple but highly parallel image-processing operation: every output pixel can be computed independently from a small neighborhood of input pixels. This makes it a good CUDA workload.

## GPU usage

The GPU implementation uses a CUDA kernel in `src/sobel.cu`.

Each CUDA thread processes one image pixel:
- `blockIdx`
- `blockDim`
- `threadIdx`

The GPU kernel reads neighboring pixels, computes the horizontal and vertical Sobel gradients, and writes one output pixel.

CUDA events are used to measure GPU kernel execution time.

## Requirements

- NVIDIA GPU
- CUDA Toolkit / `nvcc`
- Linux or WSL/Linux environment
- `make`

Check CUDA:

```bash
nvcc --version
nvidia-smi
```

## Build

```bash
make clean
make
```

## Run

Default run:

```bash
make run
```

Or directly:

```bash
./cuda_sobel data/input.ppm output/gpu_edges.pgm
```

## Output

The program prints:

- image dimensions
- CPU execution time
- GPU kernel execution time
- speedup
- output file names

It creates:

```text
output/cpu_edges.pgm
output/gpu_edges.pgm
```

## Input format

The included input is a binary PPM (`P6`) image.

The program supports `P6` PPM input with 8-bit RGB pixels.

## Reproducible demo

```bash
make clean
make
make run
```

Example output format:

```text
CUDA Image Processing Accelerator
Input: data/input.ppm
Image: 512 x 512

CPU Sobel time: ... ms
GPU Sobel kernel time: ... ms
GPU speedup: ...x

CPU output: output/cpu_edges.pgm
GPU output: output/gpu_edges.pgm
```

Exact timings depend on the GPU and system.

## Project structure

```text
cuda_image_processing_capstone/
├── Makefile
├── README.md
├── src/
│   └── sobel.cu
├── data/
│   └── input.ppm
└── output/
```

## Presentation outline

### 1. Problem
Image edge detection is computationally parallel but can become expensive for large images.

### 2. CUDA approach
One GPU thread processes one output pixel.

### 3. CPU vs GPU
Run both implementations on the same image and compare execution times.

### 4. Result
Show the generated edge image and the measured GPU speedup.

### 5. CUDA concepts demonstrated
- CUDA kernel
- thread/block indexing
- device memory
- host-to-device copy
- device-to-host copy
- CUDA events
- synchronization
- parallel image processing

## Future improvements

- RGB processing
- Gaussian blur
- sharpening
- CUDA streams
- tiled/shared-memory Sobel implementation
- larger benchmark images
- comparison of global-memory and shared-memory kernels
