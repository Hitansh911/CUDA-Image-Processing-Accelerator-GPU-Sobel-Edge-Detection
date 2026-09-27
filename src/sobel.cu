#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;
};

static void skipComments(std::istream& in) {
    char c;
    while (in >> std::ws && in.peek() == '#') {
        std::string line;
        std::getline(in, line);
    }
}

static Image readPPM(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    std::string magic;
    in >> magic;
    if (magic != "P6") {
        throw std::runtime_error("Only binary P6 PPM files are supported.");
    }

    skipComments(in);
    int width, height, maxValue;
    in >> width;
    skipComments(in);
    in >> height;
    skipComments(in);
    in >> maxValue;
    in.get();

    if (width <= 0 || height <= 0 || maxValue != 255) {
        throw std::runtime_error("Invalid PPM header.");
    }

    Image img;
    img.width = width;
    img.height = height;
    img.rgb.resize(static_cast<size_t>(width) * height * 3);

    in.read(reinterpret_cast<char*>(img.rgb.data()),
            static_cast<std::streamsize>(img.rgb.size()));

    if (!in) {
        throw std::runtime_error("Could not read complete PPM pixel data.");
    }

    return img;
}

static void writePGM(const std::string& filename,
                     int width,
                     int height,
                     const std::vector<unsigned char>& pixels) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot create output file: " + filename);
    }

    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()),
              static_cast<std::streamsize>(pixels.size()));
}

static std::vector<unsigned char>
grayscaleCPU(const Image& img) {
    std::vector<unsigned char> gray(
        static_cast<size_t>(img.width) * img.height);

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            size_t i = (static_cast<size_t>(y) * img.width + x) * 3;
            float value = 0.299f * img.rgb[i] +
                          0.587f * img.rgb[i + 1] +
                          0.114f * img.rgb[i + 2];

            gray[static_cast<size_t>(y) * img.width + x] =
                static_cast<unsigned char>(
                    std::min(255.0f, std::max(0.0f, value)));
        }
    }

    return gray;
}

static std::vector<unsigned char>
sobelCPU(const std::vector<unsigned char>& gray,
         int width,
         int height) {
    std::vector<unsigned char> output(
        static_cast<size_t>(width) * height, 0);

    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel =
                        gray[(y + ky) * width + (x + kx)];

                    sumX += pixel * gx[ky + 1][kx + 1];
                    sumY += pixel * gy[ky + 1][kx + 1];
                }
            }

            float magnitude =
                std::sqrt(static_cast<float>(
                    sumX * sumX + sumY * sumY));

            output[y * width + x] =
                static_cast<unsigned char>(
                    std::min(255.0f, magnitude));
        }
    }

    return output;
}

__global__ void sobelKernel(const unsigned char* gray,
                            unsigned char* output,
                            int width,
                            int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || y == 0 ||
        x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int sumX = 0;
    int sumY = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int pixel = gray[(y + ky) * width + (x + kx)];

            sumX += pixel * gx[ky + 1][kx + 1];
            sumY += pixel * gy[ky + 1][kx + 1];
        }
    }

    float magnitude =
        sqrtf(static_cast<float>(
            sumX * sumX + sumY * sumY));

    output[y * width + x] =
        static_cast<unsigned char>(
            fminf(255.0f, magnitude));
}

static std::vector<unsigned char>
sobelGPU(const std::vector<unsigned char>& gray,
         int width,
         int height,
         float& kernelMilliseconds) {
    const size_t bytes =
        static_cast<size_t>(width) * height * sizeof(unsigned char);

    unsigned char* d_gray = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gray, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_gray,
        gray.data(),
        bytes,
        cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    sobelKernel<<<grid, block>>>(
        d_gray,
        d_output,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(
        &kernelMilliseconds,
        start,
        stop));

    std::vector<unsigned char> output(
        static_cast<size_t>(width) * height);

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}

static bool compareImages(
    const std::vector<unsigned char>& a,
    const std::vector<unsigned char>& b) {
    if (a.size() != b.size()) {
        return false;
    }

    size_t mismatches = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        int diff =
            std::abs(static_cast<int>(a[i]) -
                     static_cast<int>(b[i]));

        if (diff > 1) {
            ++mismatches;
        }
    }

    return mismatches == 0;
}

int main(int argc, char** argv) {
    try {
        std::string input =
            argc > 1 ? argv[1] : "data/input.ppm";

        std::string output =
            argc > 2 ? argv[2] : "output/gpu_edges.pgm";

        std::cout << "CUDA Image Processing Accelerator\n";
        std::cout << "Input: " << input << "\n";

        Image image = readPPM(input);

        std::cout << "Image: "
                  << image.width << " x "
                  << image.height << "\n\n";

        auto cpuGrayStart =
            std::chrono::high_resolution_clock::now();

        std::vector<unsigned char> gray =
            grayscaleCPU(image);

        auto cpuGrayEnd =
            std::chrono::high_resolution_clock::now();

        double grayMs =
            std::chrono::duration<double, std::milli>(
                cpuGrayEnd - cpuGrayStart).count();

        auto cpuStart =
            std::chrono::high_resolution_clock::now();

        std::vector<unsigned char> cpuOutput =
            sobelCPU(gray, image.width, image.height);

        auto cpuEnd =
            std::chrono::high_resolution_clock::now();

        double cpuMs =
            std::chrono::duration<double, std::milli>(
                cpuEnd - cpuStart).count();

        float gpuMs = 0.0f;

        std::vector<unsigned char> gpuOutput =
            sobelGPU(
                gray,
                image.width,
                image.height,
                gpuMs);

        std::string cpuFile = "output/cpu_edges.pgm";

        writePGM(
            cpuFile,
            image.width,
            image.height,
            cpuOutput);

        writePGM(
            output,
            image.width,
            image.height,
            gpuOutput);

        bool correct =
            compareImages(cpuOutput, gpuOutput);

        std::cout << "CPU grayscale time: "
                  << grayMs << " ms\n";

        std::cout << "CPU Sobel time: "
                  << cpuMs << " ms\n";

        std::cout << "GPU Sobel kernel time: "
                  << gpuMs << " ms\n";

        if (gpuMs > 0.0f) {
            std::cout << "GPU kernel speedup: "
                      << cpuMs / gpuMs << "x\n";
        }

        std::cout << "CPU/GPU result match: "
                  << (correct ? "YES" : "NO") << "\n";

        std::cout << "CPU output: "
                  << cpuFile << "\n";

        std::cout << "GPU output: "
                  << output << "\n";

        return correct ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
