NVCC := nvcc
TARGET := cuda_sobel
SRC := src/sobel.cu

NVCCFLAGS := -O2 -std=c++14

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) data/input.ppm output/gpu_edges.pgm

clean:
	rm -f $(TARGET)
	rm -f output/cpu_edges.pgm output/gpu_edges.pgm

.PHONY: all run clean
