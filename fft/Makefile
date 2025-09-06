NVCC := nvcc
NVCC_FLAGS := -O3 -lineinfo
NVCC_ARCH := -arch=sm_61

INCLUDE_DIR := include
SRC_DIR := src

TARGET := fft_demo
SRC_FILES := $(SRC_DIR)/fft_radix2.cu $(SRC_DIR)/fft_stockham.cu $(SRC_DIR)/main.cu

all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -I$(INCLUDE_DIR) $^ -o $@

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
