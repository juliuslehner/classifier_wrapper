# -------------------------------------------------------------------
# Makefile for wrfft: cuFFT, cuFFTDx & VkFFT backends + test_wrfft
# -------------------------------------------------------------------

# Paths
CUDA_HOME       := /usr/local/cuda
ORT_HOME        := $(HOME)/ort-gpu
MATHDX_INCLUDE  := /u/home/lehnj/nvidia-mathdx-24.01.0/nvidia/mathdx/24.01/include
VKFFT_DIR       := /u/home/lehnj/benchmarking_suite/VkFFT

# Compilers
NVCC            := $(CUDA_HOME)/bin/nvcc
CXX             := g++

# Common flags
COMMON_FLAGS    := -std=c++17 -O3
CUDA_FLAGS      := -arch=sm_80 -DCUDA_ARCH=800

# VkFFT backend selection and flags
VKFFT_BACKEND   := cuda      # cuda | vulkan | hip | opencl | level_zero | metal
CUDA_TOOLKIT_ROOT_DIR := $(CUDA_HOME)

# VkFFT backend-specific flags (copied from the VkFFT makefile)
VULKAN_FLAGS    := -lvulkan -lspirv -lglslang
VKFFT_BACKEND_FLAGS := -DVKFFT_BACKEND=1 -DCUDA_TOOLKIT_ROOT_DIR=\"$(CUDA_TOOLKIT_ROOT_DIR)\"

# Include dirs for all backends + test
INCLUDES        := \
  -I. \
  -I$(CUDA_HOME)/include \
  -I$(MATHDX_INCLUDE) \
  -I$(VKFFT_DIR)/vkFFT \
  -I$(VKFFT_DIR)/half_lib \
  -I$(VKFFT_DIR)/benchmark_scripts/vkFFT_scripts/include \
  -I$(ORT_HOME)/include \
  -I$(HOME)/include     # for nlohmann/json.hpp

NVCCFLAGS       := $(COMMON_FLAGS) $(CUDA_FLAGS) $(INCLUDES) $(VKFFT_BACKEND_FLAGS)
CXXFLAGS        := $(COMMON_FLAGS) $(INCLUDES) $(VKFFT_BACKEND_FLAGS)

# Library dirs & libs
LIB_DIRS        := \
  -L$(CUDA_HOME)/lib64 \
  -L$(ORT_HOME)/lib
LIBS            := -lcudart -lcufft -lfftw3 -lfftw3l -lm -lnvidia-ml -lcuda -lnvrtc -lquadmath \
  				   -lonnxruntime -lonnxruntime_providers_cuda -lpthread

# Backend sources (with correct paths)
STATIC_LIB_DIR  := backends

# cuFFT sources
CUFFT_DIR       := $(STATIC_LIB_DIR)/cufft
CUFFT_WRAPPERS  := $(CUFFT_DIR)/cufft_wrapper_1d.cpp $(CUFFT_DIR)/cufft_wrapper_2d.cpp
CUFFT_OBJS      := $(CUFFT_WRAPPERS:.cpp=.o)

# cuFFTDx sources
CUFFTDX_DIR     := $(STATIC_LIB_DIR)/cufftdx
CUFFTDX_WRAPPERS:= $(CUFFTDX_DIR)/cufftdx_wrapper_1d.cu $(CUFFTDX_DIR)/cufftdx_wrapper_2d.cu
CUFFTDX_OBJS    := $(CUFFTDX_WRAPPERS:.cu=.o)

# VkFFT sources
VKFFT_LIB_DIR   := $(STATIC_LIB_DIR)/vkfft
VKFFT_WRAPPERS  := $(VKFFT_LIB_DIR)/vkfft_wrapper_1d.cpp $(VKFFT_LIB_DIR)/vkfft_wrapper_2d.cpp
VKFFT_UTILS     := $(VKFFT_LIB_DIR)/utils_VkFFT.cpp
VKFFT_SRCS      := $(VKFFT_WRAPPERS) $(VKFFT_UTILS)
VKFFT_OBJS      := $(VKFFT_SRCS:.cpp=.o)

# High-level wrapper sources
WRFFT_SRC       := wrfft.cpp
WRFFT_OBJ       := $(WRFFT_SRC:.cpp=.o)

# Test source
TEST_SRC        := test_wrfft.cpp
TEST_OBJ        := $(TEST_SRC:.cpp=.o)

# All objects
ALL_OBJS        := $(CUFFT_OBJS) $(CUFFTDX_OBJS) $(VKFFT_OBJS) $(WRFFT_OBJ) $(TEST_OBJ)

# Default target
.PHONY: all clean
all: test_wrfft

# Ensure directories exist
# $(shell mkdir -p $(CUFFT_DIR) $(CUFFTDX_DIR) $(VKFFT_LIB_DIR))

# Pattern rules for object compilation
$(CUFFT_DIR)/%.o: $(CUFFT_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUFFTDX_DIR)/%.o: $(CUFFTDX_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(VKFFT_LIB_DIR)/%.o: $(VKFFT_LIB_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link everything into the test executable
test_wrfft: $(ALL_OBJS)
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o $@

# Optional: Build a static library for the wrapper
libwrfft.a: $(CUFFT_OBJS) $(CUFFTDX_OBJS) $(VKFFT_OBJS) $(WRFFT_OBJ)
	ar rcs $@ $^

clean:
	rm -f $(ALL_OBJS) test_wrfft libwrfft.a