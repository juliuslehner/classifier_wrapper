# -------------------------------------------------------------------
# Makefile: Build cuFFT wrappers (1D + 2D), wrfft library and test
# -------------------------------------------------------------------

# --- paths -----------------------------------------------------------
ORT_HOME   := $(HOME)/ort-gpu
CUDA_HOME  := /usr/local/cuda
BUILD_DIR  := build

# --- toolchain -------------------------------------------------------
NVCC      := nvcc        # we’ll use nvcc for every TU – simplest
CXXFLAGS  := -std=c++20 -O3 \
             -I$(ORT_HOME)/include      \
             -I$(HOME)/include          \
             -I$(CUDA_HOME)/include     # cuda_fp16 intrinsics

LDFLAGS   := -L$(ORT_HOME)/lib  \
             -lonnxruntime -lonnxruntime_providers_cuda \
             -L$(CUDA_HOME)/lib64 -lcufft -lcudart \
             -Xcompiler -pthread -ldl

# --- sources / targets ----------------------------------------------
# cuFFT wrappers
WRAPPER1D_SRC := cufft_wrapper/cufft_wrapper_1d.cpp
WRAPPER1D_LIB := libcufft_wrapper_1d.a

WRAPPER2D_SRC := cufft_wrapper/cufft_wrapper_2d.cpp
WRAPPER2D_LIB := libcufft_wrapper_2d.a

# wrfft core
WRFFT_SRC   := wrfft.cpp
WRFFT_LIB   := libwrfft.a

# test
TEST_SRC    := test_wrfft.cpp
TEST_BIN    := test_wrfft

# -------------------------------------------------------------------
.PHONY: all wrapper wrfft test clean
all: wrapper wrfft test

# -------------------------------------------------------------------
# 1) Build both static cuFFT wrappers
# -------------------------------------------------------------------
wrapper: | $(BUILD_DIR)
	@echo "[wrapper] building 1D wrapper ($(WRAPPER1D_LIB))"
	$(NVCC) -std=c++17 -c $(WRAPPER1D_SRC) -o $(BUILD_DIR)/cufft_wrapper_1d.o
	ar rcs $(BUILD_DIR)/$(WRAPPER1D_LIB) $(BUILD_DIR)/cufft_wrapper_1d.o
	@echo "[wrapper] building 2D wrapper ($(WRAPPER2D_LIB))"
	$(NVCC) -std=c++17 -c $(WRAPPER2D_SRC) -o $(BUILD_DIR)/cufft_wrapper_2d.o
	ar rcs $(BUILD_DIR)/$(WRAPPER2D_LIB) $(BUILD_DIR)/cufft_wrapper_2d.o
	@rm -f $(BUILD_DIR)/cufft_wrapper_1d.o \
	       $(BUILD_DIR)/cufft_wrapper_2d.o

# -------------------------------------------------------------------
# 2) Build the wrfft static library (including both wrappers)
# -------------------------------------------------------------------
wrfft: wrapper | $(BUILD_DIR)
	@echo "[wrfft]  building $(WRFFT_LIB)"
	$(NVCC) $(CXXFLAGS) -c $(WRFFT_SRC) -o $(BUILD_DIR)/wrfft.o
	ar rcs $(BUILD_DIR)/$(WRFFT_LIB)             \
	       $(BUILD_DIR)/wrfft.o                  \
	       $(BUILD_DIR)/$(WRAPPER1D_LIB)        \
	       $(BUILD_DIR)/$(WRAPPER2D_LIB)
	@rm -f $(BUILD_DIR)/wrfft.o

# -------------------------------------------------------------------
# 3) Build the standalone test program
# -------------------------------------------------------------------
test: wrfft
	@echo "[test]   building $(TEST_BIN)"
	$(NVCC) $(CXXFLAGS) -I. $(TEST_SRC)          \
	      -L$(BUILD_DIR)                        \
	      -lwrfft                               \
	      -lcufft_wrapper_1d                    \
	      -lcufft_wrapper_2d                    \
	      $(LDFLAGS)                            \
	      -o $(TEST_BIN)

# -------------------------------------------------------------------
# Utility targets
# -------------------------------------------------------------------
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TEST_BIN)
