#ifndef VKFFT_WRAPPER_1D_H
#define VKFFT_WRAPPER_1D_H

// #include "vkfft_common.h"
#include "../../wrfft_common.h"         // Assumes this defines WrFFTErrors, etc.
#include <string>
#include "vkFFT.h"         // vkFFT main header
#include "half.hpp"        // For half precision support
#include "utils_VkFFT.h"   // Contains definition of VkGPU and related utilities

#ifdef __cplusplus
extern "C" {
#endif

// Structure representing a 1D vkFFT context.
// Notice that we now include a VkGPU (from utils_VkFFT.h) as an intrinsic field,
// so that all GPU initialization results are stored in the context.
typedef struct {
    VkFFTApplication app;              // vkFFT application instance.
    VkFFTConfiguration configuration;  // vkFFT configuration.
    VkGPU gpu;                         // The complete VkGPU structure.
    void* device_buffer;               // GPU buffer for input/output data.
    uint64_t buffer_size;              // Size of the GPU-side buffer.
    int n;                             // FFT size.
    WrFFTPrecision precision;          // Precision type (single, double, half).
} Vkfft1DContext;

/**
 * @brief Initializes the vkFFT backend specifically for 1D transforms.
 *
 * This function performs GPU/Backend initialization and populates the Vkfft1DContext,
 * including setting up the VkGPU structure. It must be called once before planning.
 *
 * @param contextOut Output pointer that will receive an initialized Vkfft1DContext.
 * @return WrFFTErrors WRFFT_SUCCESS on success or an appropriate error code on failure.
 */
WrFFTErrors vkfft1d_initialize(Vkfft1DContext** contextOut);

// /**
//  * @brief Reads 1D binary data from a file.
//  *
//  * The file is expected to have interleaved doubles (real, imag) that are converted to PrecType.
//  *
//  * @param filename The name of the file.
//  * @param input Preallocated buffer for data (interleaved).
//  * @param n Number of complex elements.
//  */
// void vkfft1d_read_binary(const std::string& filename, PrecType* input, int n);

// /**
//  * @brief Writes 1D binary data to a file.
//  *
//  * @param filepath Output file path.
//  * @param output Buffer containing data (interleaved).
//  * @param n Number of complex elements.
//  */
// void vkfft1d_write_binary(const std::string& filepath, PrecType* output, int n);

/**
 * @brief Creates a 1D vkFFT plan.
 *
 * This function assumes that the vkFFT backend is already initialized in the context.
 * It allocates device memory for input and output and sets up the vkFFT configuration.
 *
 * @param n Size of the FFT (number of complex elements).
 * @param context Pointer to an already-initialized Vkfft1DContext (from vkfft1d_initialize).
 * @param precision Precision type (single, double, half).
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an error code on failure.
 */
WrFFTErrors vkfft1d_plan(int n, Vkfft1DContext* context, WrFFTPrecision precision);

/**
 * @brief Executes the 1D FFT using the provided context.
 *
 * @param context Pointer to a valid Vkfft1DContext.
 * @param host_data Pointer to the host buffer where the result will be copied.
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an error code on failure.
 */
WrFFTErrors vkfft1d_execute(Vkfft1DContext* context, void* host_data);

/**
 * @brief Cleans up the vkFFT 1D context, releasing device memory and deinitializing vkFFT.
 *
 * @param context Pointer to the Vkfft1DContext.
 * @return WrFFTErrors WRFFT_SUCCESS on success.
 */
WrFFTErrors vkfft1d_cleanup(Vkfft1DContext* context);

#ifdef __cplusplus
}
#endif

#endif // VKFFT_WRAPPER_1D_H
