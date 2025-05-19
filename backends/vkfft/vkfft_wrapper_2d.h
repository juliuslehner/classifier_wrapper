#ifndef VKFFT_WRAPPER_2D_H
#define VKFFT_WRAPPER_2D_H

// #include "vkfft_common.h"
#include "../../wrfft_common.h"         // Assumes this defines WrFFTErrors, etc.
#include <string>
#include "vkFFT.h"         // vkFFT main header
#include "half.hpp"        // For half precision support
#include "utils_VkFFT.h"   // Contains definition of VkGPU and related utilities

#ifdef __cplusplus
extern "C" {
#endif

// Structure representing a 2D vkFFT context.
// Notice that we now include a VkGPU (from utils_VkFFT.h) as an intrinsic field,
// so that all GPU initialization results are stored in the context.
typedef struct {
    VkFFTApplication app;              // vkFFT application instance.
    VkFFTConfiguration configuration;  // vkFFT configuration.
    VkGPU gpu;                         // The complete VkGPU structure.
    void*     device_buffer;           // GPU-side buffer pointer
    uint64_t  buffer_size;             // Size of device buffer.
    WrFFTPrecision precision;          // Precision type (single, double, half).
    int nx;                            // FFT size x-dimension.
    int ny;                            // FFT size y-dimension.
} Vkfft2DContext;

/**
 * @brief Initializes the vkFFT backend specifically for 2D transforms.
 *
 * This function performs GPU/Backend initialization and populates the Vkfft2DContext,
 * including setting up the VkGPU structure. It must be called once before planning.
 *
 * @param contextOut Output pointer that will receive an initialized Vkfft2DContext.
 * @return WrFFTErrors WRFFT_SUCCESS on success or an appropriate error code on failure.
 */
WrFFTErrors vkfft2d_initialize(Vkfft2DContext** contextOut);

// /**
//  * @brief Reads 2D binary data from a file.
//  *
//  * The file is expected to have interleaved doubles (real, imag) that are converted to PrecType.
//  *
//  * @param filename The name of the file.
//  * @param input Preallocated buffer for data (interleaved).
//  * @param n Number of complex elements.
//  */
// void vkfft2d_read_binary(const std::string& filename, PrecType* input, int n);

// /**
//  * @brief Writes 2D binary data to a file.
//  *
//  * @param filepath Output file path.
//  * @param output Buffer containing data (interleaved).
//  * @param n Number of complex elements.
//  */
// void vkfft2d_write_binary(const std::string& filepath, PrecType* output, int n);

/**
 * @brief Creates a 2D vkFFT plan.
 *
 * This function assumes that the vkFFT backend is already initialized in the context.
 * It allocates device memory for input and output and sets up the vkFFT configuration.
 *
 * @param nx      FFT size in x-dimension.
 * @param ny      FFT size in y-dimension.
 * @param context Pointer to an already-initialized Vkfft2DContext (from vkfft2d_initialize).
 * @param precision Precision type (single, double, half).
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an error code on failure.
 */
WrFFTErrors vkfft2d_plan(int nx, int ny, Vkfft2DContext* context, WrFFTPrecision precision);

/**
 * @brief Executes the 2D FFT using the provided context.
 *
 * @param context Pointer to a valid Vkfft2DContext.
 * @param host_data Pointer to the host buffer where the result will be copied.
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an error code on failure.
 */
WrFFTErrors vkfft2d_execute(Vkfft2DContext* context, void* host_data);

/**
 * @brief Cleans up the vkFFT 2D context, releasing device memory and deinitializing vkFFT.
 *
 * @param context Pointer to the Vkfft2DContext.
 * @return WrFFTErrors WRFFT_SUCCESS on success.
 */
WrFFTErrors vkfft2d_cleanup(Vkfft2DContext* context);

#ifdef __cplusplus
}
#endif

#endif // VKFFT_WRAPPER_2D_H
