#ifndef CUFFT_WRAPPER_2D_H
#define CUFFT_WRAPPER_2D_H

// #include "../../wrfft.h"    // Assumes this header defines WrFFTErrors, etc.
#include "cufft_common.h" 
#include "../wrfft_common.h" 
#include <cufft.h>
#include <cuda_fp16.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// Structure representing a 2D CUFFT context (plan and associated device memory).
typedef struct {
    cufftHandle plan;      // The CUFFT plan.
    int nx;                // FFT size in the X dimension.
    int ny;                // FFT size in the Y dimension.
    void* d_input;  // Device pointer for input data.
    void* d_output; // Device pointer for output data.
    CufftPrecision precision; // Precision of the FFT (single, double, half)
} Cufft2DContext;

// /**
//  * @brief Reads 2D binary data (interleaved doubles: real, imag)
//  *        from a file into a provided buffer.
//  *
//  * @param filename Name of the input file.
//  * @param input Preallocated array of ComplexType.
//  * @param nx Number of points in the X dimension.
//  * @param ny Number of points in the Y dimension.
//  */
// WrFFTErrors cufft_read_binary_2d(const std::string& filename, ComplexType* input, int nx, int ny);

// /**
//  * @brief Writes 2D binary data (interleaved real and imaginary parts converted to double)
//  *        from the provided buffer to a file.
//  *
//  * @param filepath Name of the output file.
//  * @param output Array of ComplexType to write.
//  * @param nx Number of points in the X dimension.
//  * @param ny Number of points in the Y dimension.
//  */
// WrFFTErrors cufft_write_binary_2d(const std::string& filepath, ComplexType* output, int nx, int ny);

/**
 * @brief Creates a 2D CUFFT context for an FFT of size (nx, ny) and allocates
 *        device memory for input and output.
 *
 * @param nx Number of points in the X dimension.
 * @param ny Number of points in the Y dimension.
 * @param contextOut Output pointer that will receive the allocated Cufft2DContext.
 * @param precision Precision of the FFT (single, double, half).
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an appropriate error code on failure.
 */
WrFFTErrors cufft2d_plan(int nx, int ny, Cufft2DContext** contextOut, CufftPrecision precision);

/**
 * @brief Executes the 2D FFT using the provided CUFFT context.
 *
 * The device memory pointers stored in the context are used.
 *
 * @param context Pointer to a valid Cufft2DContext.
 * @param direction Direction flag (CUFFT_FORWARD or CUFFT_INVERSE).
 * @return WrFFTErrors WRFFT_SUCCESS on success, or an appropriate error code on failure.
 */
WrFFTErrors cufft2d_execute(Cufft2DContext* context, int direction);

/**
 * @brief Cleans up the 2D CUFFT context and frees the allocated device memory.
 *
 * @param context Pointer to the Cufft2DContext.
 * @return WrFFTErrors WRFFT_SUCCESS on success.
 */
WrFFTErrors cufft2d_cleanup(Cufft2DContext* context);

#ifdef __cplusplus
}
#endif

#endif // CUFFT_WRAPPER_2D_H
