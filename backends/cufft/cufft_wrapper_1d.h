#ifndef CUFFT_WRAPPER_1D_H
#define CUFFT_WRAPPER_1D_H

// #include "../../wrfft.h"    // for WrFFTErrors, etc.
// #include "cufft_common.h"
#include "../../wrfft_common.h" 
#include <cufft.h>
#include <cuda_fp16.h>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

// Structure representing a 1D CUFFT plan along with pre-allocated device memory.
typedef struct {
    cufftHandle plan;
    int size;              // FFT size (n)
    void* d_input;          // Device pointer for input data
    void* d_output;         // Device pointer for output data
    WrFFTPrecision precision; // Precision of the FFT (single, double, half)
} Cufft1DContext;

// /**
//  * @brief Read interleaved double-precision real+imag data from disk,
//  *        convert to ComplexType and copy into GPU buffer.
//  *
//  * @param filename The file to read.
//  * @param input    Pointer to pre-allocated host buffer [n].
//  * @param n        Number of ComplexType elements to read.
//  * @return WrFFTErrors WRFFT_SUCCESS on success; WRFFT_ERROR_LIBRARY_FAILURE on I/O or convert error.
//  */
// WrFFTErrors cufft_read_binary(const std::string& filename, void* input, int n, WrFFTPrecision precision);

// /**
//  * @brief Write the host ComplexType buffer back out as interleaved double-precision
//  *        real+imag binary.
//  *
//  * @param filepath     Output file path.
//  * @param host_output  Pointer to host buffer [n].
//  * @param n            Number of elements to write.
//  * @return WrFFTErrors WRFFT_SUCCESS on success; WRFFT_ERROR_LIBRARY_FAILURE on I/O error.
//  */
// WrFFTErrors cufft_write_binary(const std::string& filepath, void* output, int n, WrFFTPrecision precision);

/**
 * @brief Copy the host data to the device and execute a 1D FFT on it.
 *
 * @param context   Pointer to a valid Cufft1DContext.
 * @param host_data Pointer to the host buffer where the result will be copied.
 * @param direction CUFFT_FORWARD or CUFFT_INVERSE.
 * @return WrFFTErrors WRFFT_SUCCESS on success.
 */
WrFFTErrors cufft1d_execute(Cufft1DContext* context, void* host_data, int direction);

/**
 * @brief Allocate and initialize a 1D cuFFT plan and device buffers for length n.
 *
 * @param n           FFT length.
 * @param contextOut  Pointer to receive the newly allocated Cufft1DContext*.
 * @param precision   Precision of the FFT (single, double, half).
 * @return WrFFTErrors WRFFT_SUCCESS on success; error code otherwise.
 */
WrFFTErrors cufft1d_plan(int n, Cufft1DContext** contextOut, WrFFTPrecision precision);

/**
 * @brief Cleanup: destroy plan, free device memory, delete context.
 *
 * @param context Pointer to the Cufft1DContext to destroy.
 * @return WrFFTErrors WRFFT_SUCCESS on success.
 */
WrFFTErrors cufft1d_cleanup(Cufft1DContext* context);

#ifdef __cplusplus
}
#endif

#endif // CUFFT_WRAPPER_1D_H
