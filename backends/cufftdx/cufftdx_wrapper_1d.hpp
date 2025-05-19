#ifndef CUFFTDX_WRAPPER_1D_HPP
#define CUFFTDX_WRAPPER_1D_HPP

#include "../../wrfft_common.h"
// #include <cufftdx_common.hpp>
#include <string>
// #include <cufftdx.hpp>
// #include <cuda_fp16.h>
// #include "block_io.hpp"
// #include "common.hpp"
// #include "random.hpp"

// #ifdef __cplusplus
// extern "C" {
// #endif

/// Opaque context holding FFT size and device buffer pointer.
typedef struct {
    unsigned int  size;         //< FFT length (must match a supported power-of-two)
    void*  d_data;              //< Pointer to device (managed) memory holding FFT data
    WrFFTPrecision precision; //< Precision of the FFT (single, double, half)
} Cufftdx1DContext;

// /**
//  * Read interleaved double-precision real+imag samples from a binary file
//  * into a pre-allocated ComplexType buffer.
//  *
//  * @param filename   Path to the input file.
//  * @param input      Host or managed pointer to an array of ComplexType.
//  * @param n          Number of complex samples to read.
//  * @param precision  Precision of the FFT (single, double, half).
//  */
// WrFFTErrors cufftdx1d_read_binary(const std::string& filename, void* input, int n, WrFFTPrecision precision);

// /**
//  * Write interleaved double-precision real+imag samples from a ComplexType
//  * buffer out to a binary file.
//  *
//  * @param filename   Path to the output file.
//  * @param output     Host or managed pointer to an array of ComplexType.
//  * @param n          Number of complex samples to write.
//  */
// WrFFTErrors cufftdx1d_write_binary(const std::string& filename, ComplexType* output, int n);

/**
 * Allocate and initialize a cufftdx FFT context for 1D transforms.
 * This allocates device (managed) memory for `n` complex points
 * and sets `contextOut->size = n`.
 *
 * @param n           FFT length (power of two).
 * @param contextOut  On success, receives a newly allocated context.
 * @param precision   Precision of the FFT (single, double, half).
 * @return WRFFT_SUCCESS or an error code.
 */
WrFFTErrors cufftdx1d_plan(int n, Cufftdx1DContext** contextOut, WrFFTPrecision precision);

/**
 * Execute the FFT stored in `context` in-place.
 * The direction is fixed to forward complex-to-complex.
 *
 * @param context     Initialized Cufftdx1DContext from plan().
 * @param host_data   Pointer to FFT input/output data on the host.
 * @return WRFFT_SUCCESS or an error code.
 */
WrFFTErrors cufftdx1d_execute(Cufftdx1DContext* context, void* host_data);

/**
 * Free the FFT context and its associated device memory.
 *
 * @param context   The context returned by cufftdx1d_plan().
 * @return WRFFT_SUCCESS or an error code.
 */
WrFFTErrors cufftdx1d_cleanup(Cufftdx1DContext* context);

// #ifdef __cplusplus
// }
// #endif

#endif // CUFFTDX_WRAPPER_1D_HPP
