#ifndef CUFFTDX_WRAPPER_2D_HPP
#define CUFFTDX_WRAPPER_2D_HPP

#include "../../wrfft_common.h"
// #include "cufftdx_common.hpp"
#include <string>


/// Opaque context for a rectangular 2D FFT of size nx×ny.
typedef struct {
    unsigned int size_x;    // FFT length in X dimension (must be a power of two)
    unsigned int size_y;    // FFT length in Y dimension (must be a power of two)
    void* d_input;           // Pointer to device (managed) memory holding FFT input
    void* d_output;          // Pointer to device (managed) memory for FFT output
    WrFFTPrecision precision; // Precision of the FFT (single, double, half)  
    cudaStream_t stream;    // CUDA stream to use for the FFT
} Cufftdx2DContext;

/**
 * @brief Plan (allocate and load) a rectangular 2D FFT of dimension nx×ny.
 *
 * @param nx      The FFT size in X dimension (must be a power of two).
 * @param ny      The FFT size in Y dimension (must be a power of two).
 * @param contextOut  On success, receives a newly allocated context.
 * @param precision   Precision of the FFT (single, double).
 */
WrFFTErrors cufftdx2d_plan(int nx, int ny, Cufftdx2DContext** contextOut, WrFFTPrecision precision);

/**
 * @brief Execute the 2D FFT (and write the result to `outfile`).
 *
 * @param context  The FFT context created by cufftdx2d_plan().
 * @param host_data  Pointer to the host data buffer (input/output).
 */
WrFFTErrors cufftdx2d_execute(Cufftdx2DContext* context, void* host_data);

/**
 * @brief Free the context and its associated device memory.
 * 
 * @param context  The FFT context to clean up.
 */
WrFFTErrors cufftdx2d_cleanup(Cufftdx2DContext* context);

#endif // CUFFTDX_WRAPPER_2D_HPP
