#include "cufft_wrapper_1d.h"
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error-checking macros mapping low-level errors to WrFFTErrors.
#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                             \
    {                                                 \
        cufftResult res = (call);                     \
        if (res != CUFFT_SUCCESS) {                   \
            return WRFFT_ERROR_LIBRARY_FAILURE;       \
        }                                             \
    }
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(error)                                                   \
    {                                                                       \
        cudaError_t res = (error);                                          \
        if (res != cudaSuccess) {                                           \
            std::cerr << cudaGetErrorString(res) << " at "                  \
                      << __FILE__ << ":" << __LINE__ << std::endl;       \
            return WRFFT_ERROR_LIBRARY_FAILURE;                             \
        }                                                                   \
    }
#endif

static size_t elem_bytes(CufftPrecision p) {
    switch(p) {
      case CUFFT_PREC_SINGLE: return sizeof(cufftComplex);
      case CUFFT_PREC_DOUBLE: return sizeof(cufftDoubleComplex);
      case CUFFT_PREC_HALF:   return sizeof(half2);
    }
    return 0;
  }

// -------------------------------------------------------------------
// File I/O Functions (now return error codes instead of throwing)
// -------------------------------------------------------------------
// WrFFTErrors cufft_read_binary(const std::string& filename, void* d_input, int n, CufftPrecision prec) {
//     // 1) Read into a host-side buffer:
//     // std::vector<ComplexType> h_input(n);
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//     return WRFFT_ERROR_INVALID_INPUT;
//     }
//     size_t bytes = elem_bytes(prec) * n;
//     switch (prec) {
//         case CUFFT_PREC_SINGLE: {
//           // host buffer of float‐complex
//           std::vector<cufftComplex> h_input(n);
//           double real, imag;
//           for (int i = 0; i < n; ++i) {
//             file.read((char*)&real, sizeof(double));
//             file.read((char*)&imag, sizeof(double));
//             if (!file) return WRFFT_ERROR_INVALID_INPUT;
//             h_input[i].x = static_cast<float>(real);
//             h_input[i].y = static_cast<float>(imag);
//           }
//           file.close();
//           CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
//           break;
//         }
//         case CUFFT_PREC_DOUBLE: {
//           std::vector<cufftDoubleComplex> h_input(n);
//           double real, imag;
//           for (int i = 0; i < n; ++i) {
//             file.read((char*)&real, sizeof(double));
//             file.read((char*)&imag, sizeof(double));
//             if (!file) return WRFFT_ERROR_INVALID_INPUT;
//             h_input[i].x = real;
//             h_input[i].y = imag;
//           }
//           file.close();
//           CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
//           break;
//         }
//         case CUFFT_PREC_HALF: {
//           std::vector<half2> h_input(n);
//           double real, imag;
//           for (int i = 0; i < n; ++i) {
//             file.read((char*)&real, sizeof(double));
//             file.read((char*)&imag, sizeof(double));
//             if (!file) return WRFFT_ERROR_INVALID_INPUT;
//             h_input[i].x = __double2half(real);
//             h_input[i].y = __double2half(imag);
//           }
//           file.close();
//           CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
//           break;
//         }
//       }
//       return WRFFT_SUCCESS;
//   }
// }

// WrFFTErrors cufft_write_binary(const std::string& filepath, ComplexType* d_output, int n) {
//     // 1) Copy device→host
//     std::vector<ComplexType> h_output(n);
//     cudaError_t cudaErr = cudaMemcpy(h_output.data(), d_output, n * sizeof(ComplexType), cudaMemcpyDeviceToHost);
//     if (cudaErr != cudaSuccess) {
//         return WRFFT_ERROR_LIBRARY_FAILURE;
//     }

//     // 2) Write host buffer to disk
//     std::ofstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         return WRFFT_ERROR_INVALID_INPUT;
//     }

//     for (int i = 0; i < n; ++i) {
//         double real = static_cast<double>(h_output[i].x);
//         double imag = static_cast<double>(h_output[i].y);
//         file.write(reinterpret_cast<const char*>(&real), sizeof(double));
//         file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
//         if (!file) {
//             return WRFFT_ERROR_LIBRARY_FAILURE;
//         }
//     }
//     file.close();

//     return WRFFT_SUCCESS;
// }

// -------------------------------------------------------------------
// Plan, Execute, and Cleanup Functions
// -------------------------------------------------------------------
WrFFTErrors cufft1d_plan(int n, Cufft1DContext** contextOut, CufftPrecision prec) {
    if (contextOut == nullptr) return WRFFT_ERROR_INVALID_INPUT;
    Cufft1DContext* context = new Cufft1DContext;
    context->size = n;
    context->precision = prec;
    context->d_input = nullptr;
    context->d_output = nullptr;

    // Allocate device memory
    size_t bytes = elem_bytes(prec) * n;
    CUDA_CHECK(cudaMalloc((void**)&(context->d_input), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(context->d_output), bytes));

    // Create CUFFT plan
    cufftResult res = cufftCreate(&(context->plan));
    if (res != CUFFT_SUCCESS) {
        cudaFree(context->d_input);
        cudaFree(context->d_output);
        delete context;
        return WRFFT_ERROR_LIBRARY_FAILURE;
    }
    if (prec == CUFFT_PREC_SINGLE) {
        res = cufftPlan1d(&(context->plan), n, CUFFT_C2C, 1);
    } else if (prec == CUFFT_PREC_DOUBLE) {
        res = cufftPlan1d(&(context->plan), n, CUFFT_Z2Z, 1);
    } else {
        long long sig_size = n;
        size_t ws = 0;
        res = cufftXtMakePlanMany(context->plan,
                              1, &sig_size,
                              NULL, 1, 1, CUDA_C_16F,
                              NULL, 1, 1, CUDA_C_16F,
                              1, &ws, CUDA_C_16F);
    }
   
    if (res != CUFFT_SUCCESS) {
        cufftDestroy(context->plan);
        cudaFree(context->d_input);
        cudaFree(context->d_output);
        delete context;
        return WRFFT_ERROR_LIBRARY_FAILURE;
    }
    *contextOut = context;
    return WRFFT_SUCCESS;
}

WrFFTErrors cufft1d_execute(Cufft1DContext* context, int direction) {
    if (!context || !context->d_input || !context->d_output) return WRFFT_ERROR_INVALID_INPUT;
    cufftResult res;
    switch (context->precision) {
      case CUFFT_PREC_SINGLE:
        res = cufftExecC2C(context->plan,
             (cufftComplex*)context->d_input,
             (cufftComplex*)context->d_output,
             direction);
        break;
      case CUFFT_PREC_DOUBLE:
        res = cufftExecZ2Z(context->plan,
             (cufftDoubleComplex*)context->d_input,
             (cufftDoubleComplex*)context->d_output,
             direction);
        break;

      case CUFFT_PREC_HALF:
        // most direct for half: treat as C2C of 16-bit real+imag
        res = cufftXtExec(context->plan,
             context->d_input, context->d_output,
             direction);
        break;
    }
    if (res != CUFFT_SUCCESS) return WRFFT_ERROR_LIBRARY_FAILURE;
    // 5) Synchronize to catch any async errors
    CUDA_CHECK(cudaDeviceSynchronize());
    return WRFFT_SUCCESS;
}

WrFFTErrors cufft1d_cleanup(Cufft1DContext* context) {
    if (!context) return WRFFT_SUCCESS;
    cufftDestroy(context->plan);
    if (context->d_input) {
        cudaFree(context->d_input);
        context->d_input = nullptr;
    }
    if (context->d_output) {
        cudaFree(context->d_output);
        context->d_output = nullptr;
    }
    delete context;
    return WRFFT_SUCCESS;
}

#ifdef __cplusplus
}
#endif