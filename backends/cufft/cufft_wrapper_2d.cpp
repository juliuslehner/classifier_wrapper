#include "cufft_wrapper_2d.h"
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
                      << __FILE__ << ":" << __LINE__ << std::endl;          \
            return WRFFT_ERROR_LIBRARY_FAILURE;                             \
        }                                                                   \
    }
#endif

static size_t elem_bytes(WrFFTPrecision p) {
    switch(p) {
      case WRFFT_PREC_SINGLE: return sizeof(cufftComplex);
      case WRFFT_PREC_DOUBLE: return sizeof(cufftDoubleComplex);
      case WRFFT_PREC_HALF:   return sizeof(half2);
    }
    return 0;
  }

// // -------------------------------------------------------------------
// // File I/O Functions for 2D Data
// // -------------------------------------------------------------------
// WrFFTErrors cufft_read_binary_2d(const std::string& filename, ComplexType* d_input, int nx, int ny) {
//     const size_t total = size_t(nx) * size_t(ny);
//     std::vector<ComplexType> h_input(total);

//     // 1) Read into host buffer
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         return WRFFT_ERROR_INVALID_INPUT;
//     }

//     double real, imag;
//     for (size_t i = 0; i < total; ++i) {
//         file.read(reinterpret_cast<char*>(&real), sizeof(double));
//         file.read(reinterpret_cast<char*>(&imag), sizeof(double));
//         if (!file) {
//             return WRFFT_ERROR_INVALID_INPUT;
//         }
// #ifdef HALF_PRECISION
//         h_input[i].x = __double2half(real);
//         h_input[i].y = __double2half(imag);
// #else
//         h_input[i].x = static_cast<PrecType>(real);
//         h_input[i].y = static_cast<PrecType>(imag);
// #endif
//     }
//     file.close();

//     // 2) Copy up to device
//     cudaError_t cerr = cudaMemcpy(d_input, h_input.data(), total * sizeof(ComplexType), cudaMemcpyHostToDevice);
//     return (cerr == cudaSuccess ? WRFFT_SUCCESS : WRFFT_ERROR_LIBRARY_FAILURE);
// }

// WrFFTErrors cufft_write_binary_2d(const std::string& filepath, ComplexType* d_output, int nx, int ny) {
//     const size_t total = size_t(nx) * size_t(ny);
//     std::vector<ComplexType> h_output(total);

//     // 1) Copy down from device
//     cudaError_t cerr = cudaMemcpy(h_output.data(), d_output, total * sizeof(ComplexType), cudaMemcpyDeviceToHost);
//     if (cerr != cudaSuccess) {
//         return WRFFT_ERROR_LIBRARY_FAILURE;
//     }

//     // 2) Write host buffer to disk
//     std::ofstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         return WRFFT_ERROR_INVALID_INPUT;
//     }

//     for (size_t i = 0; i < total; ++i) {
//         double real = static_cast<double>(h_output[i].x);
//         double imag = static_cast<double>(h_output[i].y);
//         file.write(reinterpret_cast<char*>(&real), sizeof(double));
//         file.write(reinterpret_cast<char*>(&imag), sizeof(double));
//         if (!file) {
//             return WRFFT_ERROR_LIBRARY_FAILURE;
//         }
//     }
//     file.close();
//     return WRFFT_SUCCESS;
// }

// -------------------------------------------------------------------
// Plan, Execute, and Cleanup Functions for 2D FFT
// -------------------------------------------------------------------
WrFFTErrors cufft2d_plan(int nx, int ny, Cufft2DContext** contextOut, WrFFTPrecision precision) {
    if (contextOut == nullptr) return WRFFT_ERROR_INVALID_INPUT;
    Cufft2DContext* context = new Cufft2DContext;
    context->nx = nx;
    context->ny = ny;
    context->precision = precision;
    context->d_input = nullptr;
    context->d_output = nullptr;
    int total = nx * ny;

    // Allocate device memory for input and output arrays.
    size_t bytes = total * elem_bytes(precision);
    CUDA_CHECK(cudaMallocManaged((void**)&(context->d_input),bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&(context->d_output), bytes));

    // Create CUFFT plan.
    cufftResult res = cufftCreate(&(context->plan));
    if (res != CUFFT_SUCCESS) {
        cudaFree(context->d_input);
        cudaFree(context->d_output);
        delete context;
        return WRFFT_ERROR_LIBRARY_FAILURE;
    }
    switch (context->precision) {
        case WRFFT_PREC_SINGLE:
            res = cufftPlan2d(&(context->plan), nx, ny, CUFFT_C2C);
            break;
        case WRFFT_PREC_DOUBLE:
            res = cufftPlan2d(&(context->plan), nx, ny, CUFFT_Z2Z);
            break;
        case WRFFT_PREC_HALF:
            long long dims[2] = { nx, ny };
            size_t ws = 0;
            res = cufftXtMakePlanMany(context->plan, 2, dims, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F);
            break;
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

WrFFTErrors cufft2d_execute(Cufft2DContext* context, void* host_data, int direction) {
    if (!context || !host_data) return WRFFT_ERROR_INVALID_INPUT;
    cufftResult res;
    size_t bytes = elem_bytes(context->precision) * context->nx * context->ny;

    // Copy device output to host
    // CUDA_CHECK(cudaMemcpy(context->d_input, host_data, bytes, cudaMemcpyHostToDevice));
    int n = context->nx * context->ny;
    ComplexData* host = reinterpret_cast<ComplexData*>(host_data);
    switch(context->precision) {
        case WRFFT_PREC_HALF: {
            auto d_input = reinterpret_cast<half2*>(context->d_input);
            for (int i = 0; i < n; ++i) {
                d_input[i].x = __double2half(static_cast<double>(host[i].x));
                d_input[i].y = __double2half(static_cast<double>(host[i].y));
            }
            break;
        }
        case WRFFT_PREC_SINGLE: {
            auto d_input = reinterpret_cast<cufftComplex*>(context->d_input);
            for (int i = 0; i < n; ++i) {
                d_input[i].x = static_cast<float>(host[i].x);
                d_input[i].y = static_cast<float>(host[i].y);
            }
            break;
        }
        case WRFFT_PREC_DOUBLE: {
            auto d_input = reinterpret_cast<cufftDoubleComplex*>(context->d_input);
            for (int i = 0; i < n; ++i) {
                d_input[i].x = static_cast<double>(host[i].x);
                d_input[i].y = static_cast<double>(host[i].y);
            }
            break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    switch (context->precision) {
        case WRFFT_PREC_SINGLE:
            res = cufftExecC2C(context->plan,
                (cufftComplex*)context->d_input,
                (cufftComplex*)context->d_output,
                direction);
            break;
        case WRFFT_PREC_DOUBLE:
            res = cufftExecZ2Z(context->plan, (cufftDoubleComplex*)context->d_input, (cufftDoubleComplex*)context->d_output, direction);
            break;
        case WRFFT_PREC_HALF:
            res = cufftXtExec(context->plan, context->d_input, context->d_output, CUFFT_FORWARD);
            break;
        default:
            return WRFFT_ERROR_INVALID_INPUT;
    }

    if (res != CUFFT_SUCCESS) return WRFFT_ERROR_LIBRARY_FAILURE;
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy d_output->host
    // CUDA_CHECK(cudaMemcpy(host_data, context->d_output, bytes, cudaMemcpyDeviceToHost));
    switch (context->precision) {
        case WRFFT_PREC_HALF: {
            auto d_output = reinterpret_cast<half2*>(context->d_output);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(__half2float(d_output[i].x));
                host[i].y = static_cast<double>(__half2float(d_output[i].y));
            }
            break;
        }
        case WRFFT_PREC_SINGLE: {
            auto d_output = reinterpret_cast<cufftComplex*>(context->d_output);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(d_output[i].x);
                host[i].y = static_cast<double>(d_output[i].y);
            }
            break;
        }
        case WRFFT_PREC_DOUBLE: {
            auto d_output = reinterpret_cast<cufftDoubleComplex*>(context->d_output);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(d_output[i].x);
                host[i].y = static_cast<double>(d_output[i].y);
            }
            break;
        }
    }
    
    return WRFFT_SUCCESS;
}

WrFFTErrors cufft2d_cleanup(Cufft2DContext* context) {
    if (context) {
        cufftDestroy(context->plan);
        if (context->d_input) cudaFree(context->d_input);
        if (context->d_output) cudaFree(context->d_output);
        delete context;
    }
    return WRFFT_SUCCESS;
}
