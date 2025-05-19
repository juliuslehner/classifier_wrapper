#include <complex>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <array>
#include <algorithm>
#include <string>
#include <iomanip>
#include <unistd.h>
#include "cufftdx_wrapper_1d.hpp"
#include "cufftdx_common.hpp"
// In cufftdx folder
#include <cufftdx.hpp>
#include <cuda_fp16.h>
#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

#ifndef CUDA_ARCH
#error "You must define CUDA_ARCH at compile time (e.g. -DCUDA_ARCH=800)"
#endif

// WrFFTErrors cufftdx1d_read_binary(const std::string& filename, ComplexType* input, int num_elements) {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         // throw std::runtime_error("Could not open file to read");
//         return WRFFT_ERROR_INVALID_INPUT;
//     }

//     double real, imag;
//     for (int i = 0; i < num_elements; ++i) {
//         file.read(reinterpret_cast<char*>(&real), sizeof(double));
//         file.read(reinterpret_cast<char*>(&imag), sizeof(double));
//         if (!file) {
//             // throw std::runtime_error("Unexpected end of file while reading: " + filename);
//             return WRFFT_ERROR_INVALID_INPUT;
//         }
// #ifdef HALF_PRECISION
//         float v1 = static_cast<float>(real);
//         float v2 = static_cast<float>(imag);
//         // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
//         input[i] = ComplexType {__half2 {v1, v2}, __half2 {v1, v2}};
// #elif defined(DOUBLE_PRECISION)
//         input[i].x = static_cast<double>(real);
//         input[i].y = static_cast<double>(imag);
// #else   
//         input[i].x = static_cast<float>(real);
//         input[i].y = static_cast<float>(imag);
// #endif
//     }
//     file.close();
//     return WRFFT_SUCCESS;
// }

// WrFFTErrors cufftdx1d_write_binary(const std::string& filepath, ComplexType* output, int n) {
//     std::ofstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         // throw std::runtime_error("Could not open file to write");
//         return WRFFT_ERROR_INVALID_INPUT;
//     }

//     for (int i = 0; i < n; ++i) {
// #ifdef HALF_PRECISION
//         double real = static_cast<double>(__half2float(output[i].x.x));
//         double imag = static_cast<double>(__half2float(output[i].x.y));
// #elif defined(DOUBLE_PRECISION)
//         double real = output[i].x;
//         double imag = output[i].y;
// #else
//         double real = static_cast<double>(output[i].x);
//         double imag = static_cast<double>(output[i].y);
// #endif
//         file.write(reinterpret_cast<const char*>(&real), sizeof(double));
//         file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
//     }
        
//     file.close();
//     return WRFFT_SUCCESS;
// }

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ unsigned char _smem[];
    auto* fft_shared_mem = reinterpret_cast<complex_type*>(_smem);
    FFT().execute(thread_data, fft_shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

template <unsigned S, unsigned int Arch, typename ValueType>
WrFFTErrors plan_impl_1d(Cufftdx1DContext* context) {
    if (context == nullptr) return WRFFT_ERROR_INVALID_INPUT;
    using namespace cufftdx;
    // 1) Build the FFT type
    using FFT_base = decltype(Block() + Type<fft_type::c2c>() 
                                        + Precision<ValueType>() 
                                        + SM<Arch>());
    using FFT_with_direction = decltype(FFT_base() + Direction<fft_direction::forward>());
    using FFT = decltype(FFT_with_direction() + Size<S>());
    using complex_type = typename FFT::value_type;
    // std::cout << "Allocating Memory" << std::endl;
    // Allocate managed memory for input/output
    complex_type* gpu_buf;
    constexpr size_t implicit_batching = FFT::implicit_type_batching;
    auto             size              = FFT::ffts_per_block / implicit_batching * cufftdx::size_of<FFT>::value;
    auto             size_bytes        = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&gpu_buf, size_bytes));   
    // Store and return
    context->d_data = gpu_buf;
    // std::cout << "Allocated GPU Memory\n" << std::endl;
    return WRFFT_SUCCESS;
}

template <unsigned int S, unsigned int Arch, typename ValueType>
WrFFTErrors execute_impl_1d(void* host_data, Cufftdx1DContext* context) {
    if (!context || !host_data) return WRFFT_ERROR_INVALID_INPUT;

    using namespace cufftdx;
    using FFT_base = decltype(Block() + Type<fft_type::c2c>() 
                                        + Precision<ValueType>() 
                                        + SM<Arch>());
    using FFT_with_direction = decltype(FFT_base() + Direction<fft_direction::forward>());
    using FFT = decltype(FFT_with_direction() + Size<S>());

    // Copy data from host to device
    using complex_t     = typename FFT::value_type;
    size_t count = cufftdx::size_of<FFT>::value;
    // Cast so we don't have issues with void* pointers
    ComplexData* host = reinterpret_cast<ComplexData*>(host_data);
    complex_t* d_data = reinterpret_cast<complex_t*>(context->d_data);
    if constexpr (std::is_same_v<ValueType, float>) {
        for (size_t i = 0; i < count; ++i) {
            d_data[i].x = static_cast<float>(host[i].x);
            d_data[i].y = static_cast<float>(host[i].y);
        }
    }
    else if constexpr (std::is_same_v<ValueType, double>) {
        for (size_t i = 0; i < count; ++i) {
            d_data[i].x = static_cast<double>(host[i].x);
            d_data[i].y = static_cast<double>(host[i].y);
        }
    }
    else{
        for (size_t i = 0; i < count; ++i) {
            float v1 = static_cast<float>(host[i].x);
            float v2 = static_cast<float>(host[i].y);
            // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
            d_data[i] = complex_t{__half2 {v1, v2}, __half2 {v1, v2}};
        }
    }
    // size_t bytes = count * sizeof(complex_t);
    // CUDA_CHECK_AND_EXIT(cudaMemcpy(context->d_data, host_data, bytes, cudaMemcpyHostToDevice));

    // Set up kernel launch
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));
    
    // Launch kernel
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(
        static_cast<typename FFT::value_type*>(d_data));
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy data back to host
    // CUDA_CHECK_AND_EXIT(cudaMemcpy(host_data, context->d_data, bytes, cudaMemcpyDeviceToHost));
    if constexpr (std::is_same_v<ValueType, float>) {
        for (size_t i = 0; i < count; ++i) {
            host[i].x = static_cast<double>(d_data[i].x);
            host[i].y = static_cast<double>(d_data[i].y);
        }
    }
    else if constexpr (std::is_same_v<ValueType, double>) {
        for (size_t i = 0; i < count; ++i) {
            host[i].x = d_data[i].x;
            host[i].y = d_data[i].y;
        }
    }
    else {
        for (size_t i = 0; i < count; ++i) {
            host[i].x = static_cast<double>(__half2float(d_data[i].x.x));
            host[i].y = static_cast<double>(__half2float(d_data[i].x.y));
        }
    }
    return WRFFT_SUCCESS;
}

// Dispatch for plan: templated on the SM architecture
WrFFTErrors cufftdx1d_plan(int n, Cufftdx1DContext** out, WrFFTPrecision precision) {
    if (!out) return WRFFT_ERROR_INVALID_INPUT;
    Cufftdx1DContext* context = new Cufftdx1DContext;
    context->size = n;
    context->d_data = nullptr;
    context->precision = precision;
    WrFFTErrors err;
    switch(precision) {
        case WRFFT_PREC_SINGLE:
            switch (n) {
                case 512: err = plan_impl_1d<512, CUDA_ARCH, float>(context); break;
                case 1024: err = plan_impl_1d<1024, CUDA_ARCH, float>(context); break;
                case 2048: err = plan_impl_1d<2048, CUDA_ARCH, float>(context); break;
                case 4096: err = plan_impl_1d<4096, CUDA_ARCH, float>(context); break;
                case 8192: err = plan_impl_1d<8192, CUDA_ARCH, float>(context); break;
                case 16384: err = plan_impl_1d<16384, CUDA_ARCH, float>(context); break;
                default: err =  WRFFT_ERROR_INVALID_INPUT; break;
            }
            break;
        case WRFFT_PREC_DOUBLE:
            switch (n) {
                case 512: err =  plan_impl_1d<512, CUDA_ARCH, double>(context); break;
                case 1024: err =  plan_impl_1d<1024, CUDA_ARCH, double>(context); break;
                case 2048: err =  plan_impl_1d<2048, CUDA_ARCH, double>(context); break;
                case 4096: err =  plan_impl_1d<4096, CUDA_ARCH, double>(context); break;
                case 8192: err = plan_impl_1d<8192, CUDA_ARCH, double>(context); break;
                default: err = WRFFT_ERROR_INVALID_INPUT; break;
            }
            break;
        case WRFFT_PREC_HALF:
            switch (n) {
                case 512: err = plan_impl_1d<512, CUDA_ARCH, __half>(context); break;
                case 1024: err = plan_impl_1d<1024, CUDA_ARCH, __half>(context); break;
                case 2048: err = plan_impl_1d<2048, CUDA_ARCH, __half>(context); break;
                case 4096: err = plan_impl_1d<4096, CUDA_ARCH, __half>(context); break;
                case 8192: err = plan_impl_1d<8192, CUDA_ARCH, __half>(context); break;
                case 16384: err = plan_impl_1d<16384, CUDA_ARCH, __half>(context); break;
                default: err = WRFFT_ERROR_INVALID_INPUT; break;
            }
            break;
        default:
            err = WRFFT_ERROR_INVALID_INPUT; break;
    }
    if (err != WRFFT_SUCCESS) {
        delete context;
        return err;
    }
  
    *out = context;      
    return WRFFT_SUCCESS;
}

// Dispatch for execute: tmpl’d on the same arch
WrFFTErrors cufftdx1d_execute(Cufftdx1DContext* context, void* host_data) {
    if (!context || !host_data) return WRFFT_ERROR_INVALID_INPUT;
  
    switch (context->precision) {
        case WRFFT_PREC_SINGLE:
            switch (context->size) {
                case 512: return execute_impl_1d<512, CUDA_ARCH, float>(host_data, context);
                case 1024: return execute_impl_1d<1024, CUDA_ARCH, float>(host_data, context);
                case 2048: return execute_impl_1d<2048, CUDA_ARCH, float>(host_data, context);
                case 4096: return execute_impl_1d<4096, CUDA_ARCH, float>(host_data, context);
                case 8192: return execute_impl_1d<8192, CUDA_ARCH, float>(host_data, context);
                case 16384: return execute_impl_1d<16384, CUDA_ARCH, float>(host_data, context);
                default: return WRFFT_ERROR_INVALID_INPUT;
            }
        case WRFFT_PREC_DOUBLE:
            switch (context->size) {
                case 512: return execute_impl_1d<512, CUDA_ARCH, double>(host_data, context);
                case 1024: return execute_impl_1d<1024, CUDA_ARCH, double>(host_data, context);
                case 2048: return execute_impl_1d<2048, CUDA_ARCH, double>(host_data, context);
                case 4096: return execute_impl_1d<4096, CUDA_ARCH, double>(host_data, context);
                case 8192: return execute_impl_1d<8192, CUDA_ARCH, double>(host_data, context);
                default: return WRFFT_ERROR_INVALID_INPUT;
            }
        case WRFFT_PREC_HALF:
            switch (context->size) {
                case 512: return execute_impl_1d<512, CUDA_ARCH, __half>(host_data, context);
                case 1024: return execute_impl_1d<1024, CUDA_ARCH, __half>(host_data, context);
                case 2048: return execute_impl_1d<2048, CUDA_ARCH, __half>(host_data, context);
                case 4096: return execute_impl_1d<4096, CUDA_ARCH, __half>(host_data, context);
                case 8192: return execute_impl_1d<8192, CUDA_ARCH, __half>(host_data, context);
                case 16384: return execute_impl_1d<16384, CUDA_ARCH, __half>(host_data, context);
                default: return WRFFT_ERROR_INVALID_INPUT;
            }
        default:
            return WRFFT_ERROR_INVALID_INPUT;
        }
}

WrFFTErrors cufftdx1d_cleanup(Cufftdx1DContext* context) {
    if (context == nullptr) {
        return WRFFT_ERROR_INVALID_INPUT;
    }
    // free the GPU buffer
    CUDA_CHECK_AND_EXIT(cudaFree(context->d_data));
    // delete the context struct
    delete context;

    return WRFFT_SUCCESS;
}