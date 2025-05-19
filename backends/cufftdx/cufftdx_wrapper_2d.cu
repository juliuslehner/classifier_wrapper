#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <algorithm>
#include <complex>
#include <thrust/complex.h>
#include <sstream>
#include <fstream>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#include "cufftdx_wrapper_2d.hpp"
#include "cufftdx_common.hpp"
#include "block_io.hpp"
#include "block_io_strided.hpp"
#include "common.hpp"
#include "random.hpp"
#include "fp16_common.hpp"

#ifndef CUDA_ARCH
#error "You must define CUDA_ARCH at compile time (e.g. -DCUDA_ARCH=800)"
#endif

//--------------------------------------------------------------------------------
// The cuFFTDx 2D riff: two kernels (Y then X), plus a small driver
//--------------------------------------------------------------------------------
template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_2d_kernel_y(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(input, thread_data, local_fft_id);
    // Execute FFT
    extern __shared__ unsigned char _smem[];
    auto* shared_mem = reinterpret_cast<complex_type*>(_smem);
    FFT().execute(thread_data, shared_mem, workspace);
    // Save results
    example::io<FFT>::store(thread_data, output, local_fft_id);
}

template<class FFT, unsigned int Stride, bool UseSharedMemoryStridedIO, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__ void fft_2d_kernel_x(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    extern __shared__ unsigned char _smem[];
    auto* shared_mem = reinterpret_cast<complex_type*>(_smem);
    // Local array for thread
    ComplexType thread_data[FFT::storage_size];
    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, shared_mem, local_fft_id);
    } else {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, local_fft_id);
    }
    // Execute FFT
    FFT().execute(thread_data, shared_mem, workspace);
    // Save results
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, shared_mem, output, local_fft_id);
    } else {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, output, local_fft_id);
    }
}

template<class FFTX, class FFTY, bool UseSharedMemoryStridedIO, class T>
void cufftdx_fft_2d(T* input, T* output, cudaStream_t stream) {
    using complex_type                       = typename FFTX::value_type;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Checks that FFTX and FFTY are correctly defined
    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>, cufftdx::precision_of_t<FFTY>>,
                  "FFTY and FFTX must have the same precision");
    static_assert(std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
                  "FFTY and FFTX must operator on the same type");
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>, "");
    // Checks below are not caused by any limitation in cuFFTDx, but rather in the example IO functions.
    static_assert((fft_size_x % FFTY::ffts_per_block == 0),
                  "FFTsPerBlock for FFTX must divide Y dimension as IO doesn't check if a batch is in range");

    // complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    // complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);

    // Set shared memory requirements
    auto error_code = cudaFuncSetAttribute(
        fft_2d_kernel_y<FFTY, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);

    // Shared memory IO for strided kernel may require more memory than FFTX::shared_memory_size.
    // Note: For some fft_size_x and depending on GPU architecture fft_x_shared_memory_smem_io may exceed max shared
    // memory and cudaFuncSetAttribute will fail.
    unsigned int fft_x_shared_memory_smem_io =
        std::max<unsigned>({FFTX::shared_memory_size, FFTX::ffts_per_block * fft_size_x * sizeof(complex_type)});
    unsigned int fft_x_shared_memory =
        UseSharedMemoryStridedIO ? fft_x_shared_memory_smem_io : FFTX::shared_memory_size;
    error_code = cudaFuncSetAttribute(fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      fft_x_shared_memory);
    CUDA_CHECK_AND_EXIT(error_code);

    // Create workspaces for FFTs
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Synchronize device before execution
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Define 2D FFT execution
    const auto grid_fft_size_y = ((fft_size_x + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block);
    const auto grid_fft_size_x = ((fft_size_y + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block);
    auto fft_2d_execution = [&](cudaStream_t stream) {
        fft_2d_kernel_y<FFTY, complex_type><<<grid_fft_size_y, FFTY::block_dim, FFTY::shared_memory_size, stream>>>(
            input, output, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>
            <<<grid_fft_size_x, FFTX::block_dim, fft_x_shared_memory, stream>>>(
                output, output, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Execute
    fft_2d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

//--------------------------------------------------------------------------------
// plan_impl2d + execute_impl2d
//--------------------------------------------------------------------------------

template<unsigned int size_x, unsigned int size_y, unsigned Arch, unsigned int ept_x, unsigned int fpb_x,
         unsigned int ept_y, unsigned int fpb_y, typename ValueType>
WrFFTErrors plan2d_impl(Cufftdx2DContext* context) {
    using namespace cufftdx;
    using FFT_base = decltype(Block() + Type<fft_type::c2c>() + Precision<ValueType>() + SM<Arch>());
    using FFT_with_direction = decltype(FFT_base() + Direction<fft_direction::forward>());
    using fft_y    = decltype(FFT_with_direction() + Size<size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(FFT_with_direction() + Size<size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;
    using complex_type = typename fft::value_type;
    // Host data
    static constexpr size_t flat_fft_size       = size_x * size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);

    complex_type* input;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input, flat_fft_size_bytes));
    complex_type* output;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, flat_fft_size_bytes)); 
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    context->d_input = input;
    context->d_output = output;
    return WRFFT_SUCCESS;
}

template<unsigned int size_x, unsigned int size_y, unsigned Arch, unsigned int ept_x, unsigned int fpb_x,
         unsigned int ept_y, unsigned int fpb_y, typename ValueType>
WrFFTErrors execute2d_impl(void* host_data, Cufftdx2DContext* context) {
    using namespace cufftdx;
    using FFT_base = decltype(Block() + Type<fft_type::c2c>() + Precision<ValueType>() + SM<Arch>());
    using FFT_with_direction = decltype(FFT_base() + Direction<fft_direction::forward>());
    using fft_y    = decltype(FFT_with_direction() + Size<size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(FFT_with_direction() + Size<size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;

    // Copy data from host to device
    using complex_t = typename fft::value_type;
    size_t count = cufftdx::size_of<fft>::value;
    ComplexData* host = reinterpret_cast<ComplexData*>(host_data);
    complex_t* d_input = reinterpret_cast<complex_t*>(context->d_input);
    complex_t* d_output = reinterpret_cast<complex_t*>(context->d_output);
    if constexpr (std::is_same_v<ValueType, float>) {
        for (size_t i = 0; i < count; ++i) {
            d_input[i].x = static_cast<float>(host[i].x);
            d_input[i].y = static_cast<float>(host[i].y);
        }
    } else if constexpr (std::is_same_v<ValueType, double>) {
        for (size_t i = 0; i < count; ++i) {
            d_input[i].x = static_cast<double>(host[i].x);
            d_input[i].y = static_cast<double>(host[i].y);
        }
    } 

    // Set up stream and launch transform
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));
    context->stream = stream;
    // run on default stream 0
    cufftdx_fft_2d<fft_x,fft_y,false>(d_input, d_output, context->stream);

    // Copy data from device to host
    if constexpr (std::is_same_v<ValueType, float>) {
        for (size_t i = 0; i < count; ++i) {
            host[i].x = static_cast<double>(d_output[i].x);
            host[i].y = static_cast<double>(d_output[i].y);
        }
    }
    else if constexpr (std::is_same_v<ValueType, double>) {
        for (size_t i = 0; i < count; ++i) {
            host[i].x = d_output[i].x;
            host[i].y = d_output[i].y;
        }
    }
    return WRFFT_SUCCESS;
}

//--------------------------------------------------------------------------------
// Runtime dispatchers
//--------------------------------------------------------------------------------

WrFFTErrors cufftdx2d_plan(int nx, int ny, Cufftdx2DContext** contextOut, WrFFTPrecision precision) {
    if (!contextOut) return WRFFT_ERROR_INVALID_INPUT;
    Cufftdx2DContext* context = new Cufftdx2DContext;
    context->size_x = nx;
    context->size_y = ny;
    context->precision = precision;
    WrFFTErrors err;
    switch(precision) {
        case WRFFT_PREC_SINGLE:
            switch (nx) {
                case 1024:
                    switch (ny) {
                        case 1024: err = plan2d_impl<1024, 1024, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        case 2048: err = plan2d_impl<1024, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 2048:
                    switch (ny) {
                        case 1024: err = plan2d_impl<2048, 1024, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        case 2048: err = plan2d_impl<2048, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        case 4096: err = plan2d_impl<2048, 4096, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 4096:
                    switch (ny) {
                        case 2048: err = plan2d_impl<4096, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        case 4096: err = plan2d_impl<4096, 4096, CUDA_ARCH, 8, 4, 8, 4, float>(context); break;
                        case 8192: err = plan2d_impl<4096, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 8192:
                    switch (ny) {
                        case 4096: err = plan2d_impl<8192, 4096, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        case 8192: err = plan2d_impl<8192, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        case 16384: err = plan2d_impl<8192, 16384, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 16384:
                    switch (ny) {
                        case 8192: err = plan2d_impl<16384, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        case 16384: err = plan2d_impl<16384, 16384, CUDA_ARCH, 16, 2, 16, 1, float>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                }
                break;
        case WRFFT_PREC_DOUBLE:
            switch (nx) {
                case 1024:
                    switch (ny) {
                        case 1024: err = plan2d_impl<1024, 1024, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        case 2048: err = plan2d_impl<1024, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 2048:
                    switch (ny) {
                        case 1024: err = plan2d_impl<2048, 1024, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        case 2048: err = plan2d_impl<2048, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        case 4096: err = plan2d_impl<2048, 4096, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 4096:
                    switch (ny) {
                        case 2048: err = plan2d_impl<4096, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        case 4096: err = plan2d_impl<4096, 4096, CUDA_ARCH, 8, 4, 8, 4, double>(context); break;
                        case 8192: err = plan2d_impl<4096, 8192, CUDA_ARCH,16 ,2 ,16 ,1 ,double>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 8192:
                    switch (ny) {
                        case 4096: err = plan2d_impl<8192, 4096, CUDA_ARCH, 16, 2, 16, 1, double>(context); break;
                        case 8192: err = plan2d_impl<8192, 8192, CUDA_ARCH, 16, 2, 16, 1, double>(context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
            }
            break;
        default:
            err = WRFFT_ERROR_INVALID_INPUT;
            break;
    }
    
    if (err != WRFFT_SUCCESS) {
        delete context;
        return err;
    }
    *contextOut = context;      
    return WRFFT_SUCCESS;
}

WrFFTErrors cufftdx2d_execute(Cufftdx2DContext* context, void* host_data) {
    if (!context || !host_data) return WRFFT_ERROR_INVALID_INPUT;
    WrFFTErrors err;
    switch(context->precision) {
        case WRFFT_PREC_SINGLE:
            switch (context->size_x) {
                case 1024:
                    switch (context->size_y) {
                        case 1024: err = execute2d_impl<1024, 1024, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        case 2048: err = execute2d_impl<1024, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 2048:
                    switch (context->size_y) {
                        case 1024: err = execute2d_impl<2048, 1024, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        case 2048: err = execute2d_impl<2048, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        case 4096: err = execute2d_impl<2048, 4096, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 4096:
                    switch (context->size_y) {
                        case 2048: err = execute2d_impl<4096, 2048, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        case 4096: err = execute2d_impl<4096, 4096, CUDA_ARCH, 8, 4, 8, 4, float>(host_data, context); break;
                        case 8192: err = execute2d_impl<4096, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 8192:
                    switch (context->size_y) {
                        case 4096: err = execute2d_impl<8192, 4096, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        case 8192: err = execute2d_impl<8192, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        case 16384: err = execute2d_impl<8192, 16384, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 16384:
                    switch (context->size_y) {
                        case 8192: err = execute2d_impl<16384, 8192, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        case 16384: err = execute2d_impl<16384, 16384, CUDA_ARCH, 16, 2, 16, 1, float>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                }
                break;
        case WRFFT_PREC_DOUBLE:
            switch (context->size_x) {
                case 1024:
                    switch (context->size_y) {
                        case 1024: err = execute2d_impl<1024, 1024, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        case 2048: err = execute2d_impl<1024, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 2048:
                    switch (context->size_y) {
                        case 1024: err = execute2d_impl<2048, 1024, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        case 2048: err = execute2d_impl<2048, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        case 4096: err = execute2d_impl<2048, 4096, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 4096:
                    switch (context->size_y) {
                        case 2048: err = execute2d_impl<4096, 2048, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        case 4096: err = execute2d_impl<4096, 4096, CUDA_ARCH, 8, 4, 8, 4, double>(host_data, context); break;
                        case 8192: err = execute2d_impl<4096, 8192, CUDA_ARCH,16 ,2 ,16 ,1 ,double>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
                case 8192:
                    switch (context->size_y) {
                        case 4096: err = execute2d_impl<8192, 4096, CUDA_ARCH, 16, 2, 16, 1, double>(host_data, context); break;
                        case 8192: err = execute2d_impl<8192, 8192, CUDA_ARCH, 16, 2, 16, 1, double>(host_data, context); break;
                        default: err = WRFFT_ERROR_INVALID_INPUT; break;
                    }
                    break;
            }
            break;
        default:
            err = WRFFT_ERROR_INVALID_INPUT;
            break;
    }
}

WrFFTErrors cufftdx2d_cleanup(Cufftdx2DContext* context) {
    if (!context) return WRFFT_ERROR_INVALID_INPUT;
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(context->stream));
    // Free CUDA buffers
    CUDA_CHECK_AND_EXIT(cudaFree(context->d_input));
    CUDA_CHECK_AND_EXIT(cudaFree(context->d_output)); 
    delete context;
    return WRFFT_SUCCESS;
}
