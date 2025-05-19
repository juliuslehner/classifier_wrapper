#include "vkfft_wrapper_2d.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
using half_float::half;

// Error-checking macro for CUDA calls.
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

static uint64_t elem_bytes(WrFFTPrecision p)
{
    switch (p) {
        case WRFFT_PREC_SINGLE: return sizeof(float);
        case WRFFT_PREC_DOUBLE: return sizeof(double);
        case WRFFT_PREC_HALF:   return sizeof(half);
    }
    return 0;
}

// -------------------------------------------------------------------
// vkFFT 2D File I/O Functions
// -------------------------------------------------------------------
// void vkfft2d_read_binary(const std::string& filename, PrecType* input, int n) {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open file: " + filename);
//     }
//     // Read interleaved binary data: real, imag, real, imag, etc.
//     double real, imag;
//     for (int i = 0; i < n; ++i) {
//         file.read(reinterpret_cast<char*>(&real), sizeof(double));
//         file.read(reinterpret_cast<char*>(&imag), sizeof(double));
//         if (!file) {
//             throw std::runtime_error("Unexpected end of file while reading: " + filename);
//         }
//         input[2 * i] = static_cast<PrecType>(real);
//         input[2 * i + 1] = static_cast<PrecType>(imag);
//     }
//     file.close();
// }

// void vkfft2d_write_binary(const std::string& filepath, PrecType* output, int n) {
//     std::ofstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open file for writing: " + filepath);
//     }
//     for (int i = 0; i < n; ++i) {
//         double real = static_cast<double>(output[2 * i]);
//         double imag = static_cast<double>(output[2 * i + 1]);
//         file.write(reinterpret_cast<const char*>(&real), sizeof(double));
//         file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
//     }
//     file.close();
// }

// -------------------------------------------------------------------
// vkFFT 2D Initialization Function
// -------------------------------------------------------------------
WrFFTErrors vkfft2d_initialize(Vkfft2DContext** contextOut) {
    if (contextOut == nullptr) return WRFFT_ERROR_INVALID_INPUT;
    Vkfft2DContext* context = new Vkfft2DContext;

    // Assume sample_id is 0 for this example.
    uint64_t sample_id = 0;
    VkGPU vkGPU_base = {};
    VkGPU* vkGPU = &vkGPU_base;
    vkGPU->device_id = 0;
    #if(VKFFT_BACKEND==0)
        VkResult res = VK_SUCCESS;
        //create instance - a connection between the application and the Vulkan library 
        res = createInstance(vkGPU, sample_id);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        //set up the debugging messenger 
        res = setupDebugMessenger(vkGPU);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        //check if there are GPUs that support Vulkan and select one
        res = findPhysicalDevice(vkGPU);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        //create logical device representation
        res = createDevice(vkGPU, sample_id);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        //create fence for synchronization 
        res = createFence(vkGPU);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        //create a place, command buffer memory is allocated from
        res = createCommandPool(vkGPU);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
        vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

        glslang_initialize_process();//compiler can be initialized before VkFFT
    #elif(VKFFT_BACKEND==1)
        CUresult res = CUDA_SUCCESS;
        cudaError_t res2 = cudaSuccess;
        res = cuInit(0);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res2 = cudaSetDevice((int)vkGPU->device_id);
        if (res2 != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
    #elif(VKFFT_BACKEND==2)
        hipError_t res = hipSuccess;
        res = hipInit(0);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = hipSetDevice((int)vkGPU->device_id);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = hipDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = hipCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
        if (res != 0) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
    #elif(VKFFT_BACKEND==3)
        cl_int res = CL_SUCCESS;
        cl_uint numPlatforms;
        res = clGetPlatformIDs(0, 0, &numPlatforms);
        if (res != CL_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
        if (!platforms) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = clGetPlatformIDs(numPlatforms, platforms, 0);
        if (res != CL_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        uint64_t k = 0;
        for (uint64_t j = 0; j < numPlatforms; j++) {
            cl_uint numDevices;
            res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
            cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
            if (!deviceList) {
                delete context;
                return WRFFT_ERROR_LIBRARY_FAILURE;
            }
            res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
            if (res != CL_SUCCESS) {
                delete context;
                return WRFFT_ERROR_LIBRARY_FAILURE;
            }
            for (uint64_t i = 0; i < numDevices; i++) {
                if (k == vkGPU->device_id) {
                    vkGPU->platform = platforms[j];
                    vkGPU->device = deviceList[i];
                    vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
                    if (res != CL_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
                    if (res != CL_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    vkGPU->commandQueue = commandQueue;
                    i=numDevices;
                    j=numPlatforms;
                }
                else {
                    k++;
                }
            }
            free(deviceList);
        }
        free(platforms);
    #elif(VKFFT_BACKEND==4)
        ze_result_t res = ZE_RESULT_SUCCESS;
        res = zeInit(0);
        if (res != ZE_RESULT_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        uint32_t numDrivers = 0;
        res = zeDriverGet(&numDrivers, 0);
        if (res != ZE_RESULT_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(numDrivers * sizeof(ze_driver_handle_t));
        if (!drivers) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        res = zeDriverGet(&numDrivers, drivers);
        if (res != ZE_RESULT_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        uint64_t k = 0;
        for (uint64_t j = 0; j < numDrivers; j++) {
            uint32_t numDevices = 0;
            res = zeDeviceGet(drivers[j], &numDevices, nullptr);
            if (res != ZE_RESULT_SUCCESS) {
                delete context;
                return WRFFT_ERROR_LIBRARY_FAILURE;
            }
            ze_device_handle_t* deviceList = (ze_device_handle_t*)malloc(numDevices * sizeof(ze_device_handle_t));
            if (!deviceList) {
                delete context;
                return WRFFT_ERROR_LIBRARY_FAILURE;
            }
            res = zeDeviceGet(drivers[j], &numDevices, deviceList);
            if (res != ZE_RESULT_SUCCESS) {
                delete context;
                return WRFFT_ERROR_LIBRARY_FAILURE;
            }
            for (uint64_t i = 0; i < numDevices; i++) {
                if (k == vkGPU->device_id) {
                    vkGPU->driver = drivers[j];
                    vkGPU->device = deviceList[i];
                    ze_context_desc_t contextDescription = {};
                    contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
                    res = zeContextCreate(vkGPU->driver, &contextDescription, &vkGPU->context);
                    if (res != ZE_RESULT_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }

                    uint32_t queueGroupCount = 0;
                    res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, 0);
                    if (res != ZE_RESULT_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)malloc(queueGroupCount * sizeof(ze_command_queue_group_properties_t));
                    if (!cmdqueueGroupProperties) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, cmdqueueGroupProperties);
                    if (res != ZE_RESULT_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    uint32_t commandQueueID = -1;
                    for (uint32_t i = 0; i < queueGroupCount; ++i) {
                        if ((cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && (cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
                            commandQueueID = i;
                            break;
                        }
                    }
                    if (commandQueueID == -1) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
                    vkGPU->commandQueueID = commandQueueID;
                    ze_command_queue_desc_t commandQueueDescription = {};
                    commandQueueDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
                    commandQueueDescription.ordinal = commandQueueID;
                    commandQueueDescription.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
                    commandQueueDescription.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
                    res = zeCommandQueueCreate(vkGPU->context, vkGPU->device, &commandQueueDescription, &vkGPU->commandQueue);
                    if (res != ZE_RESULT_SUCCESS) {
                        delete context;
                        return WRFFT_ERROR_LIBRARY_FAILURE;
                    }
                    free(cmdqueueGroupProperties);
                    i=numDevices;
                    j=numDrivers;
                }
                else {
                    k++;
                }
            }

            free(deviceList);
        }
        free(drivers);
    #elif(VKFFT_BACKEND==5)
        NS::Array* devices = MTL::CopyAllDevices();
        MTL::Device* device = (MTL::Device*)devices->object(vkGPU->device_id);
        vkGPU->device = device;
        MTL::CommandQueue* queue = device->newCommandQueue();
        vkGPU->queue = queue;
    #endif
    uint64_t isCompilerInitialized = 1;
    context->gpu = *vkGPU;
    *contextOut = context;
    return WRFFT_SUCCESS;
}

// -------------------------------------------------------------------
// vkFFT 2D Plan, Execute, and Cleanup Functions
// -------------------------------------------------------------------
// Attention! CPU Buffer needs to be allocated and filled in higher level, then passed within context struct
WrFFTErrors vkfft2d_plan(int nx, int ny, Vkfft2DContext* context, WrFFTPrecision precision) {
    if (!context) return WRFFT_ERROR_INVALID_INPUT;
    context->nx = nx;
    context->ny = ny;
    //zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 2; //FFT dimension
    configuration.size[0] = nx; //FFT size
    configuration.size[1] = ny;
    switch(precision){
        case WRFFT_PREC_DOUBLE:
            configuration.doublePrecision = true;
            break;
        case WRFFT_PREC_HALF:
            configuration.halfPrecision = true;
            break;
        defualt:
            break;
    }
    //Device management + code submission
    #if(VKFFT_BACKEND==5)
        configuration.device = context->gpu.device;
    #else
        configuration.device = &context->gpu.device;
    #endif

    #if(VKFFT_BACKEND==0) //Vulkan API
        configuration.queue = &context->gpu.queue;
        configuration.fence = &context->gpu.fence;
        configuration.commandPool = &context->gpu.commandPool;
        configuration.physicalDevice = &context->gpu.physicalDevice;
        configuration.isCompilerInitialized = isCompilerInitialized;
    #elif(VKFFT_BACKEND==3) //OpenCL API
        configuration.context = &context->gpu.context;
    #elif(VKFFT_BACKEND==4)
        configuration.context = &context->gpu.context;
        configuration.commandQueue = &context->gpu.commandQueue;
        configuration.commandQueueID = context->gpu.commandQueueID;
    #elif(VKFFT_BACKEND==5)
        configuration.queue = context->gpu.queue;
    #endif
    
    // Calculate buffer size
    context->buffer_size = elem_bytes(precision) * 2 * nx * ny;
    configuration.bufferSize  = &context->buffer_size;
    
    VkFFTResult resFFT = VKFFT_SUCCESS;
    // Allocate Memory on GPU
    #if(VKFFT_BACKEND==0)
        VkBuffer buffer = {};
        VkDeviceMemory bufferDeviceMemory = {};
        resFFT = allocateBuffer(&context->gpu, &context->device_buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, context->buffer_size);
        if (resFFT != VKFFT_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        configuration.buffer = &context->device_buffer;
    #elif(VKFFT_BACKEND==1)
        // cuFloatComplex* buffer = 0;
        cudaError_t res2 = cudaSuccess;
        res2 = cudaMallocManaged((void**)&context->device_buffer, context->buffer_size);
        if (res2 != cudaSuccess) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        configuration.buffer = (void**)&context->device_buffer;
    #elif(VKFFT_BACKEND==2)
        hipFloatComplex* buffer = 0;
        res = hipMalloc((void**)&buffer, context->buffer_size);
        if (res != hipSuccess) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        configuration.buffer = (void**)&buffer;
    #elif(VKFFT_BACKEND==3)
        cl_mem buffer = 0;
        buffer = clCreateBuffer(context->gpu.context, CL_MEM_READ_WRITE, context->buffer_size, 0, &res);
        if (res != CL_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        configuration.buffer = &buffer;
    #elif(VKFFT_BACKEND==4)
        void* buffer = 0;
        ze_device_mem_alloc_desc_t device_desc = {};
        device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        res = zeMemAllocDevice(context->gpu.context, &device_desc, context->buffer_size, sizeof(float), context->gpu.device, &buffer);
        if (res != ZE_RESULT_SUCCESS) {
            delete context;
            return WRFFT_ERROR_LIBRARY_FAILURE;
        }
        configuration.buffer = &buffer;
    #elif(VKFFT_BACKEND==5)
        MTL::Buffer* buffer = 0;
        buffer = context->gpu.device->newBuffer(context->buffer_size, MTL::ResourceStorageModePrivate);
        configuration.buffer = &buffer;
    #endif
    configuration.bufferSize = &context->buffer_size;

    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) {
        delete context;
        return WRFFT_ERROR_LIBRARY_FAILURE;
    }
    context->app = app;
    context->configuration = configuration;
    return WRFFT_SUCCESS;
}

WrFFTErrors vkfft2d_execute(Vkfft2DContext* context, void* host_data) {
    if (!context || !host_data) return WRFFT_ERROR_INVALID_INPUT;
    VkFFTResult resFFT = VKFFT_SUCCESS;
    int n = context->nx * context->ny;
    // resFFT = transferDataFromCPU(&context->gpu, host_data, &context->device_buffer, context->buffer_size);
    ComplexData* host = reinterpret_cast<ComplexData*>(host_data);
    switch(context->precision) {
        case WRFFT_PREC_HALF: {
            auto device_buffer = reinterpret_cast<half*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                device_buffer[2*i] = half(static_cast<double>(host[i].x));
                device_buffer[2*i + 1] = half(static_cast<double>(host[i].y));
            }
            break;
        }
        case WRFFT_PREC_SINGLE: {
            auto device_buffer = reinterpret_cast<float*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                device_buffer[2*i] = static_cast<float>(host[i].x);
                device_buffer[2*i + 1] = static_cast<float>(host[i].y);
            }
            break;
        }
        case WRFFT_PREC_DOUBLE: {
            auto device_buffer = reinterpret_cast<double*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                device_buffer[2*i] = static_cast<double>(host[i].x);
                device_buffer[2*i + 1] = static_cast<double>(host[i].y);
            }
            break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = &context->device_buffer;
    resFFT = VkFFTAppend(&context->app, -1, &launchParams);
    if (resFFT != VKFFT_SUCCESS) {
        delete context;
        std::cerr << "VkFFT Append failed" << std::endl;
        return WRFFT_ERROR_LIBRARY_FAILURE;
    }
    std::cout << "VkFFT Appended" << std::endl;

    // Copy the result from device to host
// #if(VKFFT_BACKEND==0)
//     resFFT = transferDataToCPU(&context->gpu, host_data, context->device_buffer, context->buffer_size);
//     if (resFFT != VKFFT_SUCCESS) {
//         delete context;
//         return WRFFT_ERROR_LIBRARY_FAILURE;
//     }
// #else
//     resFFT = transferDataToCPU(&context->gpu, host_data, &context->device_buffer, context->buffer_size);
//     if (resFFT != VKFFT_SUCCESS) {
//         delete context;
//         return WRFFT_ERROR_LIBRARY_FAILURE;
//     }
// #endif
    CUDA_CHECK(cudaDeviceSynchronize());
    switch (context->precision) {
        case WRFFT_PREC_HALF: {
            auto device_buffer = reinterpret_cast<half*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(device_buffer[2*i]);
                host[i].y = static_cast<double>(device_buffer[2*i + 1]);
            }
            break;
        }
        case WRFFT_PREC_SINGLE: {
            auto device_buffer = reinterpret_cast<float*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(device_buffer[2*i]);
                host[i].y = static_cast<double>(device_buffer[2*i + 1]);
            }
            break;
        }
        case WRFFT_PREC_DOUBLE: {
            auto device_buffer = reinterpret_cast<double*>(context->device_buffer);
            for (int i = 0; i < n; ++i) {
                host[i].x = static_cast<double>(device_buffer[2*i]);
                host[i].y = static_cast<double>(device_buffer[2*i + 1]);
            }
            break;
        }
    }
    
    return WRFFT_SUCCESS;
}

WrFFTErrors vkfft2d_cleanup(Vkfft2DContext* context) {
    if (!context) return WRFFT_ERROR_INVALID_INPUT;

    // destroy the VkFFT app
    deleteVkFFT(&context->app);

  #if (VKFFT_BACKEND==1)
    cudaFree(context->device_buffer);
  #elif (VKFFT_BACKEND==0)
    vkDestroyBuffer(context->gpu.device, context->vk_buffer, nullptr);
    vkFreeMemory(context->gpu.device, context->device_memory, nullptr);
  #endif

    // cpu_buffer is owned by caller, so don’t free it here
    delete context;
    return WRFFT_SUCCESS;
}
